"""End-to-end integration tests for the signal server.

These tests exercise the public WebSocket surface without relying on
real sockets: the SignalServer's message handlers accept any object with
`.send(bytes)`, `.recv()`, `.close()`, and `remote_address`, so a thin
in-memory fake is sufficient.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from src.common.protocol import (
    PROTOCOL_VERSION,
    SIGNAL_MATCH,
    SIGNAL_ACK,
    decode_message,
    encode_message,
    make_signal_heartbeat,
    make_signal_query,
    make_signal_register,
)
from src.signal.server import SignalServer


MODEL = "Qwen/Qwen2.5-0.5B"


# ---------------------------------------------------------------------------
# Fake websocket
# ---------------------------------------------------------------------------
class FakeWebSocket:
    """Minimal fake: a pair of asyncio.Queues + remote_address."""

    def __init__(self, ip: str = "127.0.0.1"):
        self.incoming: asyncio.Queue = asyncio.Queue()  # frames peer sends us
        self.outgoing: list[bytes] = []                  # frames we send to peer
        self.remote_address = (ip, 40000)
        self.closed = False

    async def send(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.outgoing.append(data)

    async def recv(self):
        return await self.incoming.get()

    async def close(self, *a, **kw):
        self.closed = True
        # Unblock any pending recv() with an EOF marker.
        await self.incoming.put(ConnectionClosedSentinel)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is ConnectionClosedSentinel:
            raise StopAsyncIteration
        return item


class ConnectionClosedSentinel:
    pass


async def _register_node_on_server(
    server: SignalServer,
    node_id: str,
    capabilities: dict,
    location: dict | None = None,
    models_supported: list[str] | None = None,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> tuple[FakeWebSocket, asyncio.Task]:
    """Spawn a handler task for a node connection; return (ws, task).

    Directly sets the layer assignment after registration rather than
    going through the scheduler — the scheduler flow is covered by
    groove-network's own tests; integration here focuses on signal routing.
    """
    ws = FakeWebSocket()
    first = make_signal_register(
        node_id=node_id,
        capabilities=capabilities,
        location=location,
        models_supported=models_supported or [MODEL],
    )
    task = asyncio.create_task(server._handle_node(ws, first))

    # Wait until the node shows up in the registry (handler has processed hello).
    for _ in range(50):
        rec = server.registry.get_node(node_id)
        if rec is not None:
            break
        await asyncio.sleep(0.01)
    assert server.registry.get_node(node_id) is not None, \
        f"node {node_id} did not appear in registry"

    # Force the node straight to 'active' so matcher sees it — bypass the
    # scheduler, which would otherwise send ASSIGN_LAYERS and wait for an ack
    # we don't simulate here.
    rec = server.registry.get_node(node_id)
    rec.assignment_status = "active"
    rec.assigned_model = MODEL
    if layer_start is not None:
        rec.layer_start = layer_start
    if layer_end is not None:
        rec.layer_end = layer_end
    return ws, task


async def _stop_node(ws: FakeWebSocket, task: asyncio.Task):
    await ws.incoming.put(ConnectionClosedSentinel)
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()


# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_register_then_query_returns_match():
    server = SignalServer()

    ws, task = await _register_node_on_server(
        server, "0xnode1",
        capabilities={"device": "cuda", "vram_mb": 24000, "ram_mb": 32000},
        location={"lat": 40.7, "lon": -74.0},
        layer_start=0, layer_end=23,
    )

    consumer_ws = FakeWebSocket()
    query = make_signal_query(
        session_id=uuid.uuid4().hex,
        model_name=MODEL,
        consumer_location={"lat": 40.7, "lon": -74.0},
    )
    await server._handle_signal_query(consumer_ws, query)

    assert len(consumer_ws.outgoing) == 1
    reply = decode_message(consumer_ws.outgoing[0])
    assert reply["type"] == SIGNAL_MATCH
    assert len(reply["nodes"]) == 1
    assert reply["nodes"][0]["node_id"] == "0xnode1"
    assert reply["nodes"][0]["score"] > 0.5

    await _stop_node(ws, task)


@pytest.mark.asyncio
async def test_multi_node_query_is_ranked_by_score():
    server = SignalServer()
    ws1, t1 = await _register_node_on_server(
        server, "near",
        capabilities={"device": "cuda", "vram_mb": 24000, "ram_mb": 32000},
        location={"lat": 40.7, "lon": -74.0},
        layer_start=0, layer_end=23,
    )
    ws2, t2 = await _register_node_on_server(
        server, "far",
        capabilities={"device": "cpu", "vram_mb": 0, "ram_mb": 16000},
        location={"lat": -33.86, "lon": 151.21},
        layer_start=0, layer_end=23,
    )

    consumer_ws = FakeWebSocket()
    query = make_signal_query(
        session_id=uuid.uuid4().hex,
        model_name=MODEL,
        consumer_location={"lat": 40.7, "lon": -74.0},
    )
    await server._handle_signal_query(consumer_ws, query)
    reply = decode_message(consumer_ws.outgoing[0])
    ids = [n["node_id"] for n in reply["nodes"]]
    assert ids == ["near", "far"], f"expected 'near' first, got {ids}"

    await _stop_node(ws1, t1)
    await _stop_node(ws2, t2)


@pytest.mark.asyncio
async def test_node_disconnect_removes_from_query_results():
    server = SignalServer()
    ws, task = await _register_node_on_server(
        server, "0xgone",
        capabilities={"device": "cuda", "vram_mb": 24000},
        layer_start=0, layer_end=23,
    )

    await _stop_node(ws, task)
    # Give the server a chance to run its finalizer.
    await asyncio.sleep(0.05)

    consumer_ws = FakeWebSocket()
    query = make_signal_query(
        session_id=uuid.uuid4().hex,
        model_name=MODEL,
        consumer_location=None,
    )
    await server._handle_signal_query(consumer_ws, query)
    reply = decode_message(consumer_ws.outgoing[0])
    assert reply["nodes"] == []


@pytest.mark.asyncio
async def test_register_acknowledges_with_signal_id():
    server = SignalServer(signal_id="sig-test-abc")
    ws = FakeWebSocket()
    first = make_signal_register(
        node_id="0xabc",
        capabilities={"device": "cpu", "ram_mb": 8000},
        models_supported=[MODEL],
    )
    task = asyncio.create_task(server._handle_node(ws, first))
    # Wait for the ack to land in outgoing.
    for _ in range(50):
        if ws.outgoing:
            break
        await asyncio.sleep(0.01)
    assert ws.outgoing, "no SIGNAL_ACK received"
    ack = decode_message(ws.outgoing[0])
    assert ack["type"] == SIGNAL_ACK
    assert ack["accepted"] is True
    assert ack["signal_id"] == "sig-test-abc"

    await _stop_node(ws, task)


@pytest.mark.asyncio
async def test_heartbeat_updates_active_sessions():
    server = SignalServer()
    ws, task = await _register_node_on_server(
        server, "0xh",
        capabilities={"device": "cuda", "vram_mb": 8000},
        layer_start=0, layer_end=23,
    )

    hb = make_signal_heartbeat(
        node_id="0xh", active_sessions=3, load=0.4,
        capabilities={"device": "cuda", "vram_mb": 8000, "ram_mb": 32000},
    )
    await ws.incoming.put(encode_message(hb))
    # Give the message loop a tick to process.
    await asyncio.sleep(0.05)

    rec = server.registry.get_node("0xh")
    assert rec.active_sessions == 3
    assert rec.load == pytest.approx(0.4)

    await _stop_node(ws, task)
