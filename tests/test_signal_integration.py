"""End-to-end integration tests for the signal server.

These tests exercise the public WebSocket surface without relying on
real sockets: the SignalServer's message handlers accept any object with
`.send(bytes)`, `.recv()`, `.close()`, and `remote_address`, so a thin
in-memory fake is sufficient.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from src.common.protocol import (
    ERROR,
    PIPELINE_CONFIG,
    PROTOCOL_VERSION,
    SESSION_INIT,
    SIGNAL_MATCH,
    SIGNAL_ACK,
    decode_message,
    encode_message,
    make_session_init,
    make_signal_heartbeat,
    make_signal_query,
    make_signal_register,
)
from src.signal.server import SignalServer


MODEL = "Qwen/Qwen2.5-0.5B"


# ---------------------------------------------------------------------------
# Fake websocket
# ---------------------------------------------------------------------------
class _FakeState:
    def __init__(self, name: str = "OPEN"):
        self.name = name


class FakeWebSocket:
    """Minimal fake: a pair of asyncio.Queues + remote_address."""

    def __init__(self, ip: str = "127.0.0.1"):
        self.incoming: asyncio.Queue = asyncio.Queue()  # frames peer sends us
        self.outgoing: list[bytes] = []                  # frames we send to peer
        self.remote_address = (ip, 40000)
        self.closed = False
        self.state = _FakeState("OPEN")

    async def send(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.outgoing.append(data)

    async def recv(self):
        return await self.incoming.get()

    async def close(self, *a, **kw):
        self.closed = True
        self.state = _FakeState("CLOSED")
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
NODE_ID_1 = "0x" + "a1" * 20  # 0xa1a1...a1 (valid 40-hex)
NODE_ID_2 = "0x" + "b2" * 20
NODE_ID_3 = "0x" + "c3" * 20
NODE_ID_4 = "0x" + "d4" * 20
NODE_ID_5 = "0x" + "e5" * 20
NODE_ID_NEAR = "0x" + "11" * 20
NODE_ID_FAR = "0x" + "22" * 20
NODE_ID_GONE = "0x" + "33" * 20
NODE_ID_ACK = "0x" + "44" * 20
NODE_ID_HB = "0x" + "55" * 20


@pytest.mark.asyncio
async def test_register_then_query_returns_match():
    server = SignalServer(require_auth=False)

    ws, task = await _register_node_on_server(
        server, NODE_ID_1,
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
    assert reply["nodes"][0]["node_id"] == NODE_ID_1
    assert reply["nodes"][0]["score"] > 0.5

    await _stop_node(ws, task)


@pytest.mark.asyncio
async def test_multi_node_query_is_ranked_by_score():
    server = SignalServer(require_auth=False)
    ws1, t1 = await _register_node_on_server(
        server, NODE_ID_NEAR,
        capabilities={"device": "cuda", "vram_mb": 24000, "ram_mb": 32000},
        location={"lat": 40.7, "lon": -74.0},
        layer_start=0, layer_end=23,
    )
    ws2, t2 = await _register_node_on_server(
        server, NODE_ID_FAR,
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
    assert ids == [NODE_ID_NEAR, NODE_ID_FAR], f"expected 'near' first, got {ids}"

    await _stop_node(ws1, t1)
    await _stop_node(ws2, t2)


@pytest.mark.asyncio
async def test_node_disconnect_removes_from_query_results():
    server = SignalServer(require_auth=False)
    ws, task = await _register_node_on_server(
        server, NODE_ID_GONE,
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
    server = SignalServer(signal_id="sig-test-abc", require_auth=False)
    ws = FakeWebSocket()
    first = make_signal_register(
        node_id=NODE_ID_ACK,
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
    server = SignalServer(require_auth=False)
    ws, task = await _register_node_on_server(
        server, NODE_ID_HB,
        capabilities={"device": "cuda", "vram_mb": 8000},
        layer_start=0, layer_end=23,
    )

    hb = make_signal_heartbeat(
        node_id=NODE_ID_HB, active_sessions=3, load=0.4,
        capabilities={"device": "cuda", "vram_mb": 8000, "ram_mb": 32000},
    )
    await ws.incoming.put(encode_message(hb))
    # Give the message loop a tick to process.
    await asyncio.sleep(0.05)

    rec = server.registry.get_node(NODE_ID_HB)
    assert rec.active_sessions == 3
    assert rec.load == pytest.approx(0.4)

    await _stop_node(ws, task)


# ---------------------------------------------------------------------------
# Grace period / NODE_GONE tests
# ---------------------------------------------------------------------------

async def _open_consumer_session(server: SignalServer, model: str = MODEL) -> FakeWebSocket:
    """Open a consumer relay session and return the consumer's FakeWebSocket."""
    consumer_ws = FakeWebSocket()
    init = make_session_init(
        session_id=uuid.uuid4().hex,
        model_name=model,
        layer_start=0,
        layer_end=-1,
        config={},
    )
    # _handle_session_init enters the envelope loop which blocks, so run it
    # as a background task and wait just for the PIPELINE_CONFIG reply.
    task = asyncio.create_task(server._handle_session_init(consumer_ws, init))
    for _ in range(100):
        if consumer_ws.outgoing:
            break
        await asyncio.sleep(0.01)
    assert consumer_ws.outgoing, "no PIPELINE_CONFIG received"
    first = decode_message(consumer_ws.outgoing[0])
    assert first["type"] == PIPELINE_CONFIG, f"expected PIPELINE_CONFIG, got {first['type']}"
    return consumer_ws


@pytest.mark.asyncio
async def test_node_disconnect_triggers_delayed_teardown():
    """After a node disconnects, NODE_GONE must NOT be sent immediately;
    it should only arrive after the grace period expires."""
    server = SignalServer(require_auth=False)
    server.teardown_grace_period = 0.5

    ws1, t1 = await _register_node_on_server(
        server, "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        capabilities={"device": "cuda", "vram_mb": 24000, "ram_mb": 32000},
        layer_start=0, layer_end=12,
    )
    ws2, t2 = await _register_node_on_server(
        server, "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABB",
        capabilities={"device": "cuda", "vram_mb": 24000, "ram_mb": 32000},
        layer_start=12, layer_end=24,
    )

    consumer_ws = await _open_consumer_session(server)
    # Clear the PIPELINE_CONFIG from outgoing so we can check for NODE_GONE.
    consumer_ws.outgoing.clear()

    # Disconnect node 1.
    await ws1.incoming.put(ConnectionClosedSentinel)
    await asyncio.sleep(0.1)
    await t1

    # NODE_GONE should NOT have been sent yet (within grace period).
    node_gone_msgs = [
        decode_message(m) for m in consumer_ws.outgoing
        if decode_message(m).get("code") == "NODE_GONE"
    ]
    assert len(node_gone_msgs) == 0, "NODE_GONE sent before grace period expired"

    # Wait past the grace period.
    await asyncio.sleep(0.6)

    node_gone_msgs = [
        decode_message(m) for m in consumer_ws.outgoing
        if decode_message(m).get("code") == "NODE_GONE"
    ]
    assert len(node_gone_msgs) == 1, (
        f"Expected exactly 1 NODE_GONE after grace period, got {len(node_gone_msgs)}"
    )

    await _stop_node(ws2, t2)


@pytest.mark.asyncio
async def test_node_reconnect_within_grace_period_preserves_stream():
    """If a node reconnects within the grace period, the consumer stream
    must NOT be torn down."""
    server = SignalServer(require_auth=False)
    server.teardown_grace_period = 2.0

    node_id = "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    ws1, t1 = await _register_node_on_server(
        server, node_id,
        capabilities={"device": "cuda", "vram_mb": 24000, "ram_mb": 32000},
        layer_start=0, layer_end=24,
    )

    consumer_ws = await _open_consumer_session(server)
    consumer_ws.outgoing.clear()

    # Disconnect the node.
    await ws1.incoming.put(ConnectionClosedSentinel)
    await asyncio.sleep(0.1)
    await t1

    # Reconnect the same node within grace period.
    ws2, t2 = await _register_node_on_server(
        server, node_id,
        capabilities={"device": "cuda", "vram_mb": 24000, "ram_mb": 32000},
        layer_start=0, layer_end=24,
    )

    # Wait past what would have been the grace period.
    await asyncio.sleep(2.5)

    # Consumer should NOT have received NODE_GONE.
    node_gone_msgs = [
        decode_message(m) for m in consumer_ws.outgoing
        if decode_message(m).get("code") == "NODE_GONE"
    ]
    assert len(node_gone_msgs) == 0, (
        "NODE_GONE was sent even though node reconnected within grace period"
    )

    # Stream should still be tracked.
    assert len(server.streams) >= 1, "Consumer stream was removed despite node reconnect"

    await _stop_node(ws2, t2)


@pytest.mark.asyncio
async def test_pipeline_rejects_stale_heartbeat_node():
    """A node whose last_heartbeat is >30s old should be rejected from
    new consumer pipelines."""
    server = SignalServer(require_auth=False)

    ws, task = await _register_node_on_server(
        server, "0xCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        capabilities={"device": "cuda", "vram_mb": 24000, "ram_mb": 32000},
        layer_start=0, layer_end=24,
    )

    # Artificially age the heartbeat.
    rec = server.registry.get_node("0xCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
    rec.last_heartbeat = time.time() - 60.0

    consumer_ws = FakeWebSocket()
    init = make_session_init(
        session_id=uuid.uuid4().hex,
        model_name=MODEL,
        layer_start=0,
        layer_end=-1,
        config={},
    )
    await server._handle_session_init(consumer_ws, init)

    assert consumer_ws.outgoing, "no response received"
    reply = decode_message(consumer_ws.outgoing[0])
    assert reply["type"] == ERROR, f"expected ERROR, got {reply['type']}"
    assert reply["code"] == "NO_NODES"

    await _stop_node(ws, task)
