"""End-to-end: dynamic node registration triggers relay scheduler → ASSIGN_LAYERS.

Exercises the integration between relay.RelayNode and the protocol that
ComputeNodeServer speaks: REGISTER_NODE (no layers) → REGISTER_ACK → the
relay's scheduler pushes ASSIGN_LAYERS → the "node" responds with an
ASSIGNMENT_ACK → consumer SESSION_INIT assembles the pipeline.
"""

import asyncio
import socket

import pytest
import websockets

from src.common.protocol import (
    ASSIGN_LAYERS,
    ASSIGNMENT_ACK,
    PIPELINE_CONFIG,
    PROTOCOL_VERSION,
    REBALANCE,
    REGISTER_ACK,
    decode_message,
    encode_message,
    make_assignment_ack,
    make_register_node,
    make_session_init,
)
from src.relay.relay import RelayNode
from src.relay.scheduler import MODEL_REGISTRY


MODEL = next(iter(MODEL_REGISTRY))
TOTAL = MODEL_REGISTRY[MODEL]["total_layers"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _recv(ws) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return decode_message(raw)


@pytest.mark.asyncio
async def test_dynamic_single_node_end_to_end():
    port = _free_port()
    relay = RelayNode(host="127.0.0.1", port=port, require_auth=False)
    server_task = asyncio.create_task(relay.start())
    try:
        await asyncio.sleep(0.15)

        node_ws = await websockets.connect(f"ws://127.0.0.1:{port}", max_size=1 << 20)
        await node_ws.send(encode_message(make_register_node(
            "0x" + "a" * 40,
            layer_start=None,
            layer_end=None,
            capabilities={"ram_mb": 16000, "device": "cpu"},
        )))
        ack = await _recv(node_ws)
        assert ack["type"] == REGISTER_ACK and ack["accepted"] is True

        assign = await _recv(node_ws)
        assert assign["type"] == ASSIGN_LAYERS
        assert assign["model_name"] == MODEL
        assert assign["layer_start"] == 0
        assert assign["layer_end"] == TOTAL
        assert assign["total_layers"] == TOTAL

        await node_ws.send(encode_message(make_assignment_ack(
            node_id="0x" + "a" * 40,
            model_name=MODEL,
            layer_start=0,
            layer_end=TOTAL,
            accepted=True,
            load_time_ms=5,
        )))

        for _ in range(20):
            entry = relay.nodes.get("0x" + "a" * 40)
            if entry is not None and entry.assignment_status == "active":
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("node never reached active after ack")

        consumer_ws = await websockets.connect(f"ws://127.0.0.1:{port}", max_size=1 << 20)
        await consumer_ws.send(encode_message(make_session_init(
            "sess-1", MODEL, 0, -1, {},
        )))
        cfg = await _recv(consumer_ws)
        assert cfg["type"] == PIPELINE_CONFIG
        assert cfg["protocol_version"] == PROTOCOL_VERSION
        assert cfg["nodes"] == [
            {"node_id": "0x" + "a" * 40, "layer_start": 0, "layer_end": TOTAL}
        ]

        await consumer_ws.close()
        await node_ws.close()
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_dynamic_two_nodes_trigger_rebalance():
    """Second node joining causes a REBALANCE to the first node."""
    port = _free_port()
    relay = RelayNode(host="127.0.0.1", port=port, require_auth=False)
    server_task = asyncio.create_task(relay.start())
    try:
        await asyncio.sleep(0.15)

        nid_a = "0x" + "a" * 40
        nid_b = "0x" + "b" * 40

        # A is a slow CPU node (effective 3000 MB → 13 layers of Qwen3-4B).
        # B is a fast GPU node (effective 5500 MB → 25 layers).
        # Together 13+25=38 > 36 → full coverage after rebalance.
        # B being faster causes minimize_hops to reorder, so A's range
        # shifts and it receives a REBALANCE.
        caps_a = {"ram_mb": 6000, "device": "cpu"}
        caps_b = {"vram_mb": 5500, "device": "cuda", "bench_ms_per_layer": 1.0}

        ws_a = await websockets.connect(f"ws://127.0.0.1:{port}", max_size=1 << 20)
        await ws_a.send(encode_message(make_register_node(
            nid_a, layer_start=None, layer_end=None,
            capabilities=caps_a,
        )))
        ack_a = await _recv(ws_a)
        assert ack_a["accepted"]
        assign_a = await _recv(ws_a)
        assert assign_a["type"] == ASSIGN_LAYERS
        await ws_a.send(encode_message(make_assignment_ack(
            nid_a, MODEL, assign_a["layer_start"], assign_a["layer_end"],
            accepted=True, load_time_ms=1,
        )))

        for _ in range(20):
            if relay.nodes[nid_a].assignment_status == "active":
                break
            await asyncio.sleep(0.05)

        ws_b = await websockets.connect(f"ws://127.0.0.1:{port}", max_size=1 << 20)
        await ws_b.send(encode_message(make_register_node(
            nid_b, layer_start=None, layer_end=None,
            capabilities=caps_b,
        )))
        ack_b = await _recv(ws_b)
        assert ack_b["accepted"]

        frames_a: list[dict] = []
        frames_b: list[dict] = []

        async def collect(ws, bucket, n):
            for _ in range(n):
                f = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(f, str):
                    f = f.encode("utf-8")
                bucket.append(decode_message(f))

        async def respond(ws, nid, bucket):
            await collect(ws, bucket, 1)
            frame = bucket[-1]
            if frame["type"] in (ASSIGN_LAYERS, REBALANCE):
                ls = frame.get("layer_start", frame.get("new_layer_start"))
                le = frame.get("layer_end", frame.get("new_layer_end"))
                await ws.send(encode_message(make_assignment_ack(
                    nid, MODEL, ls, le, accepted=True, load_time_ms=1,
                )))

        await asyncio.gather(
            respond(ws_a, nid_a, frames_a),
            respond(ws_b, nid_b, frames_b),
        )

        assert frames_a[0]["type"] == REBALANCE
        assert frames_b[0]["type"] == ASSIGN_LAYERS

        for _ in range(40):
            if (
                relay.nodes[nid_a].assignment_status == "active"
                and relay.nodes[nid_b].assignment_status == "active"
            ):
                break
            await asyncio.sleep(0.05)

        spans = sorted([
            (relay.nodes[nid_a].layer_start, relay.nodes[nid_a].layer_end),
            (relay.nodes[nid_b].layer_start, relay.nodes[nid_b].layer_end),
        ])
        assert spans[0][0] == 0
        assert spans[-1][1] == TOTAL
        assert spans[0][1] == spans[1][0]

        await ws_a.close()
        await ws_b.close()
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
