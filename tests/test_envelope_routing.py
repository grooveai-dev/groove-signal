"""Integration test: envelope round-trip via an in-process RelayNode.

Spawns a relay, attaches a fake node ws and a fake consumer ws, and
asserts an envelope sent by the consumer reaches the node and a reply
envelope from the node reaches the consumer.
"""

import asyncio
import socket

import msgpack
import pytest
import websockets

from src.common.protocol import (
    ENVELOPE,
    PIPELINE_CONFIG,
    PROTOCOL_VERSION,
    REGISTER_ACK,
    decode_message,
    encode_message,
    make_envelope,
    make_register_node,
    make_session_init,
)
from src.relay.relay import RelayNode


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
async def test_envelope_round_trip():
    port = _free_port()
    relay = RelayNode(host="127.0.0.1", port=port)
    server_task = asyncio.create_task(relay.start())
    try:
        await asyncio.sleep(0.15)

        node_ws = await websockets.connect(f"ws://127.0.0.1:{port}", max_size=1 << 20)
        await node_ws.send(encode_message(
            make_register_node("node-A", layer_start=0, layer_end=31, capabilities={"ram_mb": 1})
        ))
        ack = await _recv(node_ws)
        assert ack["type"] == REGISTER_ACK and ack["accepted"] is True

        consumer_ws = await websockets.connect(f"ws://127.0.0.1:{port}", max_size=1 << 20)
        await consumer_ws.send(encode_message(
            make_session_init("sess-x", "test-model", 0, -1, {})
        ))
        cfg = await _recv(consumer_ws)
        assert cfg["type"] == PIPELINE_CONFIG
        assert cfg["protocol_version"] == PROTOCOL_VERSION
        stream_id = cfg["stream_id"]
        assert cfg["nodes"] == [{"node_id": "node-A", "layer_start": 0, "layer_end": 31}]

        c2n_payload = msgpack.packb({"hello": "from-consumer"}, use_bin_type=True)
        await consumer_ws.send(encode_message(
            make_envelope(stream_id, c2n_payload, target_node_id="node-A")
        ))
        node_env = await _recv(node_ws)
        assert node_env["type"] == ENVELOPE
        assert node_env["stream_id"] == stream_id
        assert node_env["target_node_id"] == "node-A"
        assert msgpack.unpackb(node_env["payload"], raw=False) == {"hello": "from-consumer"}

        n2c_payload = msgpack.packb({"hello": "from-node"}, use_bin_type=True)
        await node_ws.send(encode_message(
            make_envelope(stream_id, n2c_payload, target_node_id=None)
        ))
        consumer_env = await _recv(consumer_ws)
        assert consumer_env["type"] == ENVELOPE
        assert consumer_env["stream_id"] == stream_id
        assert consumer_env["target_node_id"] is None
        assert msgpack.unpackb(consumer_env["payload"], raw=False) == {"hello": "from-node"}

        await consumer_ws.close()
        await node_ws.close()
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
