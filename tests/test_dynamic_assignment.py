"""Tests for M2 dynamic layer assignment on ComputeNodeServer.

These tests exercise the node in isolation: they patch load_model_shard so
the heavy HF model never actually loads, push ASSIGN_LAYERS / REBALANCE
frames through the node's receive loop, and assert the ACKs and state
transitions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("torch")
import msgpack

from src.common.protocol import (
    ENVELOPE,
    decode_message,
    encode_message,
    make_envelope,
)
from src.node.server import (
    ASSIGN_LAYERS,
    ASSIGNMENT_ACK,
    REBALANCE,
    ComputeNodeServer,
    _capabilities,
)


# ----- Fake websocket ----------------------------------------------------------

class _FakeWS:
    """In-memory websocket double. recv() blocks on a queue; send() records."""

    def __init__(self):
        self._inbound: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.sent.append(decode_message(data))

    def push(self, msg: dict) -> None:
        self._inbound.put_nowait(encode_message(msg))

    def close_inbound(self) -> None:
        self._inbound.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        raw = await self._inbound.get()
        if raw is None:
            raise StopAsyncIteration
        return raw


def _fake_shard(layer_start: int, layer_end: int) -> dict:
    return {
        "layers": [],
        "embed_tokens": None,
        "norm": None,
        "lm_head": None,
        "rotary_emb": None,
        "config": MagicMock(),
        "layer_start": layer_start,
        "layer_end": layer_end,
        "total_layers": 32,
    }


def _make_server(**overrides) -> ComputeNodeServer:
    kwargs = dict(
        model_name=None,
        layer_start=None,
        layer_end=None,
        device="cpu",
        node_id="0x" + "a" * 40,
        legacy_mode=False,
    )
    kwargs.update(overrides)
    return ComputeNodeServer(**kwargs)


async def _drain(ws: _FakeWS, server: ComputeNodeServer):
    ws.close_inbound()
    await server._receive_loop(ws)


def _next_ack(ws: _FakeWS) -> dict:
    for msg in ws.sent:
        if msg.get("type") == ASSIGNMENT_ACK:
            return msg
    raise AssertionError(f"no ASSIGNMENT_ACK in sent frames: {[m.get('type') for m in ws.sent]}")


# ----- Tests -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_assign_layers_loads_shard_and_acks():
    server = _make_server()
    ws = _FakeWS()

    loader = AsyncMock(return_value=_fake_shard(0, 15))
    with patch.object(server, "_load_shard_async", loader):
        ws.push({
            "type": ASSIGN_LAYERS,
            "model_name": "Qwen/Qwen2.5-0.5B",
            "layer_start": 0,
            "layer_end": 15,
        })
        await _drain(ws, server)

    loader.assert_called_once()
    assert server.shard is not None
    assert server.layer_start == 0
    assert server.layer_end == 15
    assert server.model_name == "Qwen/Qwen2.5-0.5B"

    ack = _next_ack(ws)
    assert ack["accepted"] is True
    assert ack["layer_start"] == 0
    assert ack["layer_end"] == 15
    assert ack["node_id"] == server.node_id
    assert ack["load_time_ms"] >= 0.0


@pytest.mark.asyncio
async def test_assign_layers_rejects_on_load_error():
    server = _make_server()
    ws = _FakeWS()

    with patch.object(
        server, "_load_shard_async",
        AsyncMock(side_effect=MemoryError("not enough RAM")),
    ):
        ws.push({
            "type": ASSIGN_LAYERS,
            "model_name": "Qwen/Qwen2.5-0.5B",
            "layer_start": 0,
            "layer_end": 15,
        })
        await _drain(ws, server)

    assert server.shard is None
    ack = _next_ack(ws)
    assert ack["accepted"] is False
    assert "not enough RAM" in ack["reason"]


@pytest.mark.asyncio
async def test_rebalance_swaps_shard():
    server = _make_server()
    server.shard = _fake_shard(0, 15)
    server.model_name = "Qwen/Qwen2.5-0.5B"
    server.layer_start = 0
    server.layer_end = 15
    ws = _FakeWS()

    loader = AsyncMock(return_value=_fake_shard(16, 31))
    with patch.object(server, "_load_shard_async", loader):
        ws.push({
            "type": REBALANCE,
            "node_id": server.node_id,
            "model_name": "Qwen/Qwen2.5-0.5B",
            "new_layer_start": 16,
            "new_layer_end": 31,
        })
        await _drain(ws, server)

    loader.assert_called_once()
    assert server.layer_start == 16
    assert server.layer_end == 31

    ack = _next_ack(ws)
    assert ack["accepted"] is True
    assert ack["layer_start"] == 16
    assert ack["layer_end"] == 31


@pytest.mark.asyncio
async def test_legacy_mode_rejects_assign_layers():
    server = _make_server(
        model_name="Qwen/Qwen2.5-0.5B",
        layer_start=0,
        layer_end=15,
        legacy_mode=True,
    )
    server.shard = _fake_shard(0, 15)
    ws = _FakeWS()

    with patch("src.node.server.load_model_shard") as loader:
        ws.push({
            "type": ASSIGN_LAYERS,
            "model_name": "Qwen/Qwen2.5-0.5B",
            "layer_start": 8,
            "layer_end": 24,
        })
        await _drain(ws, server)

    loader.assert_not_called()
    assert server.layer_start == 0
    assert server.layer_end == 15
    ack = _next_ack(ws)
    assert ack["accepted"] is False
    assert "legacy" in ack["reason"].lower()


@pytest.mark.asyncio
async def test_session_init_fails_without_shard():
    """Until ASSIGN_LAYERS lands, SESSION_INIT must error cleanly."""
    server = _make_server()
    ws = _FakeWS()

    payload = msgpack.packb(
        {"type": "session_init", "session_id": "s1", "config": {}},
        use_bin_type=True,
    )
    ws.push(make_envelope("stream-1", payload, target_node_id=server.node_id))
    await _drain(ws, server)

    outbound = [m for m in ws.sent if m.get("type") == ENVELOPE]
    assert len(outbound) == 1
    inner = msgpack.unpackb(outbound[0]["payload"], raw=False)
    assert inner["type"] == "error"
    assert inner["code"] == 409


@pytest.mark.asyncio
async def test_capabilities_reports_enhanced_fields():
    caps = _capabilities(
        model_name="Qwen/Qwen2.5-0.5B",
        device="cpu",
        max_context=4096,
        loaded_layers=(0, 15),
        model_preferences=["Qwen/Qwen2.5-0.5B"],
    )
    for key in (
        "ram_mb",
        "vram_mb",
        "cpu_cores",
        "gpu_model",
        "max_context_length",
        "bandwidth_mbps",
        "model_preferences",
        "protocol_version",
    ):
        assert key in caps, f"missing capability {key}"
    assert caps["max_context_length"] == 4096
    assert caps["cpu_cores"] >= 1
    assert caps["layer_start"] == 0
    assert caps["layer_end"] == 15
    assert caps["model_preferences"] == ["Qwen/Qwen2.5-0.5B"]


def test_node_id_derived_from_identity():
    from src.node.identity import generate_keypair

    ident = generate_keypair()
    server = _make_server(node_id=None, identity=ident)
    assert server.node_id == ident["address"].lower()
    assert server.node_id.startswith("0x")
    assert len(server.node_id) == 42


def test_legacy_node_id_still_works_when_passed_explicitly():
    server = _make_server(node_id="node-L0-15-999")
    assert server.node_id == "node-L0-15-999"
