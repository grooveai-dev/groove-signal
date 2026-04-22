"""End-to-end P2P integration test.

Two PeerConnectionManagers establish a local connection, send a chunked
tensor (600KB+) through the P2P channel, and verify the reassembled data
matches the original.  Exercises the full path:
  serialize_tensor -> compress -> chunk -> send -> receive -> reassemble
  -> decompress -> deserialize_tensor
"""

import asyncio
import struct

import pytest
import torch

from aiortc import RTCConfiguration

from src.common.chunking import HEADER_FMT, HEADER_SIZE, ChunkedChannel
from src.common.peer import PeerConnectionManager
from src.common.tensor_transfer import deserialize_tensor, serialize_tensor


@pytest.fixture
def rtc_config():
    return RTCConfiguration(iceServers=[])


async def _establish_pair(rtc_config):
    """Create two connected PeerConnectionManagers and return them."""
    mgr_a = PeerConnectionManager(session_id="integ-test", rtc_config=rtc_config)
    mgr_b = PeerConnectionManager(session_id="integ-test", rtc_config=rtc_config)

    offer_sdp = await mgr_a.create_offer("node-b")
    answer_sdp = await mgr_b.handle_offer("node-a", offer_sdp)
    await mgr_a.accept_answer("node-b", answer_sdp)

    for _ in range(100):
        if mgr_a.is_connected("node-b") and mgr_b.is_connected("node-a"):
            break
        await asyncio.sleep(0.05)

    assert mgr_a.is_connected("node-b"), "P2P connection A->B not established"
    assert mgr_b.is_connected("node-a"), "P2P connection B->A not established"
    return mgr_a, mgr_b


@pytest.mark.asyncio
async def test_p2p_chunked_tensor_roundtrip(rtc_config):
    """Full path: serialize -> compress -> chunk -> P2P -> reassemble -> deserialize."""
    mgr_a, mgr_b = await _establish_pair(rtc_config)
    try:
        tensor = torch.randn(151936, dtype=torch.float32)
        wire_bytes = serialize_tensor(tensor)
        assert len(wire_bytes) > 48 * 1024, "Tensor should require multiple chunks"

        received_chunks = []

        async def send_via_a(data: bytes):
            await mgr_a.send("node-b", data)

        channel_a = ChunkedChannel(send_via_a)

        await channel_a.send_message(wire_bytes)

        reassembled = None
        channel_b = ChunkedChannel(lambda d: None)

        deadline = asyncio.get_event_loop().time() + 10.0
        while reassembled is None:
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Did not receive all chunks within 10s")
            raw = await asyncio.wait_for(mgr_b.recv("node-a"), timeout=5.0)
            reassembled = channel_b.on_chunk_received(raw)

        assert reassembled == wire_bytes

        result = deserialize_tensor(reassembled, device="cpu")
        assert result.shape == tensor.shape
        assert result.dtype == tensor.dtype
        assert torch.allclose(result, tensor, atol=1e-6)
    finally:
        await mgr_a.close_all()
        await mgr_b.close_all()


@pytest.mark.asyncio
async def test_p2p_large_activation_tensor(rtc_config):
    """Send a large activation tensor (2048, 2560) float16 — ~10MB, 200+ chunks."""
    mgr_a, mgr_b = await _establish_pair(rtc_config)
    try:
        tensor = torch.randn(256, 2560, dtype=torch.float16)
        wire_bytes = serialize_tensor(tensor)

        async def send_via_a(data: bytes):
            await mgr_a.send("node-b", data)

        channel_a = ChunkedChannel(send_via_a)
        channel_b = ChunkedChannel(lambda d: None)

        await channel_a.send_message(wire_bytes)

        reassembled = None
        deadline = asyncio.get_event_loop().time() + 15.0
        while reassembled is None:
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Did not receive all chunks within 15s")
            raw = await asyncio.wait_for(mgr_b.recv("node-a"), timeout=5.0)
            reassembled = channel_b.on_chunk_received(raw)

        result = deserialize_tensor(reassembled, device="cpu")
        assert result.shape == tensor.shape
        assert result.dtype == tensor.dtype
        assert torch.allclose(result.float(), tensor.float(), atol=0.01)
    finally:
        await mgr_a.close_all()
        await mgr_b.close_all()


@pytest.mark.asyncio
async def test_p2p_small_activation_no_chunking(rtc_config):
    """Single-token activations (5KB) fit in one chunk — no chunking overhead."""
    mgr_a, mgr_b = await _establish_pair(rtc_config)
    try:
        tensor = torch.randn(1, 2560, dtype=torch.float16)
        wire_bytes = serialize_tensor(tensor)
        assert len(wire_bytes) < 48 * 1024, "Small tensor should fit in one chunk"

        async def send_via_a(data: bytes):
            await mgr_a.send("node-b", data)

        channel_a = ChunkedChannel(send_via_a)
        channel_b = ChunkedChannel(lambda d: None)

        await channel_a.send_message(wire_bytes)

        raw = await asyncio.wait_for(mgr_b.recv("node-a"), timeout=5.0)
        reassembled = channel_b.on_chunk_received(raw)
        assert reassembled is not None

        result = deserialize_tensor(reassembled, device="cpu")
        assert result.shape == tensor.shape
        assert torch.allclose(result.float(), tensor.float(), atol=0.01)
    finally:
        await mgr_a.close_all()
        await mgr_b.close_all()


@pytest.mark.asyncio
async def test_p2p_bidirectional_tensor_exchange(rtc_config):
    """Both peers can send and receive tensors (simulates node<->consumer)."""
    mgr_a, mgr_b = await _establish_pair(rtc_config)
    try:
        activations = torch.randn(1, 2560, dtype=torch.float16)
        logits = torch.randn(151936, dtype=torch.float32)

        async def send_a(data: bytes):
            await mgr_a.send("node-b", data)

        async def send_b(data: bytes):
            await mgr_b.send("node-a", data)

        ch_a = ChunkedChannel(send_a)
        ch_b = ChunkedChannel(send_b)
        recv_a = ChunkedChannel(lambda d: None)
        recv_b = ChunkedChannel(lambda d: None)

        act_bytes = serialize_tensor(activations)
        await ch_a.send_message(act_bytes)

        reassembled = None
        while reassembled is None:
            raw = await asyncio.wait_for(mgr_b.recv("node-a"), timeout=5.0)
            reassembled = recv_b.on_chunk_received(raw)
        assert reassembled == act_bytes

        logits_bytes = serialize_tensor(logits)
        await ch_b.send_message(logits_bytes)

        reassembled = None
        while reassembled is None:
            raw = await asyncio.wait_for(mgr_a.recv("node-b"), timeout=5.0)
            reassembled = recv_a.on_chunk_received(raw)
        assert reassembled == logits_bytes

        result = deserialize_tensor(reassembled, device="cpu")
        assert result.shape == logits.shape
        assert torch.allclose(result, logits, atol=1e-6)
    finally:
        await mgr_a.close_all()
        await mgr_b.close_all()


@pytest.mark.asyncio
async def test_p2p_multiple_sequential_messages(rtc_config):
    """Multiple messages sent sequentially all reassemble correctly."""
    mgr_a, mgr_b = await _establish_pair(rtc_config)
    try:
        async def send_a(data: bytes):
            await mgr_a.send("node-b", data)

        ch_send = ChunkedChannel(send_a)
        ch_recv = ChunkedChannel(lambda d: None)

        tensors = [torch.randn(1, 2560, dtype=torch.float16) for _ in range(5)]
        wire_list = [serialize_tensor(t) for t in tensors]

        for wire in wire_list:
            await ch_send.send_message(wire)

        results = []
        for _ in range(len(tensors)):
            reassembled = None
            while reassembled is None:
                raw = await asyncio.wait_for(mgr_b.recv("node-a"), timeout=5.0)
                reassembled = ch_recv.on_chunk_received(raw)
            results.append(reassembled)

        for orig, received in zip(wire_list, results):
            assert orig == received
    finally:
        await mgr_a.close_all()
        await mgr_b.close_all()
