"""Tests for src.common.chunking — ChunkedChannel chunking/reassembly.

Also includes Phase 4 resilience tests: consumer TURN credentials,
P2P reconnection monitor, and node SDP re-offer handling.
"""

import asyncio
import struct
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.chunking import (
    CHUNK_SIZE,
    HEADER_FMT,
    HEADER_SIZE,
    MAX_PENDING_MESSAGES,
    ChunkedChannel,
)


def _ensure_mock_modules():
    """Inject mock modules for heavy deps not in test env."""
    stubs = {}
    for mod_name in (
        "torch", "torch.cuda", "torch.backends", "torch.backends.mps",
        "numpy", "msgpack", "websockets", "websockets.asyncio",
        "websockets.asyncio.client", "transformers",
        "aiortc", "src.common.peer", "src.common.tensor_transfer",
        "src.common.benchmark", "src.node.identity", "src.node.kv_cache",
        "src.node.shard_loader",
    ):
        if mod_name not in sys.modules:
            stubs[mod_name] = MagicMock()
    sys.modules.update(stubs)
    return stubs


def _make_channel():
    """Create a ChunkedChannel with a capturing send_func."""
    sent = []

    async def send_func(data: bytes):
        sent.append(data)

    ch = ChunkedChannel(send_func)
    return ch, sent


@pytest.mark.asyncio
async def test_single_chunk_small_message():
    """Data smaller than CHUNK_SIZE produces one chunk with header."""
    ch, sent = _make_channel()
    data = b"hello world"
    await ch.send_message(data)

    assert len(sent) == 1
    assert len(sent[0]) == HEADER_SIZE + len(data)

    msg_id, idx, total, flags = struct.unpack(HEADER_FMT, sent[0][:HEADER_SIZE])
    assert msg_id == 0
    assert idx == 0
    assert total == 1
    assert flags == 1  # is_final
    assert sent[0][HEADER_SIZE:] == data


@pytest.mark.asyncio
async def test_single_chunk_reassembly():
    """Single-chunk message reassembles immediately."""
    ch, sent = _make_channel()
    data = b"hello world"
    await ch.send_message(data)

    receiver = ChunkedChannel(lambda d: None)
    result = receiver.on_chunk_received(sent[0])
    assert result == data


@pytest.mark.asyncio
async def test_multi_chunk_logits_size():
    """600KB message (like logits) chunks correctly and reassembles."""
    ch, sent = _make_channel()
    data = bytes(range(256)) * 2400  # 614400 bytes (~600KB)
    await ch.send_message(data)

    expected_chunks = -(-len(data) // CHUNK_SIZE)  # ceil division
    assert len(sent) == expected_chunks
    assert expected_chunks == 13  # 600KB / 48KB ≈ 13

    receiver = ChunkedChannel(lambda d: None)
    result = None
    for chunk in sent:
        result = receiver.on_chunk_received(chunk)
    assert result == data


@pytest.mark.asyncio
async def test_large_message_prompt_prefill():
    """2.5MB message (like prompt prefill) produces 54+ chunks."""
    ch, sent = _make_channel()
    data = b"\xab" * (2560 * 1024)  # 2.5MB
    await ch.send_message(data)

    expected_chunks = -(-len(data) // CHUNK_SIZE)
    assert len(sent) == expected_chunks
    assert expected_chunks >= 54

    receiver = ChunkedChannel(lambda d: None)
    result = None
    for chunk in sent:
        result = receiver.on_chunk_received(chunk)
    assert result == data


@pytest.mark.asyncio
async def test_message_id_wraparound():
    """message_id wraps at 2^32."""
    ch, sent = _make_channel()
    ch._msg_counter = (2**32) - 1
    await ch.send_message(b"wrap")

    msg_id_1, _, _, _ = struct.unpack(HEADER_FMT, sent[0][:HEADER_SIZE])
    assert msg_id_1 == (2**32) - 1

    await ch.send_message(b"after")
    msg_id_2, _, _, _ = struct.unpack(HEADER_FMT, sent[1][:HEADER_SIZE])
    assert msg_id_2 == 0


@pytest.mark.asyncio
async def test_max_pending_messages_enforcement():
    """Oldest incomplete message is dropped when MAX_PENDING_MESSAGES exceeded."""
    receiver = ChunkedChannel(lambda d: None)

    for i in range(MAX_PENDING_MESSAGES):
        header = struct.pack(HEADER_FMT, i, 0, 2, 0)
        receiver.on_chunk_received(header + b"chunk0")
    assert len(receiver._reassembly) == MAX_PENDING_MESSAGES
    assert 0 in receiver._reassembly

    header = struct.pack(HEADER_FMT, 9999, 0, 2, 0)
    receiver.on_chunk_received(header + b"chunk0")
    assert len(receiver._reassembly) == MAX_PENDING_MESSAGES
    assert 0 not in receiver._reassembly
    assert 9999 in receiver._reassembly


@pytest.mark.asyncio
async def test_zero_byte_message():
    """Zero-byte message sends one chunk with empty payload."""
    ch, sent = _make_channel()
    await ch.send_message(b"")

    assert len(sent) == 1
    assert len(sent[0]) == HEADER_SIZE  # header only, no payload

    msg_id, idx, total, flags = struct.unpack(HEADER_FMT, sent[0][:HEADER_SIZE])
    assert total == 1
    assert flags == 1

    receiver = ChunkedChannel(lambda d: None)
    result = receiver.on_chunk_received(sent[0])
    assert result == b""


def test_header_format_spec():
    """Chunk header is exactly 12 bytes with correct struct format."""
    assert HEADER_SIZE == 12
    assert HEADER_FMT == "!IIHH"

    header = struct.pack(HEADER_FMT, 42, 7, 10, 1)
    msg_id, idx, total, flags = struct.unpack(HEADER_FMT, header)
    assert msg_id == 42
    assert idx == 7
    assert total == 10
    assert flags == 1


@pytest.mark.asyncio
async def test_interleaved_messages():
    """Two concurrent messages reassemble independently."""
    receiver = ChunkedChannel(lambda d: None)

    header_a0 = struct.pack(HEADER_FMT, 100, 0, 2, 0)
    header_b0 = struct.pack(HEADER_FMT, 200, 0, 2, 0)
    header_a1 = struct.pack(HEADER_FMT, 100, 1, 2, 1)
    header_b1 = struct.pack(HEADER_FMT, 200, 1, 2, 1)

    assert receiver.on_chunk_received(header_a0 + b"A0") is None
    assert receiver.on_chunk_received(header_b0 + b"B0") is None
    result_a = receiver.on_chunk_received(header_a1 + b"A1")
    assert result_a == b"A0A1"
    result_b = receiver.on_chunk_received(header_b1 + b"B1")
    assert result_b == b"B0B1"


@pytest.mark.asyncio
async def test_chunk_too_short():
    """Chunks shorter than header are rejected."""
    receiver = ChunkedChannel(lambda d: None)
    result = receiver.on_chunk_received(b"\x00" * 8)
    assert result is None


@pytest.mark.asyncio
async def test_multiple_sequential_messages():
    """Sequential full messages increment message_id."""
    ch, sent = _make_channel()
    for i in range(5):
        await ch.send_message(f"msg{i}".encode())

    assert len(sent) == 5
    for i, raw in enumerate(sent):
        msg_id, _, _, _ = struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])
        assert msg_id == i


# ---------------------------------------------------------------------------
# Phase 4 resilience tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _mock_heavy_deps():
    """Mock torch/aiortc/etc for tests that import consumer/node modules."""
    saved = {}
    stubs = {}
    for mod_name in (
        "torch", "torch.cuda", "torch.backends", "torch.backends.mps",
        "numpy", "np", "msgpack", "websockets", "websockets.asyncio",
        "websockets.asyncio.client", "transformers",
        "aiortc", "src.common.peer", "src.common.tensor_transfer",
        "src.common.benchmark", "src.node.identity", "src.node.kv_cache",
        "src.node.shard_loader",
    ):
        if mod_name in sys.modules:
            saved[mod_name] = sys.modules[mod_name]
        mock = MagicMock()
        if mod_name == "torch":
            mock.cuda.is_available.return_value = False
            mock.backends.mps.is_available.return_value = False
        if mod_name == "msgpack":
            mock.packb.return_value = b'\x80'
        stubs[mod_name] = mock
    sys.modules.update(stubs)

    # Force re-import of protocol (it's pure python, safe)
    for key in list(sys.modules.keys()):
        if key.startswith("src.consumer.") or key.startswith("src.node.server"):
            del sys.modules[key]

    yield

    for mod_name, mock in stubs.items():
        if mod_name in saved:
            sys.modules[mod_name] = saved[mod_name]
        else:
            sys.modules.pop(mod_name, None)
    for key in list(sys.modules.keys()):
        if key.startswith("src.consumer.") or key.startswith("src.node.server"):
            del sys.modules[key]


def _make_consumer_client(**kwargs):
    """Create an InferenceClient with mocked internals for unit testing."""
    from src.consumer.client import InferenceClient

    client = InferenceClient(relay_host="localhost", relay_port=8770, **kwargs)
    client.session_id = "test-session"
    client.stream_id = "test-stream"
    client.relay_ws = AsyncMock()
    return client


@pytest.mark.usefixtures("_mock_heavy_deps")
class TestConsumerTurnCredentials:
    """TASK 1: Consumer extracts turn_servers from PIPELINE_CONFIG."""

    def test_turn_servers_extracted_from_pipeline_config(self):
        from src.consumer.client import InferenceClient
        client = InferenceClient()
        assert client._turn_servers is None

    @pytest.mark.asyncio
    async def test_establish_p2p_uses_turn_servers(self):
        client = _make_consumer_client()
        client._turn_servers = [
            {"urls": "turn:relay.example.com:3478", "username": "u", "credential": "p"},
        ]
        client.pipeline = [{"node_id": "0xAABB"}]

        mock_pm = MagicMock()
        mock_pm.create_offer = AsyncMock(return_value="v=0\r\nfake sdp")
        mock_pm.is_connected = MagicMock(return_value=True)

        mock_cls = MagicMock(return_value=mock_pm)
        mock_build = MagicMock(return_value="mock_rtc_config")
        with patch.dict("sys.modules", {"src.common.peer": MagicMock(
            PeerConnectionManager=mock_cls,
            build_rtc_config=mock_build,
        )}):
            import src.consumer.client as mod
            orig_build = mod.build_rtc_config
            mod.build_rtc_config = mock_build
            try:
                await client._establish_p2p()
            finally:
                mod.build_rtc_config = orig_build

            mock_build.assert_called_once_with(client._turn_servers)
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs.get("rtc_config") == "mock_rtc_config"

    @pytest.mark.asyncio
    async def test_establish_p2p_works_without_turn_servers(self):
        client = _make_consumer_client()
        client._turn_servers = None
        client.pipeline = [{"node_id": "0xCCDD"}]

        mock_pm = MagicMock()
        mock_pm.create_offer = AsyncMock(return_value="v=0\r\nfake sdp")
        mock_pm.is_connected = MagicMock(return_value=True)

        mock_cls = MagicMock(return_value=mock_pm)
        with patch.dict("sys.modules", {"src.common.peer": MagicMock(
            PeerConnectionManager=mock_cls,
        )}):
            await client._establish_p2p()
            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs.get("rtc_config") is None


@pytest.mark.usefixtures("_mock_heavy_deps")
class TestConsumerReconnectionMonitor:
    """TASK 3: P2P reconnection with max 3 retries per peer."""

    @pytest.mark.asyncio
    async def test_monitor_detects_dropped_peer_and_reattempts(self):
        client = _make_consumer_client()
        node_id = "0xDEAD"
        client.p2p_connected.add(node_id)
        client.pipeline = [{"node_id": node_id}]

        mock_pm = MagicMock()
        is_connected_calls = [False]
        mock_pm.is_connected = MagicMock(side_effect=lambda nid: is_connected_calls.pop(0) if is_connected_calls else True)
        mock_pm.close = AsyncMock()
        mock_pm.create_offer = AsyncMock(return_value="v=0\r\nreconnect sdp")
        client.peer_manager = mock_pm

        task = asyncio.create_task(client._monitor_p2p())
        await asyncio.sleep(2.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        mock_pm.close.assert_called_with(node_id)
        mock_pm.create_offer.assert_called_with(node_id)
        assert client._reconnect_count >= 1

    @pytest.mark.asyncio
    async def test_max_reconnection_attempts(self):
        client = _make_consumer_client()
        node_id = "0xBEEF"
        client.pipeline = [{"node_id": node_id}]
        client._reconnect_attempts[node_id] = 3

        mock_pm = MagicMock()
        mock_pm.is_connected = MagicMock(return_value=False)
        mock_pm.close = AsyncMock()
        mock_pm.create_offer = AsyncMock(return_value="v=0\r\nsdp")
        client.peer_manager = mock_pm

        client.p2p_connected.add(node_id)
        task = asyncio.create_task(client._monitor_p2p())
        await asyncio.sleep(2.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        mock_pm.create_offer.assert_not_called()

    @pytest.mark.asyncio
    async def test_monitor_cancelled_on_close(self):
        client = _make_consumer_client()
        client.pipeline = []
        client._recv_task = None

        client._p2p_monitor_task = asyncio.create_task(client._monitor_p2p())
        await asyncio.sleep(0.1)
        assert not client._p2p_monitor_task.done()

        await client.close_session()
        assert client._p2p_monitor_task is None


@pytest.mark.usefixtures("_mock_heavy_deps")
class TestNodeSdpReoffer:
    """TASK 4: Node accepts re-offers for existing peers."""

    @pytest.mark.asyncio
    async def test_reoffer_replaces_old_connection(self):
        from src.node.server import ComputeNodeServer

        server = ComputeNodeServer(
            model_name="test-model",
            layer_start=0, layer_end=4,
            device="cpu", legacy_mode=True,
        )

        mock_pm = MagicMock()
        mock_pm.is_connected = MagicMock(return_value=True)
        mock_pm.close = AsyncMock()
        mock_pm.handle_offer = AsyncMock(return_value="v=0\r\nanswer sdp")
        server.peer_manager = mock_pm

        old_channel = MagicMock()
        server.p2p_channels["0xPEER"] = old_channel

        mock_ws = AsyncMock()

        msg = {
            "type": "sdp_offer",
            "session_id": "sess-1",
            "from_node_id": "0xPEER",
            "sdp": "v=0\r\noffer sdp",
        }

        await server._handle_sdp_offer(mock_ws, msg)

        mock_pm.close.assert_called_with("0xPEER")
        mock_pm.handle_offer.assert_called_with("0xPEER", "v=0\r\noffer sdp")
        assert "0xPEER" not in server.p2p_channels

    @pytest.mark.asyncio
    async def test_node_peer_state_monitor_cleans_up(self):
        from src.node.server import ComputeNodeServer

        server = ComputeNodeServer(
            model_name="test-model",
            layer_start=0, layer_end=4,
            device="cpu", legacy_mode=True,
        )

        mock_pm = MagicMock()
        mock_pm.is_connected = MagicMock(return_value=False)
        server.peer_manager = mock_pm

        server.p2p_channels["0xDROP"] = MagicMock()

        task = asyncio.create_task(server._monitor_peer_state("0xDROP"))
        await asyncio.sleep(2.5)
        await task

        assert "0xDROP" not in server.p2p_channels


@pytest.mark.usefixtures("_mock_heavy_deps")
class TestClientMetrics:
    """TASK 5: P2P vs relay send counters and session stats."""

    def test_initial_counters_zero(self):
        client = _make_consumer_client()
        assert client._p2p_send_count == 0
        assert client._relay_send_count == 0
        assert client._reconnect_count == 0

    @pytest.mark.asyncio
    async def test_relay_send_increments_counter(self):
        client = _make_consumer_client()
        client.pipeline = [{"node_id": "0xNODE"}]
        client._recv_task = MagicMock()
        client._recv_task.done = MagicMock(return_value=False)

        async def mock_send(data):
            for f in client._waiters.values():
                if not f.done():
                    f.set_result({"type": "activations", "seq_pos": 0})

        client.relay_ws.send = mock_send

        inner = {"seq_pos": 0, "type": "activations"}
        try:
            await asyncio.wait_for(
                client._send_to_node("0xNODE", inner, timeout=1.0), timeout=2.0,
            )
        except Exception:
            pass

        assert client._relay_send_count >= 1
        assert client._p2p_send_count == 0

    @pytest.mark.asyncio
    async def test_p2p_send_increments_counter(self):
        client = _make_consumer_client()
        node_id = "0xP2P"
        client.p2p_connected.add(node_id)

        mock_channel = MagicMock()
        mock_channel.send_message = AsyncMock()
        client.p2p_channels[node_id] = mock_channel

        client._recv_task = MagicMock()
        client._recv_task.done = MagicMock(return_value=False)

        async def send_and_resolve():
            inner = {"seq_pos": 7, "type": "activations"}
            return await client._send_to_node(node_id, inner, timeout=1.0)

        task = asyncio.create_task(send_and_resolve())
        await asyncio.sleep(0.05)

        if 7 in client._waiters:
            client._waiters[7].set_result({"type": "activations", "seq_pos": 7})

        try:
            await asyncio.wait_for(task, timeout=2.0)
        except Exception:
            pass

        assert client._p2p_send_count == 1
