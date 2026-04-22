"""Tests for src.common.chunking — ChunkedChannel chunking/reassembly."""

import asyncio
import struct

import pytest

from src.common.chunking import (
    CHUNK_SIZE,
    HEADER_FMT,
    HEADER_SIZE,
    MAX_PENDING_MESSAGES,
    ChunkedChannel,
)


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
