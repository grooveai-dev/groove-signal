"""Chunked message transport for WebRTC data channels.

WebRTC data channels have a ~64KB max message size. LLM tensors can be
600KB+ (logits for vocab=151936). This module provides transparent
chunking and reassembly over any async send function.

Wire format per chunk (12 bytes header + payload):
  [4 bytes: message_id  (uint32, big-endian)]
  [4 bytes: chunk_index (uint32, big-endian)]
  [2 bytes: total_chunks (uint16, big-endian)]
  [2 bytes: flags        (uint16, big-endian)]  bit 0: is_final
"""

from __future__ import annotations

import logging
import struct

logger = logging.getLogger(__name__)

CHUNK_SIZE = 48 * 1024
MAX_CHUNKS_PER_MESSAGE = 65535
MAX_PENDING_MESSAGES = 64

HEADER_FMT = "!IIHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 12 bytes


class ChunkedChannel:
    """Wraps a WebRTC data channel with transparent chunking/reassembly."""

    def __init__(self, send_func: callable, chunk_size: int = CHUNK_SIZE):
        self._send_func = send_func
        self.chunk_size = chunk_size
        self._msg_counter: int = 0
        self._reassembly: dict[int, dict] = {}

    async def send_message(self, data: bytes) -> None:
        """Split data into chunks and send over the data channel."""
        msg_id = self._msg_counter
        self._msg_counter = (self._msg_counter + 1) % (2**32)

        if len(data) == 0:
            header = struct.pack(HEADER_FMT, msg_id, 0, 1, 1)
            await self._send_func(header)
            return

        chunks = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
        total = len(chunks)

        if total > MAX_CHUNKS_PER_MESSAGE:
            raise ValueError(
                f"Message requires {total} chunks, exceeds MAX_CHUNKS_PER_MESSAGE ({MAX_CHUNKS_PER_MESSAGE})"
            )

        for idx, chunk in enumerate(chunks):
            flags = 1 if idx == total - 1 else 0
            header = struct.pack(HEADER_FMT, msg_id, idx, total, flags)
            await self._send_func(header + chunk)

    def on_chunk_received(self, raw: bytes) -> bytes | None:
        """Process incoming chunk. Returns reassembled message when complete."""
        if len(raw) < HEADER_SIZE:
            logger.warning("chunk too short: %d bytes", len(raw))
            return None

        msg_id, chunk_idx, total, flags = struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])
        payload = raw[HEADER_SIZE:]

        if total == 0:
            logger.warning("invalid total_chunks=0 for message_id=%d", msg_id)
            return None

        if msg_id not in self._reassembly:
            if len(self._reassembly) >= MAX_PENDING_MESSAGES:
                oldest = min(self._reassembly.keys())
                del self._reassembly[oldest]
                logger.warning(
                    "reassembly buffer full, dropped oldest message_id=%d", oldest,
                )
            self._reassembly[msg_id] = {"total": total, "chunks": {}}

        entry = self._reassembly[msg_id]
        entry["chunks"][chunk_idx] = payload

        if len(entry["chunks"]) == total:
            ordered = [entry["chunks"][i] for i in range(total)]
            del self._reassembly[msg_id]
            return b"".join(ordered)

        return None
