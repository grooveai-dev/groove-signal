"""Tests for LZ4 tensor compression and backward compatibility."""

import struct

import pytest
import torch

from src.common.tensor_transfer import (
    serialize_tensor,
    deserialize_tensor,
    _COMPRESS_RAW,
    _COMPRESS_LZ4,
    _COMPRESSION_THRESHOLD,
)


class TestCompressionRoundTrip:
    def test_large_tensor_compressed(self):
        t = torch.randn(128, 256, dtype=torch.float16)
        data = serialize_tensor(t, compress=True)
        assert data[0] == _COMPRESS_LZ4
        result = deserialize_tensor(data)
        assert torch.allclose(result.float(), t.float(), atol=1e-3)

    def test_small_tensor_not_compressed(self):
        t = torch.randn(4, 4, dtype=torch.float32)
        data = serialize_tensor(t, compress=True)
        assert data[0] == _COMPRESS_RAW
        result = deserialize_tensor(data)
        assert torch.allclose(result, t)

    def test_float32_roundtrip(self):
        t = torch.randn(64, 128, dtype=torch.float32)
        data = serialize_tensor(t)
        result = deserialize_tensor(data)
        assert torch.allclose(result, t)

    def test_float16_roundtrip(self):
        t = torch.randn(64, 128, dtype=torch.float16)
        data = serialize_tensor(t)
        result = deserialize_tensor(data)
        assert torch.allclose(result.float(), t.float(), atol=1e-3)

    def test_int32_roundtrip(self):
        t = torch.randint(0, 1000, (64, 64), dtype=torch.int32)
        data = serialize_tensor(t)
        result = deserialize_tensor(data)
        assert torch.equal(result, t)

    def test_1d_tensor(self):
        t = torch.randn(100, dtype=torch.float32)
        data = serialize_tensor(t)
        result = deserialize_tensor(data)
        assert torch.allclose(result, t)

    def test_3d_tensor(self):
        t = torch.randn(2, 16, 32, dtype=torch.float16)
        data = serialize_tensor(t)
        result = deserialize_tensor(data)
        assert result.shape == t.shape


class TestCompressionThreshold:
    def test_below_threshold_skips_compression(self):
        t = torch.randn(16, dtype=torch.float16)  # 32 bytes, well below 16KB
        data = serialize_tensor(t, compress=True)
        assert data[0] == _COMPRESS_RAW

    def test_above_threshold_uses_lz4(self):
        t = torch.zeros(32 * 1024, dtype=torch.float16)
        data = serialize_tensor(t, compress=True)
        assert data[0] == _COMPRESS_LZ4

    def test_compress_false_always_raw(self):
        t = torch.randn(128, 256, dtype=torch.float16)
        data = serialize_tensor(t, compress=False)
        assert data[0] == _COMPRESS_RAW
        result = deserialize_tensor(data)
        assert torch.allclose(result.float(), t.float(), atol=1e-3)


class TestCompressionSizeReduction:
    def test_zeros_compress_well(self):
        t = torch.zeros(128, 256, dtype=torch.float16)
        raw = serialize_tensor(t, compress=False)
        compressed = serialize_tensor(t, compress=True)
        assert len(compressed) < len(raw) * 0.5

    def test_random_still_roundtrips(self):
        t = torch.randn(128, 256, dtype=torch.float16)
        data = serialize_tensor(t, compress=True)
        result = deserialize_tensor(data)
        assert torch.allclose(result.float(), t.float(), atol=1e-3)


class TestBackwardCompatibility:
    def test_raw_flag_reads_correctly(self):
        t = torch.randn(8, 8, dtype=torch.float32)
        data = serialize_tensor(t, compress=False)
        assert data[0] == _COMPRESS_RAW
        result = deserialize_tensor(data)
        assert torch.allclose(result, t)

    def test_v1_format_without_flag(self):
        """Old format (no compression flag) should still be readable.

        v1 data starts with ndim (uint8) which for typical tensors is 1-4,
        not 0x00 or 0x01, so the deserializer falls back to raw parse.
        """
        t = torch.randn(4, 8, 16, dtype=torch.float32)
        # ndim=3 means first byte is 0x03, which is neither 0x00 nor 0x01
        from src.common.tensor_transfer import _torch_to_numpy_safe, _TORCH_TO_NP
        import io
        import numpy as np

        shape = t.shape
        dtype_str = _TORCH_TO_NP[t.dtype]
        arr = _torch_to_numpy_safe(t)
        dtype_bytes = dtype_str.encode("ascii")
        ndim = len(shape)
        header_fmt = f"!B{ndim}iB"
        header = struct.pack(header_fmt, ndim, *shape, len(dtype_bytes))
        buf = io.BytesIO()
        buf.write(header)
        buf.write(dtype_bytes)
        buf.write(arr.tobytes())
        v1_data = buf.getvalue()

        result = deserialize_tensor(v1_data)
        assert torch.allclose(result, t)
