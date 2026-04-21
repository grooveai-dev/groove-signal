"""Fast tensor serialization for inter-node activation transfer.

Uses raw numpy byte buffers — no pickle (security), no torch.save (slow).
Target: <1ms round-trip for 8KB tensors on local network.
"""

import io
import struct

import numpy as np
import torch

# Header: shape_ndim (uint8) | shape (ndim x int32) | dtype_len (uint8) | dtype_str | data
_NDIM_FMT = "!B"
_DIM_FMT = "!i"
_DTYPE_LEN_FMT = "!B"

# Mapping between torch and numpy dtypes
_TORCH_TO_NP = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_NP_TO_TORCH = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _torch_to_numpy_safe(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to numpy, handling bfloat16 specially."""
    t = tensor.detach()
    if not t.is_cpu:
        t = t.cpu()
    if t.dtype == torch.bfloat16:
        return t.to(torch.float16).numpy()
    return t.numpy()


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a PyTorch tensor to a compact binary blob.

    Format: ndim | shape dims | dtype string | raw data bytes
    """
    shape = tensor.shape
    dtype_str = _TORCH_TO_NP.get(tensor.dtype)
    if dtype_str is None:
        raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")

    arr = _torch_to_numpy_safe(tensor)
    if tensor.dtype == torch.bfloat16:
        dtype_str = "bfloat16"

    dtype_bytes = dtype_str.encode("ascii")
    ndim = len(shape)

    header_fmt = f"!B{ndim}iB"
    header = struct.pack(header_fmt, ndim, *shape, len(dtype_bytes))

    buf = io.BytesIO()
    buf.write(header)
    buf.write(dtype_bytes)
    buf.write(arr.tobytes())

    return buf.getvalue()


_MAX_TENSOR_NDIM = 8
_MAX_TENSOR_ELEMENTS = 100_000_000


def deserialize_tensor(data: bytes, device: str = "cpu") -> torch.Tensor:
    """Reconstruct a PyTorch tensor from a binary blob.

    Validates shape, dtype, and data length to prevent OOM, crashes,
    or unexpected behavior from malicious payloads.
    """
    offset = 0

    (ndim,) = struct.unpack_from(_NDIM_FMT, data, offset)
    offset += struct.calcsize(_NDIM_FMT)

    if ndim > _MAX_TENSOR_NDIM:
        raise ValueError(f"tensor ndim {ndim} exceeds maximum {_MAX_TENSOR_NDIM}")

    if ndim > 0:
        shape_fmt = f"!{ndim}i"
        shape = list(struct.unpack_from(shape_fmt, data, offset))
        offset += struct.calcsize(shape_fmt)
    else:
        shape = []

    for dim in shape:
        if dim <= 0:
            raise ValueError(f"invalid tensor dimension: {dim}")

    total_elements = 1
    for dim in shape:
        total_elements *= dim
        if total_elements > _MAX_TENSOR_ELEMENTS:
            raise ValueError(
                f"tensor too large: exceeds {_MAX_TENSOR_ELEMENTS} element limit"
            )

    (dtype_len,) = struct.unpack_from(_DTYPE_LEN_FMT, data, offset)
    offset += struct.calcsize(_DTYPE_LEN_FMT)

    dtype_str = data[offset : offset + dtype_len].decode("ascii")
    offset += dtype_len

    if dtype_str != "bfloat16" and dtype_str not in _NP_TO_TORCH:
        raise ValueError(f"unsupported tensor dtype: {dtype_str!r}")

    if dtype_str == "bfloat16":
        expected_bytes = total_elements * 2
    else:
        expected_bytes = total_elements * np.dtype(dtype_str).itemsize

    if len(data) - offset < expected_bytes:
        raise ValueError(
            f"tensor data too short: got {len(data) - offset} bytes, expected {expected_bytes}"
        )
    raw_data = data[offset : offset + expected_bytes]

    if dtype_str == "bfloat16":
        arr = np.frombuffer(raw_data, dtype=np.float16).reshape(shape)
        tensor = torch.from_numpy(arr.copy()).to(torch.bfloat16)
    else:
        np_dtype = np.dtype(dtype_str)
        arr = np.frombuffer(raw_data, dtype=np_dtype).reshape(shape)
        torch_dtype = _NP_TO_TORCH[dtype_str]
        tensor = torch.from_numpy(arr.copy()).to(torch_dtype)

    if device != "cpu":
        tensor = tensor.to(device)

    return tensor
