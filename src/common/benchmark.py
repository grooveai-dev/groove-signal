"""Node startup micro-benchmark for the Groove inference pipeline.

Runs a quick benchmark before REGISTER_NODE to measure:
  - bench_ms_per_layer: average ms per transformer layer forward pass
  - bench_mem_bandwidth_gbps: memory bandwidth estimate (GB/s)

Total benchmark time target: <5 seconds. Results are included in the
node's capabilities dict so the scheduler has real inference speed data.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_FORWARD_PASS_ITERATIONS = 10
_BANDWIDTH_TENSOR_MB = 100


def benchmark_forward_pass(
    hidden_size: int,
    device: str,
    num_iterations: int = _FORWARD_PASS_ITERATIONS,
    dtype: object = None,
) -> float:
    """Benchmark a single transformer-layer-sized forward pass.

    Runs num_iterations of a linear(hidden_size -> hidden_size) + GELU
    to approximate the compute cost of one decoder layer.

    Returns average milliseconds per forward pass.
    """
    try:
        import torch
        if dtype is None:
            dtype = torch.float16
        layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size, bias=False),
            torch.nn.GELU(),
        ).to(device=device, dtype=dtype)

        x = torch.randn(1, 1, hidden_size, device=device, dtype=dtype)

        with torch.inference_mode():
            for _ in range(2):
                _ = layer(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = layer(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start

        ms_per_pass = (elapsed / num_iterations) * 1000.0

        del layer, x
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        return ms_per_pass

    except Exception as e:
        logger.warning("forward pass benchmark failed: %s", e)
        return 0.0


def benchmark_memory_bandwidth(
    device: str,
    size_mb: int = _BANDWIDTH_TENSOR_MB,
) -> float:
    """Estimate memory bandwidth by timing a large tensor copy.

    Returns estimated bandwidth in GB/s.
    """
    try:
        import torch
        num_floats = (size_mb * 1024 * 1024) // 4
        src = torch.randn(num_floats, device=device, dtype=torch.float32)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        dst = src.clone()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        bytes_copied = num_floats * 4 * 2  # read + write
        gbps = (bytes_copied / elapsed) / (1024 ** 3)

        del src, dst
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        return round(gbps, 2)

    except Exception as e:
        logger.warning("memory bandwidth benchmark failed: %s", e)
        return 0.0


def run_node_benchmark(
    hidden_size: int = 2560,
    device: str = "cpu",
) -> dict:
    """Run the full node benchmark suite.

    Args:
        hidden_size: Model hidden dimension (2560 for Qwen3-4B).
        device: Target device ('cpu', 'cuda', 'mps').

    Returns:
        Dict with bench_ms_per_layer and bench_mem_bandwidth_gbps.
    """
    logger.info("running startup benchmark on device=%s hidden_size=%d", device, hidden_size)
    overall_start = time.perf_counter()

    ms_per_layer = benchmark_forward_pass(hidden_size, device)
    bandwidth_gbps = benchmark_memory_bandwidth(device)

    elapsed = (time.perf_counter() - overall_start) * 1000.0
    logger.info(
        "benchmark complete: %.2f ms/layer, %.2f GB/s bandwidth (%.0fms total)",
        ms_per_layer, bandwidth_gbps, elapsed,
    )

    return {
        "bench_ms_per_layer": round(ms_per_layer, 3),
        "bench_mem_bandwidth_gbps": bandwidth_gbps,
    }


def classify_node_role(
    bench_ms_per_layer: float,
    bench_mem_bandwidth_gbps: float,
    device: str = "cpu",
) -> str:
    """Classify a node as 'reader' (prefill-optimized) or 'writer' (decode-optimized).

    - reader: high compute FLOPS, GPU preferred — good at prefill
    - writer: high memory bandwidth, unified memory competitive — good at decode

    Decode is memory-bandwidth-bound; prefill is compute-bound.
    """
    if bench_ms_per_layer <= 0 and bench_mem_bandwidth_gbps <= 0:
        return "writer"

    compute_score = (100.0 / bench_ms_per_layer) if bench_ms_per_layer > 0 else 0.0
    bandwidth_score = bench_mem_bandwidth_gbps

    if device in ("cuda", "gpu", "rocm"):
        compute_score *= 1.5

    if bandwidth_score <= 0 and compute_score <= 0:
        return "writer"

    total = compute_score + bandwidth_score
    if total <= 0:
        return "writer"

    compute_ratio = compute_score / total
    return "reader" if compute_ratio > 0.5 else "writer"
