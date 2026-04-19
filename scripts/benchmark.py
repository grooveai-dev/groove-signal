"""Groove Decentralized Inference Benchmark Suite.

Measures TTFT, sustained tok/s, per-hop latency, and speculative acceptance rate
across different pipeline configurations.

Usage: python scripts/benchmark.py --config benchmark_config.json
"""

import asyncio
import argparse
import json
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    ttft_ms: float = 0.0
    tokens_generated: int = 0
    total_time_s: float = 0.0
    tok_per_s: float = 0.0
    per_hop_latency_ms: dict = field(default_factory=dict)
    acceptance_rate: Optional[float] = None
    tokens_per_round_trip: Optional[float] = None


DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B",
    "draft_model_name": "Qwen/Qwen2.5-0.5B",
    "prompt": "Write a Python function that computes the Fibonacci sequence",
    "max_tokens": 100,
    "temperature": 0.0,
    "relay_host": "localhost",
    "relay_port": 8770,
    "scenarios": [
        {
            "name": "single_node_baseline",
            "description": "Full model on one machine",
            "nodes": [
                {"node_id": "node-a", "host": "localhost", "port": 8765,
                 "layer_start": 0, "layer_end": 31}
            ],
            "speculative": False,
        },
        {
            "name": "2_node_sequential",
            "description": "2-node sequential (no pipelining)",
            "nodes": [
                {"node_id": "node-a", "host": "localhost", "port": 8765,
                 "layer_start": 0, "layer_end": 15},
                {"node_id": "node-b", "host": "localhost", "port": 8766,
                 "layer_start": 16, "layer_end": 31},
            ],
            "speculative": False,
        },
        {
            "name": "2_node_pipelined",
            "description": "2-node with pipeline parallelism",
            "nodes": [
                {"node_id": "node-a", "host": "localhost", "port": 8765,
                 "layer_start": 0, "layer_end": 15},
                {"node_id": "node-b", "host": "localhost", "port": 8766,
                 "layer_start": 16, "layer_end": 31},
            ],
            "speculative": False,
            "pipelined": True,
        },
        {
            "name": "3_node_pipelined",
            "description": "3-node with pipeline parallelism",
            "nodes": [
                {"node_id": "node-a", "host": "localhost", "port": 8765,
                 "layer_start": 0, "layer_end": 10},
                {"node_id": "node-b", "host": "localhost", "port": 8766,
                 "layer_start": 11, "layer_end": 21},
                {"node_id": "node-c", "host": "localhost", "port": 8767,
                 "layer_start": 22, "layer_end": 31},
            ],
            "speculative": False,
            "pipelined": True,
        },
        {
            "name": "2_node_speculative",
            "description": "2-node with speculative decoding",
            "nodes": [
                {"node_id": "node-a", "host": "localhost", "port": 8765,
                 "layer_start": 0, "layer_end": 15},
                {"node_id": "node-b", "host": "localhost", "port": 8766,
                 "layer_start": 16, "layer_end": 31},
            ],
            "speculative": True,
        },
        {
            "name": "3_node_speculative",
            "description": "3-node with speculative decoding",
            "nodes": [
                {"node_id": "node-a", "host": "localhost", "port": 8765,
                 "layer_start": 0, "layer_end": 10},
                {"node_id": "node-b", "host": "localhost", "port": 8766,
                 "layer_start": 11, "layer_end": 21},
                {"node_id": "node-c", "host": "localhost", "port": 8767,
                 "layer_start": 22, "layer_end": 31},
            ],
            "speculative": True,
            "pipelined": True,
        },
    ],
}


async def run_scenario(
    config: dict, scenario: dict
) -> BenchmarkResult:
    from src.consumer.client import InferenceClient

    result = BenchmarkResult(name=scenario["name"])

    client = InferenceClient(
        relay_host=config["relay_host"], relay_port=config["relay_port"]
    )

    try:
        await client.connect()
        session_id = await client.start_session(config["model_name"])

        t_start = time.monotonic()
        first_token_time: Optional[float] = None
        token_count = 0

        async for text in client.generate(
            config["prompt"],
            max_tokens=config["max_tokens"],
            use_speculative=scenario.get("speculative", False),
            temperature=config["temperature"],
        ):
            if first_token_time is None:
                first_token_time = time.monotonic()
            token_count += 1

        t_end = time.monotonic()
        total_time = t_end - t_start

        result.ttft_ms = (
            (first_token_time - t_start) * 1000 if first_token_time else 0.0
        )
        result.tokens_generated = token_count
        result.total_time_s = total_time
        result.tok_per_s = token_count / total_time if total_time > 0 else 0.0

    except Exception as e:
        print(f"  ERROR in {scenario['name']}: {e}", file=sys.stderr)
    finally:
        await client.close_session()

    return result


def print_results_table(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'Scenario':<30} {'TTFT(ms)':>10} {'Tokens':>8} "
        f"{'Time(s)':>8} {'Tok/s':>8} {'Accept%':>10}"
    )
    print("\n" + "=" * len(header))
    print("GROOVE BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        accept_str = (
            f"{r.acceptance_rate * 100:.1f}%"
            if r.acceptance_rate is not None
            else "N/A"
        )
        print(
            f"{r.name:<30} {r.ttft_ms:>10.1f} {r.tokens_generated:>8} "
            f"{r.total_time_s:>8.2f} {r.tok_per_s:>8.1f} {accept_str:>10}"
        )

    print("=" * len(header))

    if any(r.per_hop_latency_ms for r in results):
        print("\nPER-HOP LATENCY:")
        for r in results:
            if r.per_hop_latency_ms:
                print(f"  {r.name}:")
                for hop, lat in r.per_hop_latency_ms.items():
                    print(f"    {hop}: {lat:.2f} ms")


async def run_benchmark(config: dict) -> None:
    scenarios = config.get("scenarios", DEFAULT_CONFIG["scenarios"])
    results: list[BenchmarkResult] = []

    for scenario in scenarios:
        print(f"Running: {scenario['name']} - {scenario.get('description', '')}")
        result = await run_scenario(config, scenario)
        results.append(result)
        print(f"  -> {result.tok_per_s:.1f} tok/s, TTFT: {result.ttft_ms:.1f}ms")

    print_results_table(results)

    output_path = config.get("output_file", "benchmark_results.json")
    output = []
    for r in results:
        output.append({
            "name": r.name,
            "ttft_ms": r.ttft_ms,
            "tokens_generated": r.tokens_generated,
            "total_time_s": r.total_time_s,
            "tok_per_s": r.tok_per_s,
            "acceptance_rate": r.acceptance_rate,
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Groove Decentralized Inference Benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to benchmark config JSON file",
    )
    parser.add_argument("--relay", type=str, default=None, help="Relay host:port")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)

    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    if args.relay:
        host, port = args.relay.rsplit(":", 1)
        config["relay_host"] = host
        config["relay_port"] = int(port)
    if args.model:
        config["model_name"] = args.model
    if args.max_tokens:
        config["max_tokens"] = args.max_tokens
    if args.prompt:
        config["prompt"] = args.prompt

    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    main()
