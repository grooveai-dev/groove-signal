"""Layer-assignment scheduler for the Groove inference pipeline.

Pure logic, pure data. No websocket / asyncio / I/O dependencies — this
module is intentionally portable so it can move into the M3 decentralized
consensus layer without modification.

Inputs are plain dicts describing node capabilities; outputs are plain
{node_id: (layer_start, layer_end_exclusive)} mappings.
"""

from __future__ import annotations


# Static model registry. Eventually this will be loaded from disk or fetched
# from the network; for now, hard-coded entries are sufficient for M2.
MODEL_REGISTRY: dict[str, dict] = {
    "Qwen/Qwen3-4B": {
        "total_layers": 36,
        "hidden_size": 2560,
        "num_heads": 32,
        "vocab_size": 151936,
        "memory_per_layer_mb": 110,
        "dtype": "bfloat16",
    },
    "Qwen/Qwen2.5-0.5B": {
        "total_layers": 24,
        "hidden_size": 896,
        "num_heads": 14,
        "vocab_size": 151936,
        "memory_per_layer_mb": 50,
        "dtype": "float16",
    },
}


def get_model_info(model_name: str) -> dict:
    """Return the registry entry for a model, or raise ValueError."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}")
    return MODEL_REGISTRY[model_name]


_MAX_PLAUSIBLE_VRAM_MB = 160_000
_MAX_PLAUSIBLE_RAM_MB = 4_000_000


def _effective_capacity_mb(node: dict) -> float:
    """Memory effectively usable for shard layers, in MB.

    GPU nodes are weighted by VRAM. CPU nodes are weighted by RAM with a
    0.5 penalty because RAM-only inference is meaningfully slower than
    VRAM and the scheduler should bias work toward GPUs.

    Self-reported values are clamped to plausible maximums to prevent
    a malicious node from gaming layer assignments.
    """
    device = (node.get("device") or "cpu").lower()
    vram = min(max(float(node.get("vram_mb") or 0), 0), _MAX_PLAUSIBLE_VRAM_MB)
    ram = min(max(float(node.get("ram_mb") or 0), 0), _MAX_PLAUSIBLE_RAM_MB)
    if device in ("cuda", "gpu", "rocm") and vram > 0:
        return vram
    if device == "mps" and vram > 0:
        return vram
    return ram * 0.5


def assign_layers(
    nodes_capabilities: list[dict],
    model_name: str,
) -> dict[str, tuple[int, int]]:
    """Assign contiguous layer ranges to nodes proportional to capacity.

    Returns {node_id: (layer_start, layer_end_exclusive)} covering
    [0, total_layers) with no gaps or overlaps. Each participating node
    receives at least one layer. Nodes are ordered by capacity descending
    so the largest node serves the earliest layers (typically slightly
    heavier due to embedding-adjacent compute).

    Raises ValueError for unknown models, empty input, or models with
    fewer layers than nodes (cannot give every node ≥1 layer).
    """
    info = get_model_info(model_name)
    total_layers: int = info["total_layers"]

    if not nodes_capabilities:
        raise ValueError("Cannot assign layers with no nodes")

    nodes = list(nodes_capabilities)
    n = len(nodes)
    if n > total_layers:
        raise ValueError(
            f"More nodes ({n}) than layers ({total_layers}); cannot give "
            "every node at least one layer"
        )

    if n == 1:
        nid = nodes[0]["node_id"]
        capacity = _effective_capacity_mb(nodes[0])
        model_mem = total_layers * info.get("memory_per_layer_mb", 0)
        if model_mem > 0 and 0 < capacity < model_mem:
            raise ValueError(
                f"Single node has {capacity:.0f}MB but model needs "
                f"~{model_mem:.0f}MB — waiting for more nodes"
            )
        return {nid: (0, total_layers)}

    capacities = [(node["node_id"], _effective_capacity_mb(node)) for node in nodes]
    total_capacity = sum(c for _, c in capacities)

    if total_capacity <= 0:
        # No capacity reported — fall back to even split.
        even = total_layers // n
        remainder = total_layers - even * n
        sizes = [even + (1 if i < remainder else 0) for i in range(n)]
        ordered_ids = [nid for nid, _ in capacities]
    else:
        # Proportional allocation, biggest node first. Each node gets at
        # least 1 layer; remainder distributed largest-first.
        capacities.sort(key=lambda x: x[1], reverse=True)
        ordered_ids = [nid for nid, _ in capacities]
        raw = [(c / total_capacity) * total_layers for _, c in capacities]
        sizes = [max(1, int(r)) for r in raw]

        # Reconcile to exactly total_layers.
        diff = total_layers - sum(sizes)
        if diff > 0:
            # Distribute extra layers to nodes with the largest fractional
            # remainders (largest first as tiebreak — they're already sorted).
            fracs = sorted(
                enumerate(raw), key=lambda x: x[1] - int(x[1]), reverse=True
            )
            i = 0
            while diff > 0:
                idx = fracs[i % len(fracs)][0]
                sizes[idx] += 1
                diff -= 1
                i += 1
        elif diff < 0:
            # Trim from the smallest assignments first, but never below 1.
            order = sorted(range(n), key=lambda i: sizes[i])
            i = 0
            while diff < 0:
                idx = order[i % len(order)]
                if sizes[idx] > 1:
                    sizes[idx] -= 1
                    diff += 1
                i += 1
                if i > 10 * n:  # pathological safety
                    raise ValueError(
                        "Cannot fit layers — too many nodes for model size"
                    )

    assignments: dict[str, tuple[int, int]] = {}
    cursor = 0
    for nid, size in zip(ordered_ids, sizes):
        assignments[nid] = (cursor, cursor + size)
        cursor += size
    return assignments


def calculate_rebalance(
    current_assignments: dict[str, tuple[int, int]],
    new_node_capabilities: list[dict],
    model_name: str,
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Compute a new assignment for the union of current + new nodes.

    Returns (new_assignments, affected_node_ids) where affected_node_ids
    is the subset of nodes whose (layer_start, layer_end) changed (or are
    newly added / no longer present).
    """
    # Build a capabilities list. Nodes already in current_assignments that
    # don't appear in new_node_capabilities are kept (with empty caps so
    # they receive default weighting); nodes in new_node_capabilities take
    # precedence.
    by_id: dict[str, dict] = {}
    for nid in current_assignments:
        by_id[nid] = {"node_id": nid}
    for caps in new_node_capabilities:
        nid = caps["node_id"]
        by_id[nid] = dict(caps)

    new_assignments = assign_layers(list(by_id.values()), model_name)

    affected: list[str] = []
    all_ids = set(new_assignments) | set(current_assignments)
    for nid in all_ids:
        old = current_assignments.get(nid)
        new = new_assignments.get(nid)
        if old != new:
            affected.append(nid)
    return new_assignments, affected


def validate_coverage(
    assignments: dict[str, tuple[int, int]],
    model_name: str,
) -> bool:
    """Check assignments tile [0, total_layers) with no gaps or overlaps."""
    info = get_model_info(model_name)
    total_layers: int = info["total_layers"]
    if not assignments:
        return False
    spans = sorted(assignments.values(), key=lambda s: s[0])
    if spans[0][0] != 0:
        return False
    if spans[-1][1] != total_layers:
        return False
    for i in range(len(spans) - 1):
        if spans[i][1] != spans[i + 1][0]:
            return False
    for start, end in spans:
        if end <= start:
            return False
    return True
