"""Consumer-to-node matching logic.

Given a consumer's request (model_name, approximate location, optional
requirements like min_vram), find the best set of nodes to serve it. Two
entry points:

  * find_best_nodes — return ranked top-N for consumer display
  * assemble_pipeline — return an ordered, layer-complete pipeline so a
    single model is covered end-to-end by high-scoring nodes

Separated from the server module so the matching logic is trivially
testable and can be reused by other signal-layer services (dashboard,
gossip, etc.).
"""

from __future__ import annotations

from src.signal.registry import NodeRegistry
from src.signal.scoring import rank_nodes
from src.relay.scheduler import MODEL_REGISTRY


def _meets_requirements(node: dict, requirements: dict | None) -> bool:
    """Filter out nodes that fail a consumer's hard requirements.

    Requirements are minimums: min_vram, min_ram, max_load, min_bandwidth_mbps,
    device (exact match). Any requirement the consumer did not specify is
    ignored. Unknown requirement keys are ignored (forward-compat).
    """
    if not requirements:
        return True
    caps = node.get("capabilities") or {}

    if "min_vram" in requirements:
        if float(caps.get("vram_mb") or 0) < float(requirements["min_vram"]):
            return False
    if "min_ram" in requirements:
        if float(caps.get("ram_mb") or 0) < float(requirements["min_ram"]):
            return False
    if "min_bandwidth_mbps" in requirements:
        bw = float(caps.get("bandwidth_mbps") or 0)
        if bw < float(requirements["min_bandwidth_mbps"]):
            return False
    if "max_load" in requirements:
        if float(node.get("load") or 0) > float(requirements["max_load"]):
            return False
    if "device" in requirements:
        want = str(requirements["device"]).lower()
        if str(caps.get("device") or "").lower() != want:
            return False
    return True


class ConsumerMatcher:
    def __init__(
        self,
        registry: NodeRegistry,
        scorer_weights: dict | None = None,
    ) -> None:
        self.registry = registry
        self.scorer_weights = scorer_weights

    def find_best_nodes(
        self,
        model_name: str,
        consumer_location: dict | None,
        requirements: dict | None = None,
        top_n: int = 10,
    ) -> list[dict]:
        """Return top-N nodes ranked by composite score."""
        candidates = self.registry.get_active_nodes(model_name=model_name)
        if not candidates:
            return []
        filtered = [n for n in candidates if _meets_requirements(n, requirements)]
        if not filtered:
            return []
        consumer = {"location": consumer_location, "requirements": requirements or {}}
        return rank_nodes(
            filtered, consumer, top_n=top_n, weights=self.scorer_weights,
        )

    def assemble_pipeline(
        self,
        model_name: str,
        consumer_location: dict | None,
        requirements: dict | None = None,
    ) -> list[dict]:
        """Return an ordered pipeline that covers every layer of `model_name`.

        Greedy: at each uncovered layer, pick the highest-scoring active
        node whose assigned layer range starts at (or straddles) that layer.
        If no node covers a given layer, returns [] — the signal server
        should respond with NO_NODES rather than an incomplete pipeline.

        This is intentionally simpler than the scheduler's proportional
        capacity allocation: the scheduler decides who gets which layers
        when nodes join; the matcher just walks the existing assignment
        graph to pick the best-scored route for a given consumer.
        """
        if model_name not in MODEL_REGISTRY:
            return []
        total_layers = MODEL_REGISTRY[model_name]["total_layers"]

        # Only nodes that are actually active AND have layer assignments.
        candidates = [
            n for n in self.registry.get_active_nodes(model_name=model_name)
            if n.get("assignment_status") == "active"
            and n.get("layer_start") is not None
            and n.get("layer_end") is not None
            and _meets_requirements(n, requirements)
        ]
        if not candidates:
            return []

        consumer = {"location": consumer_location, "requirements": requirements or {}}
        scored = rank_nodes(
            candidates, consumer, top_n=0, weights=self.scorer_weights,
        )

        pipeline: list[dict] = []
        cursor = 0
        max_hops = total_layers + len(scored) + 1
        hops = 0
        while cursor < total_layers and hops < max_hops:
            hops += 1
            pick = None
            for node in scored:
                ls = int(node["layer_start"])
                le = int(node["layer_end"])
                # Scheduler produces half-open ranges [start, end_exclusive).
                if ls <= cursor < le:
                    pick = node
                    break
            if pick is None:
                return []
            pipeline.append(pick)
            cursor = int(pick["layer_end"])
        if cursor < total_layers:
            return []
        return pipeline
