"""Tests for ConsumerMatcher."""

from __future__ import annotations

import pytest

from src.signal.matcher import ConsumerMatcher
from src.signal.registry import NodeRegistry


MODEL = "Qwen/Qwen2.5-0.5B"


class _FakeWS:
    async def close(self):
        pass


def _make_registered_node(
    reg: NodeRegistry,
    node_id: str,
    *,
    device: str = "cuda",
    vram: int = 16000,
    ram: int = 32000,
    location: dict | None = None,
    layers: tuple[int, int] | None = None,
    active_sessions: int = 0,
    downtime: int = 0,
    models_supported: list[str] | None = None,
    assigned_model: str = MODEL,
    status: str = "active",
):
    rec = reg.register(
        node_id=node_id,
        ws=_FakeWS(),
        capabilities={
            "device": device, "vram_mb": vram, "ram_mb": ram, "bandwidth_mbps": 100.0,
        },
        location=location,
        models_supported=models_supported or [MODEL],
    )
    rec.assignment_status = status
    rec.assigned_model = assigned_model
    rec.active_sessions = active_sessions
    rec.downtime_events = downtime
    if layers:
        rec.layer_start, rec.layer_end = layers
    return rec


# ---------------------------------------------------------------------------
def test_find_best_nodes_empty_registry():
    reg = NodeRegistry()
    matcher = ConsumerMatcher(reg)
    assert matcher.find_best_nodes(MODEL, None) == []


def test_find_best_nodes_returns_scored_list():
    reg = NodeRegistry()
    _make_registered_node(
        reg, "0x1", location={"lat": 40.7, "lon": -74.0}, layers=(0, 24),
    )
    _make_registered_node(
        reg, "0x2", location={"lat": -33.8, "lon": 151.2}, layers=(0, 24),
    )
    matcher = ConsumerMatcher(reg)
    ranked = matcher.find_best_nodes(MODEL, {"lat": 40.7, "lon": -74.0})
    assert len(ranked) == 2
    assert ranked[0]["node_id"] == "0x1"
    assert "score" in ranked[0]


def test_find_best_nodes_requirements_filter():
    reg = NodeRegistry()
    _make_registered_node(reg, "big", vram=24000, layers=(0, 24))
    _make_registered_node(reg, "small", device="cpu", vram=0, ram=4000, layers=(0, 24))
    matcher = ConsumerMatcher(reg)
    ranked = matcher.find_best_nodes(
        MODEL, None, requirements={"min_vram": 16000},
    )
    ids = [n["node_id"] for n in ranked]
    assert ids == ["big"]


def test_find_best_nodes_device_filter():
    reg = NodeRegistry()
    _make_registered_node(reg, "gpu", device="cuda", layers=(0, 24))
    _make_registered_node(reg, "cpu", device="cpu", layers=(0, 24))
    matcher = ConsumerMatcher(reg)
    ranked = matcher.find_best_nodes(
        MODEL, None, requirements={"device": "cuda"},
    )
    assert [n["node_id"] for n in ranked] == ["gpu"]


def test_find_best_nodes_top_n_trims():
    reg = NodeRegistry()
    for i in range(5):
        _make_registered_node(reg, f"n{i}", layers=(0, 24))
    matcher = ConsumerMatcher(reg)
    ranked = matcher.find_best_nodes(MODEL, None, top_n=3)
    assert len(ranked) == 3


# ---------------------------------------------------------------------------
def test_assemble_pipeline_single_node_full_coverage():
    reg = NodeRegistry()
    _make_registered_node(reg, "0x1", layers=(0, 24))
    matcher = ConsumerMatcher(reg)
    pipe = matcher.assemble_pipeline(MODEL, None)
    assert len(pipe) == 1
    assert pipe[0]["node_id"] == "0x1"


def test_assemble_pipeline_multi_node_covers_all_layers():
    reg = NodeRegistry()
    _make_registered_node(reg, "a", layers=(0, 12))
    _make_registered_node(reg, "b", layers=(12, 24))
    matcher = ConsumerMatcher(reg)
    pipe = matcher.assemble_pipeline(MODEL, None)
    ids = [n["node_id"] for n in pipe]
    assert ids == ["a", "b"]


def test_assemble_pipeline_missing_layer_returns_empty():
    reg = NodeRegistry()
    _make_registered_node(reg, "a", layers=(0, 6))
    # No node covers layers 6-23 -> pipeline cannot be assembled.
    matcher = ConsumerMatcher(reg)
    assert matcher.assemble_pipeline(MODEL, None) == []


def test_assemble_pipeline_unknown_model():
    reg = NodeRegistry()
    _make_registered_node(reg, "a", layers=(0, 24))
    matcher = ConsumerMatcher(reg)
    assert matcher.assemble_pipeline("not-a-real-model", None) == []


def test_assemble_pipeline_requires_active_status():
    reg = NodeRegistry()
    _make_registered_node(reg, "a", layers=(0, 24), status="pending")
    matcher = ConsumerMatcher(reg)
    assert matcher.assemble_pipeline(MODEL, None) == []


def test_configurable_weights_change_ordering():
    reg = NodeRegistry()
    # Equal capability, different locations and uptime.
    _make_registered_node(
        reg, "near-flappy",
        location={"lat": 40.7, "lon": -74.0}, layers=(0, 24), downtime=5,
    )
    _make_registered_node(
        reg, "far-stable",
        location={"lat": -33.8, "lon": 151.2}, layers=(0, 24), downtime=0,
    )
    consumer_loc = {"lat": 40.7, "lon": -74.0}

    prox_heavy = ConsumerMatcher(
        reg, scorer_weights={"proximity": 1.0, "uptime": 0.0,
                             "compute": 0.0, "load": 0.0},
    )
    uptime_heavy = ConsumerMatcher(
        reg, scorer_weights={"proximity": 0.0, "uptime": 1.0,
                             "compute": 0.0, "load": 0.0},
    )
    assert prox_heavy.find_best_nodes(MODEL, consumer_loc)[0]["node_id"] == "near-flappy"
    assert uptime_heavy.find_best_nodes(MODEL, consumer_loc)[0]["node_id"] == "far-stable"
