"""Tests for src/relay/scheduler.py — pure layer-assignment logic."""

import pytest

from src.relay.scheduler import (
    MODEL_REGISTRY,
    assign_layers,
    calculate_rebalance,
    get_model_info,
    validate_coverage,
)


MODEL = "Qwen/Qwen2.5-0.5B"
TOTAL_LAYERS = MODEL_REGISTRY[MODEL]["total_layers"]


def _caps(node_id, **fields):
    out = {"node_id": node_id}
    out.update(fields)
    return out


def test_get_model_info_unknown_model():
    with pytest.raises(ValueError):
        get_model_info("does-not-exist")


def test_assign_layers_rejects_empty():
    with pytest.raises(ValueError):
        assign_layers([], MODEL)


def test_assign_layers_rejects_more_nodes_than_layers():
    many = [_caps(f"n{i}", device="cuda", vram_mb=10_000) for i in range(TOTAL_LAYERS + 1)]
    with pytest.raises(ValueError):
        assign_layers(many, MODEL)


def test_assign_layers_single_node_gets_all():
    out = assign_layers([_caps("solo", device="cpu", ram_mb=16000)], MODEL)
    assert out == {"solo": (0, TOTAL_LAYERS)}


def test_assign_layers_two_equal_nodes_split_evenly():
    nodes = [
        _caps("a", device="cpu", ram_mb=16000),
        _caps("b", device="cpu", ram_mb=16000),
    ]
    out = assign_layers(nodes, MODEL)
    assert sum(end - start for start, end in out.values()) == TOTAL_LAYERS
    assert validate_coverage(out, MODEL)
    # Each side should get roughly half.
    spans = sorted(out.values())
    assert abs((spans[0][1] - spans[0][0]) - (spans[1][1] - spans[1][0])) <= 1


def test_assign_layers_gpu_gets_more_than_cpu():
    nodes = [
        _caps("gpu", device="cuda", vram_mb=24_000),
        _caps("cpu", device="cpu", ram_mb=32_000),
    ]
    out = assign_layers(nodes, MODEL)
    gpu_span = out["gpu"][1] - out["gpu"][0]
    cpu_span = out["cpu"][1] - out["cpu"][0]
    assert gpu_span > cpu_span
    assert validate_coverage(out, MODEL)


def test_assign_layers_coverage_is_contiguous():
    nodes = [
        _caps("a", device="cuda", vram_mb=8_000),
        _caps("b", device="cuda", vram_mb=4_000),
        _caps("c", device="cpu", ram_mb=16_000),
    ]
    out = assign_layers(nodes, MODEL)
    assert validate_coverage(out, MODEL)
    spans = sorted(out.values())
    assert spans[0][0] == 0
    assert spans[-1][1] == TOTAL_LAYERS
    for i in range(len(spans) - 1):
        assert spans[i][1] == spans[i + 1][0]


def test_assign_layers_zero_capacity_fallback():
    nodes = [_caps("a"), _caps("b"), _caps("c")]
    out = assign_layers(nodes, MODEL)
    assert sum(e - s for s, e in out.values()) == TOTAL_LAYERS
    assert validate_coverage(out, MODEL)


def test_assign_layers_every_node_gets_at_least_one():
    nodes = [
        _caps("big", device="cuda", vram_mb=80_000),
        _caps("tiny", device="cpu", ram_mb=100),
    ]
    out = assign_layers(nodes, MODEL)
    for start, end in out.values():
        assert end > start


def test_calculate_rebalance_new_node_reduces_existing():
    current = {"a": (0, TOTAL_LAYERS)}
    new_caps = [
        _caps("a", device="cuda", vram_mb=8_000),
        _caps("b", device="cuda", vram_mb=8_000),
    ]
    new_assignments, affected = calculate_rebalance(current, new_caps, MODEL)
    assert validate_coverage(new_assignments, MODEL)
    assert "a" in affected and "b" in affected
    assert new_assignments["a"] != current["a"]


def test_calculate_rebalance_no_change_yields_empty_affected():
    current = {"a": (0, 12), "b": (12, TOTAL_LAYERS)}
    caps = [
        _caps("a", device="cuda", vram_mb=8_000),
        _caps("b", device="cuda", vram_mb=8_000),
    ]
    new_assignments, affected = calculate_rebalance(current, caps, MODEL)
    assert validate_coverage(new_assignments, MODEL)
    if new_assignments == current:
        assert affected == []


def test_validate_coverage_detects_gap():
    assert not validate_coverage({"a": (0, 10), "b": (12, TOTAL_LAYERS)}, MODEL)


def test_validate_coverage_detects_overlap():
    assert not validate_coverage({"a": (0, 15), "b": (10, TOTAL_LAYERS)}, MODEL)


def test_validate_coverage_detects_missing_head():
    assert not validate_coverage({"a": (2, TOTAL_LAYERS)}, MODEL)


def test_validate_coverage_detects_missing_tail():
    assert not validate_coverage({"a": (0, TOTAL_LAYERS - 2)}, MODEL)


def test_validate_coverage_empty_is_invalid():
    assert not validate_coverage({}, MODEL)
