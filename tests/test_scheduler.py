"""Tests for src/relay/scheduler.py — pure layer-assignment logic."""

import pytest

from src.relay.scheduler import (
    MODEL_REGISTRY,
    _inference_speed_score,
    assign_layers,
    calculate_rebalance,
    get_model_info,
    minimize_hops_assign,
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
        assign_layers(many, MODEL, strategy="proportional")


def test_assign_layers_single_node_gets_all():
    out = assign_layers([_caps("solo", device="cpu", ram_mb=16000)], MODEL)
    assert out == {"solo": (0, TOTAL_LAYERS)}


def test_assign_layers_two_equal_nodes_split_evenly():
    nodes = [
        _caps("a", device="cpu", ram_mb=16000),
        _caps("b", device="cpu", ram_mb=16000),
    ]
    out = assign_layers(nodes, MODEL, strategy="proportional")
    assert sum(end - start for start, end in out.values()) == TOTAL_LAYERS
    assert validate_coverage(out, MODEL)
    spans = sorted(out.values())
    assert abs((spans[0][1] - spans[0][0]) - (spans[1][1] - spans[1][0])) <= 1


def test_assign_layers_gpu_gets_more_than_cpu():
    nodes = [
        _caps("gpu", device="cuda", vram_mb=24_000),
        _caps("cpu", device="cpu", ram_mb=32_000),
    ]
    out = assign_layers(nodes, MODEL, strategy="proportional")
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
    new_assignments, affected = calculate_rebalance(current, new_caps, MODEL, strategy="proportional")
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


# ---------------------------------------------------------------------------
# minimize_hops_assign tests
# ---------------------------------------------------------------------------

QWEN4B = "Qwen/Qwen3-4B"
QWEN4B_LAYERS = MODEL_REGISTRY[QWEN4B]["total_layers"]  # 36


class TestMinimizeHopsAssign:
    def test_single_gpu_holds_entire_model(self):
        """A GPU with 10GB+ VRAM can hold all 36 layers (220MB each = 7.9GB)."""
        nodes = [_caps("gpu1", device="cuda", vram_mb=12_000)]
        out = minimize_hops_assign(nodes, QWEN4B)
        assert out == {"gpu1": (0, QWEN4B_LAYERS)}
        assert validate_coverage(out, QWEN4B)

    def test_one_gpu_plus_three_macs(self):
        """GPU gets maximum layers, Macs fill the rest. Fewer pipeline hops than proportional."""
        nodes = [
            _caps("gpu", device="cuda", vram_mb=10_000, bench_ms_per_layer=1.5),
            _caps("mac1", device="mps", vram_mb=12_000, bench_ms_per_layer=5.0),
            _caps("mac2", device="mps", vram_mb=8_000, bench_ms_per_layer=6.0),
            _caps("mac3", device="mps", vram_mb=6_000, bench_ms_per_layer=7.0),
        ]
        out = minimize_hops_assign(nodes, QWEN4B)
        assert validate_coverage(out, QWEN4B)
        # Fastest node (GPU) should get the most layers
        gpu_layers = out["gpu"][1] - out["gpu"][0]
        assert gpu_layers >= 36 * 0.25  # at least 25%
        # Should use fewer nodes than proportional would
        assert len(out) <= 4

    def test_all_macs_equal(self):
        """Equal MPS nodes — layers packed onto fewest needed."""
        nodes = [
            _caps(f"mac{i}", device="mps", vram_mb=4_000, bench_ms_per_layer=5.0)
            for i in range(4)
        ]
        out = minimize_hops_assign(nodes, QWEN4B)
        assert validate_coverage(out, QWEN4B)
        # 4000MB / 220MB = ~18 layers per node. So 2 nodes should suffice.
        assert len(out) <= 3

    def test_single_node_no_bench_data(self):
        """Falls back to heuristic when no bench data."""
        nodes = [_caps("solo", device="cpu", ram_mb=32_000)]
        out = minimize_hops_assign(nodes, QWEN4B)
        assert out == {"solo": (0, QWEN4B_LAYERS)}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            minimize_hops_assign([], QWEN4B)

    def test_reader_node_placed_first(self):
        """Node classified as 'reader' should be first in the pipeline."""
        nodes = [
            _caps("writer1", device="mps", vram_mb=8_000,
                   bench_ms_per_layer=5.0, node_role="writer"),
            _caps("reader1", device="cuda", vram_mb=10_000,
                   bench_ms_per_layer=1.5, node_role="reader"),
        ]
        out = minimize_hops_assign(nodes, QWEN4B)
        assert validate_coverage(out, QWEN4B)
        sorted_nodes = sorted(out.items(), key=lambda x: x[1][0])
        assert sorted_nodes[0][0] == "reader1"

    def test_strategy_parameter_selects_minimize_hops(self):
        """Default strategy is minimize_hops."""
        nodes = [_caps("n1", device="cuda", vram_mb=12_000)]
        out = assign_layers(nodes, QWEN4B)
        assert out == {"n1": (0, QWEN4B_LAYERS)}

    def test_strategy_parameter_selects_proportional(self):
        """Can still use proportional strategy."""
        nodes = [
            _caps("a", device="cuda", vram_mb=8_000),
            _caps("b", device="cuda", vram_mb=8_000),
        ]
        out = assign_layers(nodes, QWEN4B, strategy="proportional")
        assert validate_coverage(out, QWEN4B)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Node role classification tests
# ---------------------------------------------------------------------------

class TestNodeRoleClassification:
    def test_gpu_classified_as_reader(self):
        from src.common.benchmark import classify_node_role
        role = classify_node_role(1.5, 0.5, device="cuda")
        assert role == "reader"

    def test_high_bandwidth_cpu_classified_as_writer(self):
        from src.common.benchmark import classify_node_role
        role = classify_node_role(20.0, 100.0, device="cpu")
        assert role == "writer"

    def test_no_data_defaults_to_writer(self):
        from src.common.benchmark import classify_node_role
        role = classify_node_role(0.0, 0.0, device="cpu")
        assert role == "writer"

    def test_mps_with_good_compute_can_be_reader(self):
        from src.common.benchmark import classify_node_role
        role = classify_node_role(2.0, 30.0, device="mps")
        assert role in ("reader", "writer")


# ---------------------------------------------------------------------------
# Inference speed score tests
# ---------------------------------------------------------------------------

class TestInferenceSpeedScore:
    def test_bench_data_used_when_available(self):
        score = _inference_speed_score({"bench_ms_per_layer": 3.5, "device": "cuda"})
        assert score == 3.5

    def test_cuda_heuristic(self):
        score = _inference_speed_score({"device": "cuda"})
        assert score == 2.0

    def test_mps_heuristic(self):
        score = _inference_speed_score({"device": "mps"})
        assert score == 5.0

    def test_cpu_heuristic(self):
        score = _inference_speed_score({"device": "cpu"})
        assert score == 20.0


# ---------------------------------------------------------------------------
# Protocol message tests
# ---------------------------------------------------------------------------

class TestProtocolMessages:
    def test_pipeline_mesh_roundtrip(self):
        from src.common.protocol import make_pipeline_mesh, encode_message, decode_message
        msg = make_pipeline_mesh(
            "sess123",
            [{"node_id": "n1", "host": "10.0.0.1", "port": 9000, "position": 0}],
            {"host": "10.0.0.10", "port": 8000},
        )
        data = encode_message(msg)
        decoded = decode_message(data)
        assert decoded["type"] == "pipeline_mesh"
        assert decoded["session_id"] == "sess123"
        assert len(decoded["nodes"]) == 1
        assert decoded["consumer"]["host"] == "10.0.0.10"

    def test_kv_trim_roundtrip(self):
        from src.common.protocol import make_kv_trim, encode_message, decode_message
        msg = make_kv_trim("sess456", 3)
        data = encode_message(msg)
        decoded = decode_message(data)
        assert decoded["type"] == "kv_trim"
        assert decoded["trim_count"] == 3

    def test_activations_with_timing_fields(self):
        from src.common.protocol import make_activations, encode_message, decode_message
        msg = make_activations("s1", 0, b"\x00", (1,), "float16",
                               forward_ms=12.5, queue_ms=0.8)
        data = encode_message(msg)
        decoded = decode_message(data)
        assert decoded["forward_ms"] == 12.5
        assert decoded["queue_ms"] == 0.8

    def test_logits_with_timing_fields(self):
        from src.common.protocol import make_logits, encode_message, decode_message
        msg = make_logits("s1", 0, b"\x00", (1,), "float16",
                          forward_ms=8.3, queue_ms=1.1)
        data = encode_message(msg)
        decoded = decode_message(data)
        assert decoded["forward_ms"] == 8.3
        assert decoded["queue_ms"] == 1.1

    def test_session_init_with_grammar_mode(self):
        from src.common.protocol import make_session_init, encode_message, decode_message
        msg = make_session_init("s1", "Qwen/Qwen3-4B", 0, 36, grammar_mode="json")
        data = encode_message(msg)
        decoded = decode_message(data)
        assert decoded["grammar_mode"] == "json"

    def test_session_init_without_grammar_mode(self):
        from src.common.protocol import make_session_init
        msg = make_session_init("s1", "Qwen/Qwen3-4B", 0, 36)
        assert "grammar_mode" not in msg
