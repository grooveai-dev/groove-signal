"""Tests for the gaussian-decay scoring module."""

from __future__ import annotations

import math

import pytest

from src.signal import scoring


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------
def test_haversine_zero_distance():
    d = scoring.haversine_km(40.0, -73.0, 40.0, -73.0)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_haversine_known_distance_nyc_to_la():
    # NYC ~ (40.7128, -74.0060), LA ~ (34.0522, -118.2437). ~3935 km.
    d = scoring.haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3900 <= d <= 4000


def test_haversine_symmetric():
    d1 = scoring.haversine_km(51.5, -0.1, 48.85, 2.35)  # London -> Paris
    d2 = scoring.haversine_km(48.85, 2.35, 51.5, -0.1)
    assert d1 == pytest.approx(d2, rel=1e-9)


# ---------------------------------------------------------------------------
# Proximity
# ---------------------------------------------------------------------------
def test_proximity_same_location_is_one():
    loc = {"lat": 40.0, "lon": -73.0}
    assert scoring.proximity_score(loc, loc) == pytest.approx(1.0)


def test_proximity_far_apart_decays():
    near = {"lat": 40.0, "lon": -73.0}
    far = {"lat": -33.86, "lon": 151.21}  # Sydney
    assert scoring.proximity_score(near, far) < 0.01


def test_proximity_nearby_is_high():
    # ~500 km apart -> should score > 0.7.
    a = {"lat": 40.7, "lon": -74.0}   # NYC
    b = {"lat": 38.9, "lon": -77.0}   # DC-ish (~330 km from NYC)
    score = scoring.proximity_score(a, b)
    assert score > 0.7


def test_proximity_missing_location_is_neutral():
    loc = {"lat": 40.0, "lon": -73.0}
    assert scoring.proximity_score(None, loc) == scoring.NEUTRAL_PROXIMITY
    assert scoring.proximity_score(loc, None) == scoring.NEUTRAL_PROXIMITY
    assert scoring.proximity_score(None, None) == scoring.NEUTRAL_PROXIMITY


def test_proximity_malformed_location_is_neutral():
    loc = {"lat": 40.0, "lon": -73.0}
    assert scoring.proximity_score({"lat": "bogus"}, loc) == scoring.NEUTRAL_PROXIMITY


# ---------------------------------------------------------------------------
# Uptime
# ---------------------------------------------------------------------------
def test_uptime_zero_events_is_one():
    assert scoring.uptime_score(0) == pytest.approx(1.0)


def test_uptime_many_events_decays():
    assert scoring.uptime_score(10) < 0.01


def test_uptime_new_node_is_neutral():
    assert scoring.uptime_score(None) == scoring.NEUTRAL_UPTIME


def test_uptime_monotonic_decrease():
    scores = [scoring.uptime_score(k) for k in range(0, 8)]
    for a, b in zip(scores, scores[1:]):
        assert a >= b


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def test_compute_gpu_outranks_cpu():
    gpu = {"device": "cuda", "vram_mb": 24000, "ram_mb": 32000}
    cpu = {"device": "cpu", "vram_mb": 0, "ram_mb": 32000}
    max_cap = 24000
    s_gpu = scoring.compute_score(gpu, max_cap)
    s_cpu = scoring.compute_score(cpu, max_cap)
    assert s_gpu > s_cpu


def test_compute_floor_applies_to_weak_nodes():
    weak = {"device": "cpu", "ram_mb": 100}
    s = scoring.compute_score(weak, max_capacity_mb=100_000)
    assert s >= scoring.COMPUTE_FLOOR


def test_compute_normalized_to_one():
    gpu = {"device": "cuda", "vram_mb": 24000}
    s = scoring.compute_score(gpu, max_capacity_mb=24000)
    assert s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def test_load_idle_node_is_one():
    caps = {"device": "cuda", "vram_mb": 24000}
    assert scoring.load_score(caps, active_sessions=0) == pytest.approx(1.0)


def test_load_saturated_node_is_zero():
    caps = {"device": "cpu", "ram_mb": 1000}
    # max_sessions = 1000/1000 = 1; busy 5 -> clamped to 0.
    assert scoring.load_score(caps, active_sessions=5) == 0.0


def test_load_half_capacity_is_half():
    caps = {"device": "cuda", "vram_mb": 1000}
    # max_sessions = 1000/500 = 2; busy 1 -> 0.5.
    s = scoring.load_score(caps, active_sessions=1)
    assert s == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Composite / rank
# ---------------------------------------------------------------------------
def _make_node(node_id, device, vram, ram, location, downtime=0, active=0):
    return {
        "node_id": node_id,
        "capabilities": {"device": device, "vram_mb": vram, "ram_mb": ram},
        "location": location,
        "downtime_events": downtime,
        "active_sessions": active,
    }


def test_rank_nodes_orders_by_score_desc():
    consumer_loc = {"lat": 40.7, "lon": -74.0}
    nodes = [
        _make_node("far", "cpu", 0, 8000,
                   {"lat": -33.86, "lon": 151.21}, downtime=3, active=0),
        _make_node("near", "cuda", 24000, 32000,
                   {"lat": 40.7, "lon": -74.0}, downtime=0, active=0),
        _make_node("mid", "cuda", 8000, 16000,
                   {"lat": 41.8, "lon": -87.6}, downtime=1, active=0),  # Chicago
    ]
    ranked = scoring.rank_nodes(
        nodes, {"location": consumer_loc}, top_n=10,
    )
    ids = [r["node_id"] for r in ranked]
    assert ids[0] == "near"
    assert ids[-1] == "far"


def test_rank_nodes_top_n_trims():
    consumer_loc = {"lat": 0.0, "lon": 0.0}
    nodes = [
        _make_node(f"n{i}", "cuda", 8000, 16000, consumer_loc)
        for i in range(5)
    ]
    ranked = scoring.rank_nodes(nodes, {"location": consumer_loc}, top_n=3)
    assert len(ranked) == 3


def test_rank_nodes_empty():
    assert scoring.rank_nodes([], {"location": None}) == []


def test_score_node_weights_are_configurable():
    consumer = {"location": {"lat": 0, "lon": 0}}
    n = _make_node("x", "cuda", 16000, 32000,
                   {"lat": 50, "lon": 50}, downtime=0)
    # weights where proximity dominates should score lower (far from consumer)
    heavy_prox = scoring.score_node(
        n, consumer,
        weights={"proximity": 1.0, "uptime": 0.0, "compute": 0.0, "load": 0.0},
    )
    # weights where uptime dominates should score ~1.0 (zero downtime)
    heavy_uptime = scoring.score_node(
        n, consumer,
        weights={"proximity": 0.0, "uptime": 1.0, "compute": 0.0, "load": 0.0},
    )
    assert heavy_uptime > heavy_prox


def test_score_node_handles_missing_location_neutrally():
    consumer = {"location": None}
    n = _make_node("x", "cuda", 16000, 32000, None, downtime=0)
    s = scoring.score_node(n, consumer)
    # Must produce a finite score in [0, 1], not NaN.
    assert 0.0 <= s <= 1.0 and not math.isnan(s)


def test_weight_normalization_rescales():
    # Unnormalized weights should still produce bounded scores.
    consumer = {"location": {"lat": 0, "lon": 0}}
    n = _make_node("x", "cuda", 16000, 32000, {"lat": 0, "lon": 0}, downtime=0)
    s = scoring.score_node(
        n, consumer,
        weights={"proximity": 10, "uptime": 10, "compute": 10, "load": 10},
    )
    assert 0.0 <= s <= 1.0
