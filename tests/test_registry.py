"""Tests for the NodeRegistry."""

from __future__ import annotations

import hashlib
import time

import pytest

from src.signal.registry import NodeRegistry


class _FakeWS:
    """Minimal stand-in for websockets.ServerConnection — we never send."""

    closed = False

    async def close(self):
        self.closed = True


def _register_node(reg: NodeRegistry, nid: str, **kw):
    return reg.register(
        node_id=nid,
        ws=_FakeWS(),
        capabilities=kw.get("capabilities", {"device": "cpu", "ram_mb": 8000}),
        location=kw.get("location"),
        models_supported=kw.get("models_supported", []),
    )


# ---------------------------------------------------------------------------
def test_register_and_get_node():
    reg = NodeRegistry()
    rec = _register_node(reg, "0xabc")
    assert reg.node_count() == 1
    assert reg.get_node("0xabc") is rec


def test_register_replaces_existing():
    reg = NodeRegistry()
    _register_node(reg, "0xabc", capabilities={"device": "cpu", "ram_mb": 1000})
    rec2 = _register_node(reg, "0xabc", capabilities={"device": "cuda", "vram_mb": 24000})
    assert reg.node_count() == 1
    assert reg.get_node("0xabc") is rec2
    assert rec2.capabilities["device"] == "cuda"


def test_deregister_removes_and_increments_downtime():
    reg = NodeRegistry()
    rec = _register_node(reg, "0xabc")
    removed = reg.deregister("0xabc")
    assert removed is rec
    assert rec.downtime_events == 1
    assert reg.node_count() == 0


def test_reregister_preserves_downtime_counter():
    reg = NodeRegistry()
    _register_node(reg, "0xabc")
    reg.deregister("0xabc")
    rec = _register_node(reg, "0xabc")
    # downtime carries across reconnect — a flapping node can't reset its score.
    assert rec.downtime_events == 1


def test_deregister_unknown_returns_none():
    reg = NodeRegistry()
    assert reg.deregister("nope") is None


# ---------------------------------------------------------------------------
def test_update_heartbeat_refreshes_timestamp():
    reg = NodeRegistry()
    rec = _register_node(reg, "0xabc")
    rec.last_heartbeat = 0.0
    reg.update_heartbeat("0xabc", active_sessions=5, load=0.3)
    assert rec.last_heartbeat > 0.0
    assert rec.active_sessions == 5
    assert rec.load == pytest.approx(0.3)


def test_update_heartbeat_unknown_returns_none():
    reg = NodeRegistry()
    assert reg.update_heartbeat("nope", active_sessions=1) is None


def test_update_heartbeat_accepts_capabilities_update():
    reg = NodeRegistry()
    _register_node(reg, "0xabc", capabilities={"device": "cpu", "ram_mb": 1000})
    reg.update_heartbeat(
        "0xabc", capabilities={"device": "cuda", "vram_mb": 24000, "ram_mb": 32000},
    )
    rec = reg.get_node("0xabc")
    assert rec.capabilities["device"] == "cuda"
    assert rec.capabilities["vram_mb"] == 24000


# ---------------------------------------------------------------------------
def test_cleanup_stale_removes_old_nodes():
    reg = NodeRegistry()
    rec = _register_node(reg, "0xabc")
    rec.last_heartbeat = time.time() - 300.0
    stale = reg.cleanup_stale(timeout_seconds=120.0)
    assert stale == ["0xabc"]
    assert reg.node_count() == 0


def test_cleanup_stale_keeps_fresh_nodes():
    reg = NodeRegistry()
    _register_node(reg, "0xabc")
    stale = reg.cleanup_stale(timeout_seconds=120.0)
    assert stale == []
    assert reg.node_count() == 1


# ---------------------------------------------------------------------------
def test_get_active_nodes_filters_by_model():
    reg = NodeRegistry()
    r1 = _register_node(reg, "0x1", models_supported=["Qwen/Qwen2.5-0.5B"])
    r2 = _register_node(reg, "0x2", models_supported=["other-model"])
    r1.assignment_status = "active"
    r2.assignment_status = "active"
    active = reg.get_active_nodes(model_name="Qwen/Qwen2.5-0.5B")
    ids = [n["node_id"] for n in active]
    assert "0x1" in ids
    assert "0x2" not in ids


def test_get_active_nodes_empty_models_supported_matches_any():
    reg = NodeRegistry()
    r = _register_node(reg, "0x1", models_supported=[])
    r.assignment_status = "active"
    # Empty models_supported is treated as "any model" — useful for bootstrap.
    active = reg.get_active_nodes(model_name="anything")
    assert len(active) == 1


def test_get_active_nodes_excludes_inactive_states():
    reg = NodeRegistry()
    r = _register_node(reg, "0x1", models_supported=["m"])
    r.assignment_status = "rebalancing"
    # 'rebalancing' and 'loading' should NOT be returned.
    assert reg.get_active_nodes(model_name="m") == []


# ---------------------------------------------------------------------------
def test_merkle_root_empty_registry():
    reg = NodeRegistry()
    # Empty registry -> SHA256 of empty string, stable baseline.
    assert reg.merkle_root() == hashlib.sha256(b"").hexdigest()


def test_merkle_root_changes_on_registration():
    reg = NodeRegistry()
    empty = reg.merkle_root()
    _register_node(reg, "0x1")
    one = reg.merkle_root()
    _register_node(reg, "0x2")
    two = reg.merkle_root()
    assert empty != one != two


def test_merkle_root_stable_under_insertion_order():
    reg_a = NodeRegistry()
    _register_node(reg_a, "0x1")
    _register_node(reg_a, "0x2")

    reg_b = NodeRegistry()
    _register_node(reg_b, "0x2")
    _register_node(reg_b, "0x1")

    assert reg_a.merkle_root() == reg_b.merkle_root()


def test_merkle_root_changes_on_deregistration():
    reg = NodeRegistry()
    _register_node(reg, "0x1")
    _register_node(reg, "0x2")
    before = reg.merkle_root()
    reg.deregister("0x1")
    after = reg.merkle_root()
    assert before != after


# ---------------------------------------------------------------------------
def test_node_count():
    reg = NodeRegistry()
    assert reg.node_count() == 0
    for i in range(5):
        _register_node(reg, f"0x{i}")
    assert reg.node_count() == 5
