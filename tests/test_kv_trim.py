"""Tests for KV cache trim functionality."""

import torch
import pytest

from src.node.kv_cache import KVCacheManager, SessionKVCache


def _populate_cache(session_cache: SessionKVCache, seq_len: int, num_layers: int = 2):
    """Populate a DynamicCache with fake KV tensors via the update() API."""
    cache = session_cache.cache
    for layer_idx in range(num_layers):
        k = torch.randn(1, 4, seq_len, 64)
        v = torch.randn(1, 4, seq_len, 64)
        cache.update(k, v, layer_idx)


class TestKVCacheTrim:
    def test_normal_trim(self):
        mgr = KVCacheManager()
        session = mgr.create_session("s1", num_layers=2)
        _populate_cache(session, seq_len=20, num_layers=2)

        mgr.trim("s1", count=5)

        assert session.cache.get_seq_length() == 15

    def test_trim_all(self):
        mgr = KVCacheManager()
        session = mgr.create_session("s1", num_layers=2)
        _populate_cache(session, seq_len=10, num_layers=2)

        mgr.trim("s1", count=10)

        assert session.cache.get_seq_length() == 0

    def test_over_trim_clamps_to_empty(self):
        mgr = KVCacheManager()
        session = mgr.create_session("s1", num_layers=2)
        _populate_cache(session, seq_len=5, num_layers=2)

        mgr.trim("s1", count=100)

        assert session.cache.get_seq_length() == 0

    def test_trim_missing_session_is_noop(self):
        mgr = KVCacheManager()
        mgr.trim("nonexistent", count=5)

    def test_trim_zero_count_is_noop(self):
        mgr = KVCacheManager()
        session = mgr.create_session("s1", num_layers=2)
        _populate_cache(session, seq_len=10, num_layers=2)

        mgr.trim("s1", count=0)

        assert session.cache.get_seq_length() == 10

    def test_trim_preserves_other_sessions(self):
        mgr = KVCacheManager()
        s1 = mgr.create_session("s1", num_layers=1)
        s2 = mgr.create_session("s2", num_layers=1)
        _populate_cache(s1, seq_len=20, num_layers=1)
        _populate_cache(s2, seq_len=20, num_layers=1)

        mgr.trim("s1", count=5)

        assert s1.cache.get_seq_length() == 15
        assert s2.cache.get_seq_length() == 20

    def test_trim_multiple_layers(self):
        mgr = KVCacheManager()
        session = mgr.create_session("s1", num_layers=4)
        _populate_cache(session, seq_len=30, num_layers=4)

        mgr.trim("s1", count=10)

        assert session.cache.get_seq_length() == 20
        for layer in session.cache.layers:
            assert layer.keys.shape[2] == 20
            assert layer.values.shape[2] == 20
