"""Tests for LookaheadDecoder N-gram context matching."""

import pytest

from src.consumer.lookahead import LookaheadDecoder


class TestLookaheadBasic:
    def test_no_context_returns_empty(self):
        la = LookaheadDecoder(min_n=3, max_n=5)
        assert la.propose_candidates([1, 2, 3]) == []

    def test_empty_input_returns_empty(self):
        la = LookaheadDecoder(min_n=3, max_n=5)
        la.update_context([1, 2, 3, 4, 5])
        assert la.propose_candidates([]) == []

    def test_finds_repeated_pattern(self):
        la = LookaheadDecoder(min_n=3, max_n=5)
        pattern = [10, 20, 30, 40, 50]
        context = pattern * 3
        la.update_context(context)
        candidates = la.propose_candidates([10, 20, 30], n_candidates=2)
        assert len(candidates) > 0
        assert candidates[0] == 40

    def test_returns_empty_for_unseen_pattern(self):
        la = LookaheadDecoder(min_n=3, max_n=5)
        la.update_context([1, 2, 3, 4, 5, 6])
        candidates = la.propose_candidates([99, 98, 97])
        assert candidates == []

    def test_respects_n_candidates_limit(self):
        la = LookaheadDecoder(min_n=3, max_n=8)
        pattern = [10, 20, 30, 40, 50, 60, 70, 80]
        la.update_context(pattern * 5)
        candidates = la.propose_candidates([10, 20, 30], n_candidates=3)
        assert len(candidates) <= 3


class TestLookaheadStats:
    def test_stats_tracking(self):
        la = LookaheadDecoder(min_n=3, max_n=5)
        la.update_context([1, 2, 3, 4, 5] * 3)
        la.propose_candidates([1, 2, 3])
        assert la.stats["total_queries"] == 1
        assert la.stats["total_matches"] == 1
        assert la.stats["match_rate"] == 1.0
        assert la.stats["longest_match"] > 0

    def test_miss_updates_query_count(self):
        la = LookaheadDecoder(min_n=3, max_n=5)
        la.update_context([1, 2, 3, 4, 5])
        la.propose_candidates([99, 98, 97])
        assert la.stats["total_queries"] == 1
        assert la.stats["total_matches"] == 0
        assert la.stats["match_rate"] == 0.0

    def test_avg_match_length(self):
        la = LookaheadDecoder(min_n=3, max_n=8)
        pattern = [10, 20, 30, 40, 50, 60]
        la.update_context(pattern * 5)
        la.propose_candidates([10, 20, 30], n_candidates=4)
        assert la.stats["avg_match_length"] > 0


class TestLookaheadContextUpdate:
    def test_update_context_replaces_old(self):
        la = LookaheadDecoder(min_n=3, max_n=5)
        la.update_context([1, 2, 3, 4, 5] * 3)
        assert la.propose_candidates([1, 2, 3]) != []
        la.update_context([10, 20, 30, 40, 50] * 3)
        assert la.propose_candidates([1, 2, 3]) == []
        assert la.propose_candidates([10, 20, 30]) != []

    def test_short_context_still_works(self):
        la = LookaheadDecoder(min_n=3, max_n=5)
        la.update_context([1, 2, 3, 4])
        candidates = la.propose_candidates([1, 2, 3])
        assert candidates == [] or candidates[0] == 4
