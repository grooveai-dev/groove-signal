"""Tests for grammar + lookahead integration in speculative decoder."""

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from src.consumer.speculative import SpeculativeDecoder
from src.consumer.grammar import (
    GrammarConstraint,
    JSONGrammarConstraint,
    JSONState,
    CodeGrammarConstraint,
    CodeBracketState,
)


class MockClient:
    def __init__(self):
        self.session_id = "test-session"
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = 2
        self.responses: list[dict] = []
        self._response_idx = 0

    async def send_to_pipeline(self, message: dict) -> dict:
        if self._response_idx < len(self.responses):
            resp = self.responses[self._response_idx]
            self._response_idx += 1
            return resp
        return {"type": "error", "message": "No more mock responses"}


class MockDraftModel:
    def __init__(self, candidates: list[list[int]]):
        self._candidates = candidates
        self._call_idx = 0
        self.tokenizer = None

    def generate_candidates(
        self, input_ids: list[int], n_candidates: int = 8, temperature: float = 0.0
    ) -> list[int]:
        if self._call_idx < len(self._candidates):
            result = self._candidates[self._call_idx][:n_candidates]
            self._call_idx += 1
            return result
        return list(range(n_candidates))


class MockLookaheadDecoder:
    def __init__(self, candidates: list[int] | None = None):
        self._candidates = candidates
        self.call_count = 0

    def propose_candidates(self, input_ids, window_size):
        self.call_count += 1
        return self._candidates


class PassthroughGrammar(GrammarConstraint):
    """Grammar that accepts all tokens, for testing constraint wiring."""

    def mask_logits(self, logits, state):
        return logits

    def initial_state(self):
        return None

    def next_state(self, state, token_str):
        return None


class RejectOddGrammar(GrammarConstraint):
    """Grammar that masks odd-numbered token IDs, for testing filtering."""

    def __init__(self, vocab_size=100):
        self._vocab_size = vocab_size

    def mask_logits(self, logits, state):
        mask = torch.zeros_like(logits)
        for i in range(logits.shape[-1]):
            if i % 2 == 1:
                mask[i] = float('-inf')
        result = logits + mask
        if torch.all(result == float('-inf')):
            return logits
        return result

    def initial_state(self):
        return 0

    def next_state(self, state, token_str):
        return state


# --- Grammar + Speculative Integration ---


@pytest.mark.asyncio
async def test_decode_step_with_grammar():
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [10, 12, 14],
            "correction_token": None,
            "num_accepted": 3,
        }
    ]
    draft = MockDraftModel([[10, 12, 14, 16, 18, 20, 22, 24]])
    grammar = PassthroughGrammar()
    decoder = SpeculativeDecoder(
        draft_model=draft, client=client, window_size=8, grammar_constraint=grammar
    )

    result = await decoder.decode_step([1, 2, 3])
    assert result == [10, 12, 14]
    stats = decoder.get_stats()
    assert stats["constrained_acceptance_rate"] > 0


@pytest.mark.asyncio
async def test_decode_step_without_grammar():
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [10, 11],
            "correction_token": 99,
            "num_accepted": 2,
        }
    ]
    draft = MockDraftModel([[10, 11, 12, 13, 14, 15, 16, 17]])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)

    result = await decoder.decode_step([1, 2, 3])
    assert result == [10, 11, 99]
    stats = decoder.get_stats()
    assert stats["unconstrained_acceptance_rate"] > 0
    assert stats["constrained_acceptance_rate"] == 0.0


# --- Lookahead + Draft Model Fallback ---


@pytest.mark.asyncio
async def test_lookahead_used_when_sufficient_candidates():
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [100, 101, 102, 103, 104],
            "correction_token": None,
            "num_accepted": 5,
        }
    ]
    draft = MockDraftModel([[10, 11, 12, 13, 14, 15, 16, 17]])
    lookahead = MockLookaheadDecoder(candidates=[100, 101, 102, 103, 104])
    decoder = SpeculativeDecoder(
        draft_model=draft, client=client, window_size=8, lookahead_decoder=lookahead
    )

    result = await decoder.decode_step([1, 2, 3])
    assert result == [100, 101, 102, 103, 104]
    assert lookahead.call_count == 1
    stats = decoder.get_stats()
    assert stats["lookahead_rate"] == 1.0
    assert stats["draft_model_rate"] == 0.0


@pytest.mark.asyncio
async def test_draft_model_fallback_when_lookahead_short():
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [10, 11, 12],
            "correction_token": 99,
            "num_accepted": 3,
        }
    ]
    draft = MockDraftModel([[10, 11, 12, 13, 14, 15, 16, 17]])
    lookahead = MockLookaheadDecoder(candidates=[50, 51])  # too short (< 3)
    decoder = SpeculativeDecoder(
        draft_model=draft, client=client, window_size=8, lookahead_decoder=lookahead
    )

    result = await decoder.decode_step([1, 2, 3])
    assert result == [10, 11, 12, 99]
    stats = decoder.get_stats()
    assert stats["lookahead_rate"] == 0.0
    assert stats["draft_model_rate"] == 1.0


@pytest.mark.asyncio
async def test_draft_model_fallback_when_lookahead_returns_none():
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [10, 11],
            "correction_token": None,
            "num_accepted": 2,
        }
    ]
    draft = MockDraftModel([[10, 11, 12, 13, 14, 15, 16, 17]])
    lookahead = MockLookaheadDecoder(candidates=None)
    decoder = SpeculativeDecoder(
        draft_model=draft, client=client, window_size=8, lookahead_decoder=lookahead
    )

    result = await decoder.decode_step([1, 2, 3])
    assert result == [10, 11]
    stats = decoder.get_stats()
    assert stats["draft_model_rate"] == 1.0


@pytest.mark.asyncio
async def test_lookahead_with_grammar():
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [100, 102, 104],
            "correction_token": None,
            "num_accepted": 3,
        }
    ]
    draft = MockDraftModel([[10, 11, 12, 13, 14, 15, 16, 17]])
    lookahead = MockLookaheadDecoder(candidates=[100, 102, 104, 106, 108])
    grammar = PassthroughGrammar()
    decoder = SpeculativeDecoder(
        draft_model=draft, client=client, window_size=8,
        grammar_constraint=grammar, lookahead_decoder=lookahead,
    )

    result = await decoder.decode_step([1, 2, 3])
    assert result == [100, 102, 104]
    stats = decoder.get_stats()
    assert stats["lookahead_rate"] == 1.0
    assert stats["constrained_acceptance_rate"] > 0


# --- Stats tracking ---


@pytest.mark.asyncio
async def test_stats_include_new_fields():
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [10, 11, 12],
            "correction_token": None,
            "num_accepted": 3,
        }
    ]
    draft = MockDraftModel([[10, 11, 12, 13, 14, 15, 16, 17]])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)
    await decoder.decode_step([1, 2, 3])

    stats = decoder.get_stats()
    expected_keys = {
        "acceptance_rate", "tokens_per_round_trip", "total_rounds",
        "total_accepted", "total_proposed", "current_window_size",
        "lookahead_rate", "draft_model_rate",
        "lookahead_acceptance_rate", "draft_acceptance_rate",
        "constrained_acceptance_rate", "unconstrained_acceptance_rate",
    }
    assert expected_keys.issubset(set(stats.keys()))


@pytest.mark.asyncio
async def test_existing_tests_still_pass_with_new_params():
    """Verify backward compat: decoder without grammar/lookahead works as before."""
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [10, 11, 12, 13, 14, 15, 16, 17],
            "correction_token": None,
            "num_accepted": 8,
        }
    ]
    draft = MockDraftModel([[10, 11, 12, 13, 14, 15, 16, 17]])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)
    result = await decoder.decode_step([1, 2, 3])

    assert result == [10, 11, 12, 13, 14, 15, 16, 17]
    assert decoder.total_accepted == 8
    assert decoder.acceptance_rate == 1.0
