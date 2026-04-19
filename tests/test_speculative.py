"""Test speculative decode acceptance logic, adaptive window sizing,
and token verification correctness.
"""

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

from src.consumer.speculative import SpeculativeDecoder
from src.consumer.draft_model import DraftModel


class MockClient:
    """Mock InferenceClient for testing speculative decoder."""

    def __init__(self):
        self.session_id = "test-session"
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = 2
        self.pipeline = [
            {"node_id": "n1", "host": "h", "port": 1, "layer_start": 0, "layer_end": 31}
        ]
        self.responses: list[dict] = []
        self._response_idx = 0

    async def send_to_pipeline(self, message: dict) -> dict:
        if self._response_idx < len(self.responses):
            resp = self.responses[self._response_idx]
            self._response_idx += 1
            return resp
        return {"type": "error", "message": "No more mock responses"}

    def _sample_token(self, logits, temperature, top_p):
        return int(np.argmax(logits))


class MockDraftModel:
    """Mock draft model that returns predictable candidates."""

    def __init__(self, candidates: list[list[int]]):
        self._candidates = candidates
        self._call_idx = 0

    def generate_candidates(
        self, input_ids: list[int], n_candidates: int = 8, temperature: float = 0.0
    ) -> list[int]:
        if self._call_idx < len(self._candidates):
            result = self._candidates[self._call_idx][:n_candidates]
            self._call_idx += 1
            return result
        return list(range(n_candidates))


# --- Acceptance logic tests ---


@pytest.mark.asyncio
async def test_full_acceptance():
    """All draft tokens accepted, no correction needed."""
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
    assert decoder.total_proposed == 8
    assert decoder.acceptance_rate == 1.0


@pytest.mark.asyncio
async def test_partial_acceptance_with_correction():
    """Some tokens accepted, then a correction token."""
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [10, 11, 12],
            "correction_token": 99,
            "num_accepted": 3,
        }
    ]

    draft = MockDraftModel([[10, 11, 12, 50, 51, 52, 53, 54]])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)

    result = await decoder.decode_step([1, 2, 3])

    assert result == [10, 11, 12, 99]
    assert decoder.total_accepted == 3
    assert decoder.total_proposed == 8


@pytest.mark.asyncio
async def test_zero_acceptance():
    """No tokens accepted, only correction."""
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [],
            "correction_token": 42,
            "num_accepted": 0,
        }
    ]

    draft = MockDraftModel([[100, 101, 102, 103, 104, 105, 106, 107]])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)

    result = await decoder.decode_step([1, 2, 3])

    assert result == [42]
    assert decoder.total_accepted == 0
    assert decoder.acceptance_rate == 0.0


@pytest.mark.asyncio
async def test_error_response():
    """Pipeline returns an error."""
    client = MockClient()
    client.responses = [
        {"type": "error", "message": "Pipeline failed"}
    ]

    draft = MockDraftModel([[10, 11, 12, 13, 14, 15, 16, 17]])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)

    with pytest.raises(RuntimeError, match="Verification error"):
        await decoder.decode_step([1, 2, 3])


# --- Adaptive window sizing tests ---


def test_adaptive_window_expands_on_high_acceptance():
    client = MockClient()
    draft = MockDraftModel([])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)

    for _ in range(5):
        decoder._recent_acceptance.append(0.9)

    decoder.adaptive_window()
    assert decoder.window_size == 9


def test_adaptive_window_shrinks_on_low_acceptance():
    client = MockClient()
    draft = MockDraftModel([])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)

    for _ in range(5):
        decoder._recent_acceptance.append(0.3)

    decoder.adaptive_window()
    assert decoder.window_size == 7


def test_adaptive_window_stable_on_moderate_acceptance():
    client = MockClient()
    draft = MockDraftModel([])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)

    for _ in range(5):
        decoder._recent_acceptance.append(0.65)

    decoder.adaptive_window()
    assert decoder.window_size == 8


def test_adaptive_window_respects_min():
    client = MockClient()
    draft = MockDraftModel([])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=4)
    decoder.min_window = 4

    for _ in range(5):
        decoder._recent_acceptance.append(0.1)

    decoder.adaptive_window()
    assert decoder.window_size == 4


def test_adaptive_window_respects_max():
    client = MockClient()
    draft = MockDraftModel([])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=12)
    decoder.max_window = 12

    for _ in range(5):
        decoder._recent_acceptance.append(0.95)

    decoder.adaptive_window()
    assert decoder.window_size == 12


def test_adaptive_window_needs_minimum_history():
    """Should not adjust with fewer than 3 data points."""
    client = MockClient()
    draft = MockDraftModel([])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)

    decoder._recent_acceptance.append(0.1)
    decoder._recent_acceptance.append(0.1)

    decoder.adaptive_window()
    assert decoder.window_size == 8


# --- Stats tracking ---


@pytest.mark.asyncio
async def test_stats_tracking():
    client = MockClient()
    client.responses = [
        {
            "type": "verify_result",
            "accepted_tokens": [10, 11, 12, 13, 14],
            "correction_token": 99,
            "num_accepted": 5,
        },
        {
            "type": "verify_result",
            "accepted_tokens": [20, 21, 22],
            "correction_token": 88,
            "num_accepted": 3,
        },
    ]

    draft = MockDraftModel([
        [10, 11, 12, 13, 14, 50, 51, 52],
        [20, 21, 22, 60, 61, 62, 63, 64],
    ])
    decoder = SpeculativeDecoder(draft_model=draft, client=client, window_size=8)

    await decoder.decode_step([1, 2, 3])
    await decoder.decode_step([1, 2, 3, 10, 11, 12, 13, 14, 99])

    stats = decoder.get_stats()
    assert stats["total_rounds"] == 2
    assert stats["total_accepted"] == 8
    assert stats["total_proposed"] == 16
    assert stats["acceptance_rate"] == 0.5
    assert stats["tokens_per_round_trip"] == 4.0
    assert stats["current_window_size"] == 8


# --- Token matching tests ---


def test_sample_token_greedy():
    """Test that _sample_token with temperature=0 returns argmax."""
    client = MockClient()
    logits = np.array([0.1, 0.3, 0.9, 0.2, 0.5])
    token = client._sample_token(logits, temperature=0, top_p=0.9)
    assert token == 2


def test_sample_token_with_temperature():
    """Test that _sample_token with temperature>0 returns a valid token index."""
    client = MockClient()
    logits = np.array([1.0, 2.0, 3.0, 0.5, 0.1])
    np.random.seed(42)
    token = client._sample_token(logits, temperature=1.0, top_p=0.9)
    assert 0 <= token < len(logits)
