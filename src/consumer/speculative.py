"""Speculative decoding for distributed inference."""

import asyncio
import logging
from collections import deque
from typing import AsyncGenerator

import numpy as np
import torch

from src.common.protocol import (
    encode_message,
    ACTIVATIONS,
    SPEC_WINDOW,
    ERROR,
)
from src.common.tensor_transfer import serialize_tensor, deserialize_tensor
from src.consumer.draft_model import DraftModel
from src.consumer.grammar import GrammarConstraint

logger = logging.getLogger(__name__)

try:
    from src.consumer.lookahead import LookaheadDecoder
except ImportError:
    LookaheadDecoder = None


class SpeculativeDecoder:
    def __init__(
        self,
        draft_model: DraftModel,
        client,
        window_size: int = 8,
        grammar_constraint: GrammarConstraint | None = None,
        lookahead_decoder=None,
    ):
        self.draft_model = draft_model
        self.client = client
        self.window_size = window_size
        self.min_window = 4
        self.max_window = 12
        self.grammar_constraint = grammar_constraint
        self.lookahead_decoder = lookahead_decoder

        self.total_accepted = 0
        self.total_proposed = 0
        self.total_rounds = 0
        self._recent_acceptance: deque[float] = deque(maxlen=10)

        self._lookahead_steps = 0
        self._draft_model_steps = 0
        self._unconstrained_steps = 0
        self._lookahead_accepted = 0
        self._lookahead_proposed = 0
        self._draft_accepted = 0
        self._draft_proposed = 0
        self._constrained_accepted = 0
        self._constrained_proposed = 0
        self._unconstrained_accepted = 0
        self._unconstrained_proposed = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed

    @property
    def tokens_per_round_trip(self) -> float:
        if self.total_rounds == 0:
            return 0.0
        return self.total_accepted / self.total_rounds

    def _apply_grammar(self, candidates: list[int], grammar_state=None) -> list[int]:
        if self.grammar_constraint is None:
            return candidates
        if grammar_state is None:
            grammar_state = self.grammar_constraint.initial_state()

        filtered: list[int] = []
        state = grammar_state
        for token_id in candidates:
            logits = torch.zeros(self.draft_model.tokenizer.vocab_size if hasattr(self.draft_model, 'tokenizer') and self.draft_model.tokenizer else max(token_id + 1, 32000))
            logits[token_id] = 1.0
            masked = self.grammar_constraint.mask_logits(logits, state)
            if masked[token_id] != float('-inf'):
                filtered.append(token_id)
                token_str = ""
                if hasattr(self.draft_model, 'tokenizer') and self.draft_model.tokenizer:
                    try:
                        token_str = self.draft_model.tokenizer.decode([token_id])
                    except Exception:
                        pass
                state = self.grammar_constraint.next_state(state, token_str)
            else:
                break
        return filtered if filtered else candidates

    async def decode_step(
        self, input_ids: list[int], temperature: float = 0.0
    ) -> list[int]:
        source = 'draft_model'
        candidates = None

        if self.lookahead_decoder is not None:
            try:
                la_candidates = self.lookahead_decoder.propose_candidates(
                    input_ids, self.window_size
                )
                if la_candidates and len(la_candidates) >= 3:
                    candidates = la_candidates
                    source = 'lookahead'
            except Exception:
                logger.debug("Lookahead proposal failed, falling back to draft model")

        if candidates is None:
            candidates = await asyncio.to_thread(
                self.draft_model.generate_candidates,
                input_ids,
                self.window_size,
                temperature,
            )
            source = 'draft_model'

        is_constrained = self.grammar_constraint is not None
        if is_constrained:
            candidates = self._apply_grammar(candidates)

        if source == 'lookahead':
            self._lookahead_steps += 1
        else:
            self._draft_model_steps += 1

        msg = {
            "type": SPEC_WINDOW,
            "session_id": self.client.session_id,
            "candidate_ids": candidates,
            "turn": self.total_rounds,
            "seq_pos": len(input_ids),
        }
        response = await self.client.send_to_pipeline(msg)

        if response["type"] == ERROR:
            raise RuntimeError(f"Verification error: {response['message']}")

        accepted_tokens: list[int] = response["accepted_tokens"]
        correction_token = response.get("correction_token")
        num_accepted: int = response["num_accepted"]

        self.total_rounds += 1
        self.total_proposed += len(candidates)
        self.total_accepted += num_accepted

        if source == 'lookahead':
            self._lookahead_accepted += num_accepted
            self._lookahead_proposed += len(candidates)
        else:
            self._draft_accepted += num_accepted
            self._draft_proposed += len(candidates)

        if is_constrained:
            self._constrained_accepted += num_accepted
            self._constrained_proposed += len(candidates)
        else:
            self._unconstrained_accepted += num_accepted
            self._unconstrained_proposed += len(candidates)

        round_rate = num_accepted / len(candidates) if candidates else 0.0
        self._recent_acceptance.append(round_rate)
        self.adaptive_window()

        result = list(accepted_tokens)
        if correction_token is not None:
            result.append(correction_token)

        return result

    def adaptive_window(self) -> None:
        if len(self._recent_acceptance) < 3:
            return

        recent_rate = sum(self._recent_acceptance) / len(self._recent_acceptance)

        if recent_rate > 0.8:
            self.window_size = min(self.max_window, self.window_size + 1)
        elif recent_rate < 0.5:
            self.window_size = max(self.min_window, self.window_size - 1)

    async def full_generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[list[int], None]:
        generated = list(prompt_ids)
        tokens_generated = 0
        eos_id = self.client.tokenizer.eos_token_id

        response = await self.client._prefill_pipelined(prompt_ids)

        if response["type"] == ERROR:
            raise RuntimeError(f"Prefill error: {response['message']}")

        from src.consumer.client import _logits_from_response, _last_token_logits

        logits = _logits_from_response(response)
        first_token = self.client._sample_token(
            _last_token_logits(logits), temperature, top_p
        )
        generated.append(first_token)
        tokens_generated += 1
        yield [first_token]

        while tokens_generated < max_tokens:
            if generated[-1] == eos_id:
                break

            new_tokens = await self.decode_step(generated, temperature)
            if not new_tokens:
                break

            if eos_id in new_tokens:
                eos_pos = new_tokens.index(eos_id)
                if eos_pos > 0:
                    yield new_tokens[:eos_pos]
                return

            generated.extend(new_tokens)
            tokens_generated += len(new_tokens)
            yield new_tokens

    def get_stats(self) -> dict:
        total_steps = self._lookahead_steps + self._draft_model_steps + self._unconstrained_steps
        return {
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_round_trip": self.tokens_per_round_trip,
            "total_rounds": self.total_rounds,
            "total_accepted": self.total_accepted,
            "total_proposed": self.total_proposed,
            "current_window_size": self.window_size,
            "lookahead_rate": self._lookahead_steps / total_steps if total_steps else 0.0,
            "draft_model_rate": self._draft_model_steps / total_steps if total_steps else 0.0,
            "lookahead_acceptance_rate": (
                self._lookahead_accepted / self._lookahead_proposed
                if self._lookahead_proposed else 0.0
            ),
            "draft_acceptance_rate": (
                self._draft_accepted / self._draft_proposed
                if self._draft_proposed else 0.0
            ),
            "constrained_acceptance_rate": (
                self._constrained_accepted / self._constrained_proposed
                if self._constrained_proposed else 0.0
            ),
            "unconstrained_acceptance_rate": (
                self._unconstrained_accepted / self._unconstrained_proposed
                if self._unconstrained_proposed else 0.0
            ),
        }
