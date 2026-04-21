"""Speculative decoding for distributed inference."""

import asyncio
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


class SpeculativeDecoder:
    def __init__(
        self,
        draft_model: DraftModel,
        client,
        window_size: int = 8,
    ):
        self.draft_model = draft_model
        self.client = client
        self.window_size = window_size
        self.min_window = 4
        self.max_window = 12

        self.total_accepted = 0
        self.total_proposed = 0
        self.total_rounds = 0
        self._recent_acceptance: deque[float] = deque(maxlen=10)

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

    async def decode_step(
        self, input_ids: list[int], temperature: float = 0.0
    ) -> list[int]:
        candidates = await asyncio.to_thread(
            self.draft_model.generate_candidates,
            input_ids,
            self.window_size,
            temperature,
        )

        msg = {
            "type": SPEC_WINDOW,
            "session_id": self.client.session_id,
            "candidate_ids": candidates,
            "turn": self.total_rounds,
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

        from src.consumer.client import _logits_from_response

        logits = _logits_from_response(response)
        first_token = self.client._sample_token(
            logits[-1] if len(logits.shape) > 1 else logits, temperature, top_p
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
        return {
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_round_trip": self.tokens_per_round_trip,
            "total_rounds": self.total_rounds,
            "total_accepted": self.total_accepted,
            "total_proposed": self.total_proposed,
            "current_window_size": self.window_size,
        }
