"""Local draft model for speculative decoding."""

import torch
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class DraftModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=False,
        )

        dtype = torch.float32
        if self.device == "mps":
            dtype = torch.float16
        elif self.device == "cuda":
            dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, trust_remote_code=False,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_candidates(
        self,
        input_ids: list[int],
        n_candidates: int = 8,
        temperature: float = 0.0,
    ) -> list[int]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        candidates: list[int] = []

        past_key_values = None
        current_ids = ids

        for _ in range(n_candidates):
            outputs = self.model(
                current_ids, past_key_values=past_key_values, use_cache=True
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if temperature <= 0:
                next_token = int(torch.argmax(logits, dim=-1).item())
            else:
                scaled = logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            candidates.append(next_token)
            current_ids = torch.tensor(
                [[next_token]], dtype=torch.long, device=self.device
            )

        return candidates

    @torch.no_grad()
    def get_logits(self, input_ids: list[int]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        outputs = self.model(ids)
        return outputs.logits[0].cpu().float().numpy()
