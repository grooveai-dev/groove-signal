"""Test that a model can be split into shards and the output of running all
shards sequentially matches the full model output.

Uses Qwen/Qwen2.5-0.5B as a tiny model for fast testing.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.node.shard_loader import forward_shard


MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def full_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def sample_input(tokenizer):
    text = "The quick brown fox jumps over the lazy dog"
    return tokenizer(text, return_tensors="pt")


def _make_shard(model, layer_start: int, layer_end: int) -> dict:
    """Build a shard dict from the full model without reloading."""
    config = model.config
    total_layers = len(model.model.layers)
    is_first = layer_start == 0
    is_last = layer_end == total_layers

    return {
        "layers": nn.ModuleList([model.model.layers[i] for i in range(layer_start, layer_end)]),
        "embed_tokens": model.model.embed_tokens if is_first else None,
        "norm": model.model.norm if is_last else None,
        "lm_head": model.lm_head if is_last else None,
        "rotary_emb": model.model.rotary_emb if hasattr(model.model, "rotary_emb") else None,
        "config": config,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "total_layers": total_layers,
    }


def test_full_model_produces_output(full_model, sample_input):
    with torch.no_grad():
        outputs = full_model(**sample_input)
    assert outputs.logits is not None
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == sample_input["input_ids"].shape[1]


def test_layer_split_matches_full_model(full_model, sample_input):
    """Split model into two halves and verify concatenated output matches full."""
    num_layers = len(full_model.model.layers)
    mid = num_layers // 2

    shard1 = _make_shard(full_model, 0, mid)
    shard2 = _make_shard(full_model, mid, num_layers)

    with torch.no_grad():
        full_outputs = full_model(**sample_input)
        full_logits = full_outputs.logits

        hidden, _ = forward_shard(shard1, sample_input["input_ids"])
        shard_logits, _ = forward_shard(shard2, hidden)

    atol = 1e-4
    assert torch.allclose(full_logits, shard_logits, atol=atol), (
        f"Max diff: {(full_logits - shard_logits).abs().max().item()}"
    )


def test_three_way_split_matches(full_model, sample_input):
    """Split model into three shards and verify output matches full model."""
    num_layers = len(full_model.model.layers)
    split1 = num_layers // 3
    split2 = 2 * num_layers // 3

    shard1 = _make_shard(full_model, 0, split1)
    shard2 = _make_shard(full_model, split1, split2)
    shard3 = _make_shard(full_model, split2, num_layers)

    with torch.no_grad():
        full_outputs = full_model(**sample_input)
        full_logits = full_outputs.logits

        hidden, _ = forward_shard(shard1, sample_input["input_ids"])
        hidden, _ = forward_shard(shard2, hidden)
        shard_logits, _ = forward_shard(shard3, hidden)

    atol = 1e-4
    assert torch.allclose(full_logits, shard_logits, atol=atol), (
        f"Max diff: {(full_logits - shard_logits).abs().max().item()}"
    )


def test_hidden_state_serialization_roundtrip(full_model, sample_input):
    """Test that hidden states survive numpy serialization (simulating network transfer)."""
    from src.common.tensor_transfer import serialize_tensor, deserialize_tensor

    num_layers = len(full_model.model.layers)
    mid = num_layers // 2
    shard1 = _make_shard(full_model, 0, mid)

    with torch.no_grad():
        hidden, _ = forward_shard(shard1, sample_input["input_ids"])
        original = hidden.clone()

        blob = serialize_tensor(hidden)
        restored = deserialize_tensor(blob)

    assert torch.allclose(original, restored, atol=1e-6), (
        f"Max diff: {(original - restored).abs().max().item()}"
    )


def test_argmax_token_matches_across_shards(full_model, sample_input):
    """Verify that the greedy next-token prediction is identical whether using
    the full model or running through shards sequentially."""
    num_layers = len(full_model.model.layers)
    mid = num_layers // 2

    shard1 = _make_shard(full_model, 0, mid)
    shard2 = _make_shard(full_model, mid, num_layers)

    with torch.no_grad():
        full_outputs = full_model(**sample_input)
        full_next = torch.argmax(full_outputs.logits[:, -1, :], dim=-1)

        hidden, _ = forward_shard(shard1, sample_input["input_ids"])
        shard_logits, _ = forward_shard(shard2, hidden)
        shard_next = torch.argmax(shard_logits[:, -1, :], dim=-1)

    assert full_next.item() == shard_next.item()
