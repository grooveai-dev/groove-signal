"""Tests for grammar-constrained speculation."""

import torch
import pytest
from unittest.mock import MagicMock

from src.consumer.grammar import (
    JSONGrammarConstraint,
    JSONState,
    CodeGrammarConstraint,
    CodeBracketState,
)


class FakeTokenizer:
    def __init__(self, vocab: dict[int, str]):
        self._vocab = vocab
        self.vocab_size = max(vocab.keys()) + 1 if vocab else 0

    def decode(self, ids):
        return "".join(self._vocab.get(i, "") for i in ids)


def _make_json_constraint(vocab: dict[int, str]) -> JSONGrammarConstraint:
    return JSONGrammarConstraint(FakeTokenizer(vocab))


def _make_code_constraint(vocab: dict[int, str]) -> CodeGrammarConstraint:
    return CodeGrammarConstraint(FakeTokenizer(vocab))


# --- JSONGrammarConstraint tests ---


class TestJSONGrammarMasking:
    def test_start_state_allows_object_and_array(self):
        vocab = {0: "{", 1: "[", 2: '"', 3: "x", 4: "1"}
        gc = _make_json_constraint(vocab)
        logits = torch.zeros(5)
        result = gc.mask_logits(logits, JSONState.START)
        assert result[0].item() != float("-inf")  # {
        assert result[1].item() != float("-inf")  # [
        assert result[2].item() != float("-inf")  # "
        assert result[3].item() == float("-inf")  # x -- not valid JSON start
        assert result[4].item() != float("-inf")  # 1 -- number

    def test_after_colon_allows_values(self):
        vocab = {0: '"', 1: "1", 2: "{", 3: "[", 4: "t", 5: "f", 6: "n"}
        gc = _make_json_constraint(vocab)
        logits = torch.ones(7)
        result = gc.mask_logits(logits, JSONState.VALUE)
        for i in range(7):
            assert result[i].item() != float("-inf"), f"token {i} ({vocab[i]}) should be allowed"

    def test_object_start_allows_key_or_close(self):
        vocab = {0: '"', 1: "}", 2: "x", 3: "{"}
        gc = _make_json_constraint(vocab)
        logits = torch.zeros(4)
        result = gc.mask_logits(logits, JSONState.OBJECT_START)
        assert result[0].item() != float("-inf")  # "
        assert result[1].item() != float("-inf")  # }
        assert result[2].item() == float("-inf")  # x
        assert result[3].item() == float("-inf")  # {

    def test_colon_state_allows_colon(self):
        vocab = {0: ":", 1: '"', 2: "x"}
        gc = _make_json_constraint(vocab)
        logits = torch.zeros(3)
        result = gc.mask_logits(logits, JSONState.COLON)
        assert result[0].item() != float("-inf")  # :
        assert result[1].item() == float("-inf")  # "
        assert result[2].item() == float("-inf")  # x

    def test_comma_or_end_allows_comma_and_close(self):
        vocab = {0: ",", 1: "}", 2: "]", 3: "x"}
        gc = _make_json_constraint(vocab)
        logits = torch.zeros(4)
        result = gc.mask_logits(logits, JSONState.COMMA_OR_END)
        assert result[0].item() != float("-inf")
        assert result[1].item() != float("-inf")
        assert result[2].item() != float("-inf")
        assert result[3].item() == float("-inf")

    def test_whitespace_always_allowed(self):
        vocab = {0: " ", 1: "\n", 2: "\t"}
        gc = _make_json_constraint(vocab)
        for state in [JSONState.START, JSONState.COLON, JSONState.COMMA_OR_END]:
            logits = torch.zeros(3)
            result = gc.mask_logits(logits, state)
            for i in range(3):
                assert result[i].item() != float("-inf")


class TestJSONSafetyFallback:
    def test_never_masks_all_tokens(self):
        vocab = {0: "x", 1: "y", 2: "z"}
        gc = _make_json_constraint(vocab)
        logits = torch.zeros(3)
        result = gc.mask_logits(logits, JSONState.COLON)
        assert not torch.all(result == float("-inf")), "must never mask all tokens"

    def test_fallback_returns_original_logits(self):
        vocab = {0: "x", 1: "y"}
        gc = _make_json_constraint(vocab)
        logits = torch.tensor([1.0, 2.0])
        result = gc.mask_logits(logits, JSONState.COLON)
        assert torch.equal(result, logits)


class TestJSONStateTransitions:
    def test_start_to_object(self):
        gc = _make_json_constraint({})
        assert gc.next_state(JSONState.START, "{") == JSONState.OBJECT_START

    def test_start_to_array(self):
        gc = _make_json_constraint({})
        assert gc.next_state(JSONState.START, "[") == JSONState.ARRAY_START

    def test_object_start_to_key(self):
        gc = _make_json_constraint({})
        assert gc.next_state(JSONState.OBJECT_START, '"k') == JSONState.KEY

    def test_key_to_colon(self):
        gc = _make_json_constraint({})
        assert gc.next_state(JSONState.KEY, '"') == JSONState.COLON

    def test_colon_to_value(self):
        gc = _make_json_constraint({})
        assert gc.next_state(JSONState.COLON, ":") == JSONState.VALUE

    def test_value_string_complete(self):
        gc = _make_json_constraint({})
        assert gc.next_state(JSONState.VALUE, '"hello"') == JSONState.COMMA_OR_END

    def test_value_number(self):
        gc = _make_json_constraint({})
        result = gc.next_state(JSONState.VALUE, "42")
        assert result == JSONState.COMMA_OR_END

    def test_value_true(self):
        gc = _make_json_constraint({})
        assert gc.next_state(JSONState.VALUE, "true") == JSONState.COMMA_OR_END

    def test_empty_object(self):
        gc = _make_json_constraint({})
        assert gc.next_state(JSONState.OBJECT_START, "}") == JSONState.COMMA_OR_END


# --- CodeGrammarConstraint tests ---


class TestCodeGrammarBracketBalancing:
    def test_allows_opening_brackets(self):
        vocab = {0: "{", 1: "[", 2: "(", 3: "x"}
        gc = _make_code_constraint(vocab)
        state = CodeBracketState()
        logits = torch.zeros(4)
        result = gc.mask_logits(logits, state)
        for i in range(4):
            assert result[i].item() != float("-inf")

    def test_masks_closing_brace_at_zero_depth(self):
        vocab = {0: "}", 1: "x", 2: "{"}
        gc = _make_code_constraint(vocab)
        state = CodeBracketState()
        logits = torch.zeros(3)
        result = gc.mask_logits(logits, state)
        assert result[0].item() == float("-inf")
        assert result[1].item() != float("-inf")
        assert result[2].item() != float("-inf")

    def test_allows_closing_brace_at_positive_depth(self):
        vocab = {0: "}", 1: "x"}
        gc = _make_code_constraint(vocab)
        state = CodeBracketState()
        state.brace_depth = 1
        logits = torch.zeros(2)
        result = gc.mask_logits(logits, state)
        assert result[0].item() != float("-inf")

    def test_masks_closing_bracket_at_zero_depth(self):
        vocab = {0: "]", 1: "x"}
        gc = _make_code_constraint(vocab)
        state = CodeBracketState()
        logits = torch.zeros(2)
        result = gc.mask_logits(logits, state)
        assert result[0].item() == float("-inf")

    def test_masks_closing_paren_at_zero_depth(self):
        vocab = {0: ")", 1: "x"}
        gc = _make_code_constraint(vocab)
        state = CodeBracketState()
        logits = torch.zeros(2)
        result = gc.mask_logits(logits, state)
        assert result[0].item() == float("-inf")


class TestCodeGrammarStringTracking:
    def test_in_string_allows_everything(self):
        vocab = {0: "}", 1: "]", 2: ")", 3: "x"}
        gc = _make_code_constraint(vocab)
        state = CodeBracketState()
        state.in_string = True
        state.string_char = '"'
        logits = torch.zeros(4)
        result = gc.mask_logits(logits, state)
        for i in range(4):
            assert result[i].item() != float("-inf")

    def test_string_entry_and_exit(self):
        gc = _make_code_constraint({})
        state = CodeBracketState()
        s2 = gc.next_state(state, '"')
        assert s2.in_string is True
        assert s2.string_char == '"'
        s3 = gc.next_state(s2, 'hello')
        assert s3.in_string is True
        s4 = gc.next_state(s3, '"')
        assert s4.in_string is False

    def test_escape_in_string(self):
        gc = _make_code_constraint({})
        state = CodeBracketState()
        state.in_string = True
        state.string_char = '"'
        s2 = gc.next_state(state, '\\')
        assert s2.escape_next is True
        s3 = gc.next_state(s2, '"')
        assert s3.in_string is True  # escaped quote doesn't close string


class TestCodeGrammarStateTransitions:
    def test_nested_braces(self):
        gc = _make_code_constraint({})
        s = CodeBracketState()
        s = gc.next_state(s, '{')
        assert s.brace_depth == 1
        s = gc.next_state(s, '{')
        assert s.brace_depth == 2
        s = gc.next_state(s, '}')
        assert s.brace_depth == 1
        s = gc.next_state(s, '}')
        assert s.brace_depth == 0

    def test_mixed_brackets(self):
        gc = _make_code_constraint({})
        s = CodeBracketState()
        s = gc.next_state(s, '{[( ')
        assert s.brace_depth == 1
        assert s.bracket_depth == 1
        assert s.paren_depth == 1

    def test_close_at_zero_clamps(self):
        gc = _make_code_constraint({})
        s = CodeBracketState()
        s = gc.next_state(s, '}')
        assert s.brace_depth == 0


class TestCodeGrammarSafetyFallback:
    def test_never_masks_all_tokens(self):
        vocab = {0: "}", 1: "]", 2: ")"}
        gc = _make_code_constraint(vocab)
        state = CodeBracketState()
        logits = torch.zeros(3)
        result = gc.mask_logits(logits, state)
        assert not torch.all(result == float("-inf"))
