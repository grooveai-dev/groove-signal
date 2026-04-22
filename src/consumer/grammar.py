"""Grammar-constrained token masking for speculative decoding.

Filters draft candidates to only syntactically valid continuations,
eliminating wasted network round-trips for structured output (JSON, code).
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

import torch

logger = logging.getLogger(__name__)


class GrammarConstraint(ABC):
    @abstractmethod
    def mask_logits(self, logits: torch.Tensor, state: Any) -> torch.Tensor:
        ...

    @abstractmethod
    def initial_state(self) -> Any:
        ...

    @abstractmethod
    def next_state(self, state: Any, token_str: str) -> Any:
        ...


class JSONState(Enum):
    START = auto()
    OBJECT_START = auto()
    KEY = auto()
    COLON = auto()
    VALUE = auto()
    STRING = auto()
    NUMBER = auto()
    COMMA_OR_END = auto()
    ARRAY_START = auto()
    ARRAY_VALUE = auto()
    ARRAY_COMMA_OR_END = auto()
    DONE = auto()


_JSON_VALUE_STARTERS = frozenset('"0123456789-{[tfn')
_DIGIT_CHARS = frozenset('0123456789.-+eE')
_WHITESPACE = frozenset(' \t\n\r')


class JSONGrammarConstraint(GrammarConstraint):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._vocab_size = tokenizer.vocab_size
        self._token_first_chars: dict[int, str] = {}
        self._build_token_map()

    def _build_token_map(self):
        for token_id in range(self._vocab_size):
            try:
                decoded = self._tokenizer.decode([token_id])
                stripped = decoded.lstrip()
                if stripped:
                    self._token_first_chars[token_id] = stripped[0]
            except Exception:
                pass

    def initial_state(self) -> JSONState:
        return JSONState.START

    def next_state(self, state: JSONState, token_str: str) -> JSONState:
        s = token_str.strip()
        if not s:
            return state

        c = s[0]

        if state in (JSONState.START, JSONState.VALUE, JSONState.ARRAY_VALUE):
            if c == '{':
                return JSONState.OBJECT_START
            elif c == '[':
                return JSONState.ARRAY_START
            elif c == '"':
                if s.count('"') >= 2:
                    if state == JSONState.ARRAY_VALUE:
                        return JSONState.ARRAY_COMMA_OR_END
                    return JSONState.COMMA_OR_END
                return JSONState.STRING
            elif c in '0123456789-':
                all_num = all(ch in _DIGIT_CHARS for ch in s)
                if all_num:
                    if state == JSONState.ARRAY_VALUE:
                        return JSONState.ARRAY_COMMA_OR_END
                    return JSONState.COMMA_OR_END
                return JSONState.NUMBER
            elif s.startswith('true') or s.startswith('false') or s.startswith('null'):
                if state == JSONState.ARRAY_VALUE:
                    return JSONState.ARRAY_COMMA_OR_END
                return JSONState.COMMA_OR_END

        elif state == JSONState.OBJECT_START:
            if c == '"':
                if s.count('"') >= 2:
                    return JSONState.COLON
                return JSONState.KEY
            elif c == '}':
                return JSONState.COMMA_OR_END

        elif state == JSONState.KEY:
            if c == '"':
                return JSONState.COLON
            return JSONState.KEY

        elif state == JSONState.COLON:
            if c == ':':
                return JSONState.VALUE

        elif state == JSONState.STRING:
            if '"' in s:
                return JSONState.COMMA_OR_END
            return JSONState.STRING

        elif state == JSONState.NUMBER:
            if all(ch in _DIGIT_CHARS for ch in s):
                return JSONState.NUMBER
            return JSONState.COMMA_OR_END

        elif state == JSONState.COMMA_OR_END:
            if c == ',':
                return JSONState.OBJECT_START
            elif c == '}':
                return JSONState.COMMA_OR_END
            elif c == ']':
                return JSONState.COMMA_OR_END

        elif state == JSONState.ARRAY_START:
            if c == ']':
                return JSONState.COMMA_OR_END
            return self.next_state(JSONState.ARRAY_VALUE, token_str)

        elif state == JSONState.ARRAY_COMMA_OR_END:
            if c == ',':
                return JSONState.ARRAY_VALUE
            elif c == ']':
                return JSONState.COMMA_OR_END

        return state

    def _allowed_first_chars(self, state: JSONState) -> frozenset[str]:
        if state in (JSONState.START, JSONState.VALUE, JSONState.ARRAY_VALUE):
            return _JSON_VALUE_STARTERS | _WHITESPACE
        elif state == JSONState.OBJECT_START:
            return frozenset('"}') | _WHITESPACE
        elif state == JSONState.KEY:
            return frozenset()  # any char allowed in string
        elif state == JSONState.COLON:
            return frozenset(':') | _WHITESPACE
        elif state == JSONState.STRING:
            return frozenset()  # any char allowed in string
        elif state == JSONState.NUMBER:
            return _DIGIT_CHARS | frozenset(',}]') | _WHITESPACE
        elif state == JSONState.COMMA_OR_END:
            return frozenset(',}]') | _WHITESPACE
        elif state == JSONState.ARRAY_START:
            return _JSON_VALUE_STARTERS | frozenset(']') | _WHITESPACE
        elif state == JSONState.ARRAY_COMMA_OR_END:
            return frozenset(',]') | _WHITESPACE
        elif state == JSONState.DONE:
            return _WHITESPACE
        return frozenset()

    def mask_logits(self, logits: torch.Tensor, state: JSONState) -> torch.Tensor:
        allowed_chars = self._allowed_first_chars(state)
        if not allowed_chars:
            return logits

        mask = torch.full_like(logits, float('-inf'))

        for token_id, first_char in self._token_first_chars.items():
            if token_id < logits.shape[-1] and first_char in allowed_chars:
                mask[..., token_id] = 0.0

        result = logits + mask

        if torch.all(result == float('-inf')):
            logger.warning(
                "JSON grammar would mask all tokens at state %s, falling back to unconstrained",
                state,
            )
            return logits

        return result


class CodeBracketState:
    __slots__ = ('brace_depth', 'bracket_depth', 'paren_depth', 'in_string', 'string_char', 'escape_next')

    def __init__(self):
        self.brace_depth: int = 0
        self.bracket_depth: int = 0
        self.paren_depth: int = 0
        self.in_string: bool = False
        self.string_char: str = ''
        self.escape_next: bool = False

    def copy(self) -> 'CodeBracketState':
        s = CodeBracketState()
        s.brace_depth = self.brace_depth
        s.bracket_depth = self.bracket_depth
        s.paren_depth = self.paren_depth
        s.in_string = self.in_string
        s.string_char = self.string_char
        s.escape_next = self.escape_next
        return s


class CodeGrammarConstraint(GrammarConstraint):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._vocab_size = tokenizer.vocab_size
        self._token_strings: dict[int, str] = {}
        self._build_token_map()

    def _build_token_map(self):
        for token_id in range(self._vocab_size):
            try:
                decoded = self._tokenizer.decode([token_id])
                if decoded:
                    self._token_strings[token_id] = decoded
            except Exception:
                pass

    def initial_state(self) -> CodeBracketState:
        return CodeBracketState()

    def next_state(self, state: CodeBracketState, token_str: str) -> CodeBracketState:
        s = state.copy()
        for ch in token_str:
            if s.escape_next:
                s.escape_next = False
                continue
            if s.in_string:
                if ch == '\\':
                    s.escape_next = True
                elif ch == s.string_char:
                    s.in_string = False
                    s.string_char = ''
                continue
            if ch in ('"', "'", '`'):
                s.in_string = True
                s.string_char = ch
            elif ch == '{':
                s.brace_depth += 1
            elif ch == '}':
                s.brace_depth = max(0, s.brace_depth - 1)
            elif ch == '[':
                s.bracket_depth += 1
            elif ch == ']':
                s.bracket_depth = max(0, s.bracket_depth - 1)
            elif ch == '(':
                s.paren_depth += 1
            elif ch == ')':
                s.paren_depth = max(0, s.paren_depth - 1)
        return s

    def _would_create_invalid_nesting(self, state: CodeBracketState, token_str: str) -> bool:
        s = state.copy()
        for ch in token_str:
            if s.escape_next:
                s.escape_next = False
                continue
            if s.in_string:
                if ch == '\\':
                    s.escape_next = True
                elif ch == s.string_char:
                    s.in_string = False
                    s.string_char = ''
                continue
            if ch in ('"', "'", '`'):
                s.in_string = True
                s.string_char = ch
                continue
            if ch == '}' and s.brace_depth == 0:
                return True
            if ch == ']' and s.bracket_depth == 0:
                return True
            if ch == ')' and s.paren_depth == 0:
                return True
            if ch == '{':
                s.brace_depth += 1
            elif ch == '}':
                s.brace_depth -= 1
            elif ch == '[':
                s.bracket_depth += 1
            elif ch == ']':
                s.bracket_depth -= 1
            elif ch == '(':
                s.paren_depth += 1
            elif ch == ')':
                s.paren_depth -= 1
        return False

    def mask_logits(self, logits: torch.Tensor, state: CodeBracketState) -> torch.Tensor:
        if state.in_string:
            return logits

        mask = torch.zeros_like(logits)
        masked_any = False

        for token_id, token_str in self._token_strings.items():
            if token_id < logits.shape[-1] and self._would_create_invalid_nesting(state, token_str):
                mask[..., token_id] = float('-inf')
                masked_any = True

        if not masked_any:
            return logits

        result = logits + mask

        if torch.all(result == float('-inf')):
            logger.warning(
                "Code grammar would mask all tokens, falling back to unconstrained",
            )
            return logits

        return result
