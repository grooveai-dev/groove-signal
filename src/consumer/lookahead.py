"""N-gram context matching for lookahead speculative decoding.

Finds repeating token patterns in existing context (common in code,
JSON, structured output) and proposes them as speculative candidates
with zero VRAM overhead.
"""

import logging

logger = logging.getLogger(__name__)


class LookaheadDecoder:
    def __init__(self, min_n: int = 3, max_n: int = 8):
        self.min_n = min_n
        self.max_n = max_n
        self._ngram_table: dict[tuple[int, ...], list[int]] = {}
        self.stats = {
            "match_rate": 0.0,
            "avg_match_length": 0.0,
            "longest_match": 0,
            "total_queries": 0,
            "total_matches": 0,
        }

    def update_context(self, token_ids: list[int]) -> None:
        self._ngram_table.clear()
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(token_ids) - n):
                key = tuple(token_ids[i : i + n - 1])
                continuation = token_ids[i + n - 1]
                if key not in self._ngram_table:
                    self._ngram_table[key] = []
                self._ngram_table[key].append(continuation)

    def propose_candidates(
        self, token_ids: list[int], n_candidates: int = 8
    ) -> list[int]:
        self.stats["total_queries"] += 1

        if not token_ids or not self._ngram_table:
            return []

        candidates: list[int] = []
        pos = len(token_ids)

        for look in range(min(self.max_n - 1, len(token_ids)), self.min_n - 2, -1):
            key = tuple(token_ids[pos - look : pos])
            continuations = self._ngram_table.get(key)
            if continuations:
                candidates.append(continuations[-1])
                break

        if not candidates:
            return []

        current = list(token_ids) + candidates
        while len(candidates) < n_candidates:
            found = False
            for look in range(
                min(self.max_n - 1, len(current)), self.min_n - 2, -1
            ):
                key = tuple(current[len(current) - look :])
                continuations = self._ngram_table.get(key)
                if continuations:
                    candidates.append(continuations[-1])
                    current.append(continuations[-1])
                    found = True
                    break
            if not found:
                break

        if candidates:
            self.stats["total_matches"] += 1
            match_len = len(candidates)
            if match_len > self.stats["longest_match"]:
                self.stats["longest_match"] = match_len
            total_q = self.stats["total_queries"]
            total_m = self.stats["total_matches"]
            self.stats["match_rate"] = total_m / total_q if total_q else 0.0
            prev_avg = self.stats["avg_match_length"]
            self.stats["avg_match_length"] = (
                prev_avg * (total_m - 1) + match_len
            ) / total_m

        return candidates
