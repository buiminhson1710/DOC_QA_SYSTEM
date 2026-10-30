"""Tìm kiếm từ khoá BM25 (không cần embedding) để bổ sung cho tìm kiếm vector.

Vì sao: embedding có thể bỏ sót (thiếu dữ liệu, model yếu, câu hỏi nêu đúng số điều/số hiệu/thuật ngữ); BM25 bắt tốt các
trường hợp khớp từ chính xác. Kết quả hai nhánh được trộn bằng Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from src.services.direct_answer import fold

_STOP = frozenset(
    "la va cua cac nhung co khong duoc trong theo nao the gi mot cho de khi nay do tai voi thi ma hay hoac se da dang "
    "nhu ve tu den boi bi nen can phai nguoi".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in fold(text).split() if t not in _STOP and (len(t) > 1 or t.isdigit())]


class BM25:
    def __init__(self, items: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids = [i for i, _ in items]
        self.tfs = [Counter(tokenize(text)) for _, text in items]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        self.df: Counter[str] = Counter()
        for tf in self.tfs:
            self.df.update(tf.keys())
        self.n = len(items)

    def search(self, query: str, k: int = 6) -> list[tuple[str, float]]:
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self.df]
        if not terms or not self.n:
            return []
        scored: list[tuple[str, float]] = []
        for idx, tf in enumerate(self.tfs):
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[idx] / self.avgdl))
            if score > 0:
                scored.append((self.ids[idx], score))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: điểm = tổng 1/(k + hạng) trên các danh sách xếp hạng."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused
