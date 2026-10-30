"""Unit test BM25 / RRF."""
from __future__ import annotations

from src.services.lexical import BM25, rrf, tokenize

DOCS = [
    ("a", "Điều 18. Chỉ định thầu. Hạn mức chỉ định thầu là 5 tỷ đồng"),
    ("b", "Điều 19. Mua sắm trực tiếp theo quy định của Nghị định này"),
    ("c", "Bãi bỏ các Nghị định số 53/2013/NĐ-CP và 34/2015/NĐ-CP"),
    ("d", "Thời hạn bảo hành tối thiểu là 24 tháng kể từ ngày nghiệm thu"),
]


def test_exact_terms_rank_first():
    bm = BM25(DOCS)
    assert bm.search("hạn mức chỉ định thầu là bao nhiêu", 3)[0][0] == "a"
    assert bm.search("nghị định 53/2013 bị bãi bỏ", 3)[0][0] == "c"


def test_diacritic_insensitive_matching_helps_noisy_ocr():
    bm = BM25(DOCS)
    assert bm.search("bao hanh toi thieu", 1)[0][0] == "d"       # truy vấn không dấu vẫn khớp văn bản có dấu


def test_stopwords_removed_and_no_match_returns_empty():
    assert "cua" not in tokenize("của và là")
    bm = BM25(DOCS)
    assert bm.search("zzz qqq", 3) == [] and BM25([]).search("a") == []


def test_scores_are_sorted_descending_and_limited():
    res = BM25(DOCS).search("nghị định", 2)
    assert len(res) <= 2 and [s for _, s in res] == sorted((s for _, s in res), reverse=True)


def test_rrf_rewards_agreement_between_branches():
    fused = rrf([["a", "b", "c"], ["b", "d", "a"]])
    order = sorted(fused, key=lambda k: -fused[k])
    assert order[0] == "b" and order[1] == "a" and set(order) == {"a", "b", "c", "d"}


def test_rrf_single_ranking_keeps_order():
    fused = rrf([["x", "y", "z"]])
    assert sorted(fused, key=lambda k: -fused[k]) == ["x", "y", "z"]
