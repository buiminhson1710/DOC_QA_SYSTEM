"""Đánh giá RAG trong tiến trình (kho TỔNG HỢP + embedding GIẢ LẬP, không cần khoá API).
Đo cơ chế pipeline (hybrid, mở rộng văn bản sửa đổi, ghim điều khoản) — KHÔNG phải chất lượng embedding/LLM thật."""
from __future__ import annotations

import asyncio

import pytest

from src.evaluation import CONFIGS, CORPUS, GOLDEN, format_markdown, run_inprocess
from src.services.direct_answer import fold


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    from conftest import reset_global_state

    rep = asyncio.run(run_inprocess(workdir=tmp_path_factory.mktemp("eval")))
    reset_global_state()
    return rep


def test_golden_evidence_actually_exists_in_the_corpus():
    """Kiểm tra CHÍNH bộ đáp án chuẩn: mọi bằng chứng (và giá trị cũ) phải có nguyên văn trong kho."""
    corpus = fold(" ".join(" ".join(builder()) for builder in CORPUS.values()))
    for q in GOLDEN:
        for ev in q.get("evidence", []) + q.get("stale", []):
            assert fold(ev) in corpus, f"{q['id']}: bằng chứng không có trong kho: {ev}"
        if q.get("answerable", True):
            assert q["evidence"], q["id"]
        else:
            assert not q["evidence"]


def test_golden_covers_required_question_types():
    cats = {q["category"] for q in GOLDEN}
    assert {"lookup", "amended", "multi_doc", "relation", "ocr_noisy", "paraphrase", "no_data"} <= cats
    assert sum(1 for q in GOLDEN if q.get("answerable", True)) >= 20


def test_report_structure_and_honest_scope(report):
    assert set(report["configs"]) == set(CONFIGS) and report["n_questions"] == len(GOLDEN)
    assert all(c["ingest"] in ("complete", "complete_with_warnings") for c in report["corpus"])
    assert "độ đúng câu trả lời" in report["not_measured_here"] and "GIẢ LẬP" in report["note"]
    md = format_markdown(report)
    assert "baseline_prev" in md and "R@3" in md


def test_hybrid_does_not_hurt_retrieval_vs_vector_only(report):
    v, h = report["configs"]["vector_only"]["overall"], report["configs"]["hybrid"]["overall"]
    assert h["recall@3"] >= v["recall@3"] and h["mrr"] >= v["mrr"]


KNOWN_HARD = {"M6"}   # câu cần nối gián tiếp Điều 9 (cơ sở) với mục "bãi bỏ khoản 3 Điều 9" trong văn bản sửa đổi: giới hạn đã biết


def _amended_rows(report, name, exclude=()):
    rows = [r for r in report["configs"][name]["per_question"] if r["category"] == "amended" and r["id"] not in exclude]
    return rows


def test_amendment_expansion_removes_context_with_only_the_old_value(report):
    """Nhóm 'amended': nếu không mở rộng văn bản sửa đổi, có câu chỉ nhận được giá trị CŨ (sai nghiêm trọng)."""
    assert any(r.get("stale_without_current") for r in _amended_rows(report, "vector_only"))
    for name in ("baseline_prev", "hybrid", "full"):
        rows = _amended_rows(report, name)
        assert not any(r.get("stale_without_current") for r in rows)
        assert all(r["context_hit"] for r in _amended_rows(report, name, exclude=KNOWN_HARD)), name


@pytest.mark.xfail(strict=True, reason="GIỚI HẠN ĐÃ BIẾT (M6): bằng chứng nằm ở mục 'bãi bỏ khoản 3 Điều 9' của văn bản sửa đổi, cần nối gián tiếp từ Điều 9; "
                                       "chunk Điều 9 không vào top truy xuất. Nếu test này bắt đầu đạt, hãy bỏ khỏi KNOWN_HARD.")
def test_known_hard_partial_repeal_question_reaches_its_evidence(report):
    assert all(r["context_hit"] for r in _amended_rows(report, "full") if r["id"] in KNOWN_HARD)


def test_current_value_comes_from_a_source_marked_as_amendment(report):
    full = report["configs"]["full"]["by_category"]["amended"]
    assert full["current_marked_amendment"] is not None and full["current_marked_amendment"] >= 0.8


def test_no_data_questions_are_tracked_separately_from_recall(report):
    full = report["configs"]["full"]
    assert len(full["unanswerable"]) == sum(1 for q in GOLDEN if not q.get("answerable", True))
    assert all(r["category"] != "no_data" for r in full["per_question"])


def test_cost_and_latency_are_reported(report):
    for r in report["configs"].values():
        assert r["est_context_tokens"]["mean"] > 0 and r["latency_ms"]["p50"] >= 0
