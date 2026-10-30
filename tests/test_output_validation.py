"""Unit test: schema đầu ra của LLM, kiểm tra số liệu/trích dẫn/tham chiếu, áp dụng diff.
Các kiểm tra này là heuristic phát hiện tham chiếu KHÔNG có trong nguồn; đạt kiểm tra không chứng minh đúng về pháp lý."""
from __future__ import annotations

import pytest

from src.services.rag_engine import answer_checks, apply_edits, reference_checks, validate_llm_output


def src(i, doc, content, article=None, page=None, status="unverified", amend=False):
    return {"index": i, "doc_number": doc, "content": content, "article": article, "page": page, "status": status,
            "is_amendment": amend, "amended_by": [], "replaced_by": [], "repealed_by": []}


SOURCES = [
    src(1, "101/2030/NĐ-CP", "Điều 5. Hạn mức mua sắm trực tiếp là 5 tỷ đồng cho mỗi gói.", "Điều 5", 3),
    src(2, "202/2031/NĐ-CP", "Sửa đổi Điều 5: hạn mức là 1,5 tỷ đồng. Hiệu lực từ ngày 01 tháng 7 năm 2031.", "sửa đổi Điều 5", 2, amend=True),
]


# ------------------------------ schema ------------------------------
def test_schema_valid_with_extra_fields_ignored():
    out, errs = validate_llm_output('{"answer": "Có [1]", "used_sources": [1], "enough_context": true, "x": 1}')
    assert out and out.answer == "Có [1]" and out.used_sources == [1] and errs == []


def test_schema_accepts_json_inside_code_fence():
    out, _ = validate_llm_output('```json\n{"answer": "ok", "used_sources": [2]}\n```')
    assert out and out.used_sources == [2] and out.enough_context is True


def test_schema_rejects_non_json_and_wrong_types():
    assert validate_llm_output("Đây là câu trả lời thường")[0] is None
    out, errs = validate_llm_output('{"answer": "", "used_sources": "1"}')
    assert out is None and errs
    out, errs = validate_llm_output('{"used_sources": [1]}')
    assert out is None and any("answer" in e for e in errs)


# ------------------------------ số liệu & trích dẫn ------------------------------
def test_numbers_in_sources_are_grounded():
    c = answer_checks("Hạn mức là 1,5 tỷ đồng [2], trước đây là 5 tỷ đồng [1].", "Hạn mức?", SOURCES)
    assert c["ungrounded_numbers"] == [] and c["invalid_citations"] == []


def test_invented_number_is_flagged_but_question_numbers_are_allowed():
    c = answer_checks("Hạn mức là 9 tỷ đồng [1] trong 45 ngày.", "Điều kiện trong 45 ngày?", SOURCES)
    assert any("9" in x for x in c["ungrounded_numbers"]) and not any("45" in x for x in c["ungrounded_numbers"])


def test_value_with_unit_must_match_the_same_unit_not_just_the_digits():
    # '10' và '45' xuất hiện trong nguồn nhưng KHÔNG đi với đơn vị 'ngày' => giá trị này không có căn cứ
    srcs = [src(1, "1/2020/NĐ-CP", "Điều 10. Thời hạn 15 ngày. Phạt 45 triệu đồng. Có 30 tháng bảo hành.")]
    c = answer_checks("Thời hạn là 10 ngày, phạt 45 ngày và bảo hành 30 tháng [1].", "?", srcs)
    assert "10 ngày" in c["ungrounded_numbers"] and "45 ngày" in c["ungrounded_numbers"] and not any("30" in x for x in c["ungrounded_numbers"])
    assert answer_checks("Thời hạn là 15 ngày, phạt 45 triệu đồng [1].", "?", srcs)["ungrounded_numbers"] == []


def test_thousand_separators_are_normalized():
    srcs = [src(1, "1/2020/NĐ-CP", "Mức phạt 1.000.000 đồng")]
    assert answer_checks("Phạt 1,000,000 đồng [1]", "?", srcs)["ungrounded_numbers"] == []


def test_invalid_citation_marks_detected():
    c = answer_checks("Theo [1] và [7], hạn mức nêu trên.", "?", SOURCES)
    assert c["invalid_citations"] == [7]


def test_uncited_long_sentences_counted():
    c = answer_checks("Hạn mức mua sắm trực tiếp thiết bị y tế được quy định trong nghị định mới nhất của chính phủ.", "?", SOURCES)
    assert c["uncited_sentences"] == 1 and c["says_no_basis"] is False


@pytest.mark.parametrize("text", [
    "Chưa đủ căn cứ trong các đoạn trích đã cung cấp để trả lời.", "Không tìm thấy căn cứ trong tài liệu.", "Hiện chưa có đủ thông tin về mức phạt.",
    "Không đủ căn cứ để kết luận.",
])
def test_no_basis_phrase_detected(text):
    assert answer_checks(text, "?", SOURCES)["says_no_basis"] is True


def test_normal_answer_is_not_treated_as_refusal():
    assert answer_checks("Hạn mức là 1,5 tỷ đồng theo Điều 5 [2].", "?", SOURCES)["says_no_basis"] is False


# ------------------------------ tham chiếu pháp lý ------------------------------
def test_supported_references_pass():
    r = reference_checks("Theo Điều 5 của Nghị định 101/2030/NĐ-CP (trang 3) và Nghị định 202/2031/NĐ-CP [2].", "?", SOURCES)
    assert r == {"unsupported_documents": [], "unsupported_articles": [], "unsupported_pages": [], "unsupported_status_claims": []}


def test_fabricated_document_article_page_are_flagged():
    r = reference_checks("Theo Điều 99 Nghị định 999/2099/NĐ-CP (trang 50) thì mức là cố định.", "?", SOURCES)
    assert r["unsupported_documents"] == ["999/2099/NĐ-CP"] and r["unsupported_articles"] == ["99"] and r["unsupported_pages"] == [50]


def test_document_number_with_ocr_garbled_suffix_still_matches_by_number_and_year():
    srcs = [src(1, "101/2030/NB-CP", "nội dung")]
    assert reference_checks("Nghị định 101/2030/NĐ-CP quy định", "?", srcs)["unsupported_documents"] == []


def test_validity_claims_require_support():
    r = reference_checks("Nghị định 101/2030/NĐ-CP vẫn còn hiệu lực.", "?", SOURCES)
    assert r["unsupported_status_claims"]                                    # hệ thống chưa xác minh hiệu lực => không được khẳng định
    r = reference_checks("Nghị định 101/2030/NĐ-CP đã bị bãi bỏ.", "?", SOURCES)
    assert r["unsupported_status_claims"]
    repealed = [src(1, "101/2030/NĐ-CP", "x", status="repealed")]
    assert reference_checks("Nghị định 101/2030/NĐ-CP đã bị bãi bỏ.", "?", repealed)["unsupported_status_claims"] == []


def test_question_provided_references_are_not_flagged():
    assert reference_checks("Điều 18 không có trong nguồn.", "Cho tôi hỏi Điều 18?", SOURCES)["unsupported_articles"] == []


# ------------------------------ diff ------------------------------
def test_apply_edits_replace_append_and_failures():
    new, ok, bad = apply_edits("Hạn mức là 5 tỷ.", [
        {"op": "replace", "find": "5 tỷ", "replace": "1,5 tỷ"}, {"op": "append", "text": "Lưu ý."},
        {"op": "replace", "find": "không có", "replace": "x"}, {"op": "delete"},
    ])
    assert new.startswith("Hạn mức là 1,5 tỷ.") and new.endswith("Lưu ý.") and len(ok) == 2 and len(bad) == 2
