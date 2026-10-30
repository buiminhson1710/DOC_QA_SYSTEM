"""Unit test nhận diện câu hỏi tra cứu trực tiếp và trích điều khoản."""
from __future__ import annotations

import pytest

from src.services.direct_answer import detect_intent, extract_amendment_blocks, extract_article, merge_chunks, page_at

RECORDS = [
    {"doc_number": "214/2025/NĐ-CP", "doc_type": "Nghị định", "relations": []},
    {"doc_number": "349/2026/NĐ-CP", "doc_type": "Nghị định", "relations": [{"type": "amends", "target": "214/2025/NĐ-CP"}]},
    {"doc_number": "359/2026/NĐ-CP", "doc_type": "Nghị định", "relations": [{"type": "repeals", "target": "53/2013/NĐ-CP"}, {"type": "repeals", "target": "34/2015/NĐ-CP"}]},
]


@pytest.mark.parametrize("q,kind", [
    ("Điều 18 Nghị định 214/2025/NĐ-CP quy định thế nào?", "article"), ("Cho tôi xem Điều 18 của Nghị định 214", "article"),
    ("Điều 18 Nghị định 214 được sửa đổi như thế nào?", "article_amendments"), ("Nghị định 53/2013/NĐ-CP còn hiệu lực không?", "status"),
    ("Nghị định 214 đã bị sửa đổi bởi văn bản nào?", "status"), ("Nghị định 359/2026/NĐ-CP bãi bỏ những nghị định nào?", "affects"),
    ("Hệ thống đang có những văn bản nào?", "list"),
])
def test_intents_answered_directly(q, kind):
    assert detect_intent(q, RECORDS).kind == kind


@pytest.mark.parametrize("q", [
    "Tóm tắt Điều 18 Nghị định 214", "Điều 18 Nghị định 214 có áp dụng cho nhà thầu nước ngoài không?", "Điều kiện để chỉ định thầu theo Điều 18 Nghị định 214 là gì?",
    "Nghị định 359/2026/NĐ-CP có hiệu lực từ ngày nào?", "So sánh Điều 18 và Điều 21 Nghị định 214", "Điều 18 quy định thế nào?",
    "Điều 18 Nghị định 214 và Nghị định 349 khác nhau thế nào", "Nghị định 349 sửa đổi những điều nào của Nghị định 214?", "Điều 99 Nghị định 777 quy định thế nào?",
])
def test_reasoning_or_ambiguous_questions_are_left_to_llm(q):
    assert detect_intent(q, RECORDS) is None


def test_article_extraction_skips_table_of_contents_and_merges_overlap():
    chunks = [{"content": "MỤC LỤC\nĐiều 18. Chỉ định thầu\nĐiều 19. Mua sắm", "page": 1}, {"content": "Điều 17. Rộng rãi\n1. Áp dụng rộng rãi.", "page": 2},
              {"content": "Điều 18. Chỉ định thầu\n1. Chỉ định thầu áp dụng trong các trường hợp sau đây: a) Gói thầu cấp bách;", "page": 3},
              {"content": "a) Gói thầu cấp bách; b) Gói thầu thuốc.\n2. Quy trình như sau", "page": 3}, {"content": "Điều 19. Mua sắm trực tiếp\n1. ...", "page": 4}]
    text, starts, pages = merge_chunks(chunks)
    a, b = extract_article(text, "18")
    body = text[a:b]
    assert body.startswith("Điều 18. Chỉ định thầu\n1. Chỉ định") and "b) Gói thầu thuốc" in body and "Điều 19" not in body
    assert body.count("a) Gói thầu cấp bách") == 1                    # phần chồng lấn đã được bỏ
    assert page_at(a, starts, pages) == 3


def test_amendment_blocks_find_items_touching_the_article():
    amend = ('Điều 1. Sửa đổi, bổ sung một số điều của Nghị định số 214/2025/NĐ-CP\n1. Sửa đổi, bổ sung Điều 18 như sau:\n"Điều 18. Chỉ định thầu\n1. Mức 1 tỷ đồng."\n'
             '2. Sửa đổi, bổ sung khoản 3 Điều 21 như sau: nội dung khác.\n3. Sửa đổi, bổ sung Điều 35, Điều 78 như sau: ...\nĐiều 2. Hiệu lực thi hành')
    assert "1 tỷ đồng" in extract_amendment_blocks(amend, "18")[0][1]
    assert extract_amendment_blocks(amend, "21") and extract_amendment_blocks(amend, "78")
    assert extract_amendment_blocks(amend, "5") == []
