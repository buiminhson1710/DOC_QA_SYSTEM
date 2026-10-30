"""Unit test: bóc tách, số hiệu, quan hệ pháp lý, phát hiện nội dung thiếu, OCR dở dang và tiếp tục."""
from __future__ import annotations

import pytest

from helpers import make_ascii_pdf, make_docx
from src.services import document_parser as DP
from src.services import llm_router


def rels(text: str, own: str = "999/2099/NĐ-CP") -> dict[tuple[str, str], str]:
    return {(r.type, r.target): r.scope for r in DP.extract_relations(text, own)}


def test_normalize_doc_number():
    assert DP.normalize_doc_number("214 / 2025 / ND-CP") == "214/2025/NĐ-CP"
    assert DP.normalize_doc_number("022/2023/QH15") == "22/2023/QH15"


def test_fix_number_with_filename_when_ocr_corrupts_suffix():
    assert DP._fix_with_filename("349/2026/NB-CP", "Nghi dinh 349_2026_ND-CP pdf.pdf") == "349/2026/NĐ-CP"
    # khác năm/số => không ghi đè bằng tên file
    assert DP._fix_with_filename("349/2027/NB-CP", "Nghi dinh 349_2026_ND-CP pdf.pdf") == "349/2027/NB-CP"


def test_relations_distinguish_amend_replace_repeal_and_partial_repeal():
    text = (
        "Sửa đổi, bổ sung một số điều của Nghị định số 214/2025/NĐ-CP ngày 04 tháng 8 năm 2025. "
        "Điều 5. Nghị định này thay thế Nghị định số 55/2019/NĐ-CP. "
        "Kể từ ngày có hiệu lực, bãi bỏ các Nghị định sau: a) Nghị định số 53/2013/NĐ-CP; b) Nghị định số 34/2015/NĐ-CP. "
        "Điều 6. Bãi bỏ khoản 3 Điều 9 của Nghị định số 101/2030/NĐ-CP."
    )
    r = rels(text)
    assert r[("amends", "214/2025/NĐ-CP")] == "partial"
    assert r[("replaces", "55/2019/NĐ-CP")] == "full"
    assert r[("repeals", "53/2013/NĐ-CP")] == "full"
    assert r[("repeals", "34/2015/NĐ-CP")] == "full"
    assert r[("amends", "101/2030/NĐ-CP")] == "partial_repeal"      # bãi bỏ MỘT khoản != bãi bỏ cả văn bản
    assert ("repeals", "101/2030/NĐ-CP") not in r
    assert ("replaces", "214/2025/NĐ-CP") not in r and ("repeals", "214/2025/NĐ-CP") not in r


def test_relations_ignore_legal_basis_citation_and_self_reference():
    text = "Căn cứ Luật Đấu thầu số 22/2023/QH15 đã được sửa đổi, bổ sung bởi Luật số 57/2024/QH15; Nghị định số 999/2099/NĐ-CP."
    assert rels(text) == {}


def test_thay_the_cum_tu_is_not_a_replacement():
    assert rels("thay thế cụm từ 'ngày' bằng cụm từ 'ngày làm việc' tại Điều 5 của Nghị định số 214/2025/NĐ-CP") == {}


def test_article_coverage_reports_missing_numbers():
    text = "Điều 1. A\nnội dung\nĐiều 2. B\nĐiều 3. C\nĐiều 5. E\nĐiều 6. F"
    cov = DP.article_coverage(text)
    assert cov["found"] == 5 and cov["max"] == 6 and cov["missing"] == [4]
    assert DP.article_coverage("không có điều nào")["found"] == 0


def test_garble_ratio_separates_clean_and_noisy_text():
    clean = "Điều 5. Hạn mức mua sắm trực tiếp là 5 tỷ đồng theo Nghị định 214/2025/NĐ-CP"
    noisy = "Đ1ều 5. H4n m|c mu4 s4m tr~c t1ếp l4 5 tỷ đ0ng the0 Ngh1 đ1nh 2l4/2O25/NĐ"
    assert DP.garble_ratio(clean) < 0.05
    assert DP.garble_ratio(noisy) > 0.25


def test_parse_docx_and_chunk_labels(cfg, tmp_path):
    path = make_docx(tmp_path / "NĐ 214-2025.docx", ["Số: 214/2025/NĐ-CP", "NGHỊ ĐỊNH", "Điều 1. Phạm vi", "Nội dung một.", "Điều 2. Đối tượng", "Nội dung hai."])
    parsed = DP.parse_file(path)
    assert parsed.doc_number == "214/2025/NĐ-CP" and parsed.doc_type == "Nghị định"
    chunks = DP.chunk_document(parsed, 1200, 150)
    assert chunks and {c.article for c in chunks} >= {"Điều 1"}
    assert parsed.diagnostics["articles"]["missing"] == []


def test_empty_pdf_is_rejected(cfg, tmp_path):
    import pymupdf

    d = pymupdf.open()
    d.new_page()
    d.save(str(tmp_path / "blank.pdf"))
    with pytest.raises(DP.ParseError):
        DP.parse_file(tmp_path / "blank.pdf")


def _scan_doc(tmp_path):
    long_text = "Page one text. " * 12  # >= 80 ký tự => trang có chữ, không OCR
    return make_ascii_pdf(tmp_path / "mixed.pdf", [long_text, None, None, long_text, None])


def test_only_scanned_pages_are_ocred_and_incomplete_ocr_is_detected(cfg, tmp_path, monkeypatch):
    requested: list[list[int]] = []

    def partial(images):
        requested.append(sorted(images))
        return {n: f"Điều {n}. Nội dung OCR trang {n} " + "x" * 90 for n in sorted(images) if n != 5}  # thiếu trang 5

    monkeypatch.setattr(DP, "_ocr_llm_batch", partial)
    with pytest.raises(DP.OCRIncomplete) as exc:
        DP.parse_file(_scan_doc(tmp_path))
    assert requested == [[2, 3, 5]]                    # chỉ trang scan được OCR, trang chữ thì không
    assert "xong 2/3" in str(exc.value) and "[5]" in str(exc.value)


def test_ocr_resume_reuses_cache_and_only_requests_missing_pages(cfg, tmp_path, monkeypatch):
    path = _scan_doc(tmp_path)
    calls: list[list[int]] = []

    def first(images):
        calls.append(sorted(images))
        return {n: f"OCR trang {n} " + "y" * 90 for n in sorted(images) if n in (2, 3)}

    monkeypatch.setattr(DP, "_ocr_llm_batch", first)
    with pytest.raises(DP.OCRIncomplete):
        DP.parse_file(path)

    def second(images):
        calls.append(sorted(images))
        return {n: f"OCR trang {n} " + "y" * 90 for n in images}

    monkeypatch.setattr(DP, "_ocr_llm_batch", second)
    parsed = DP.parse_file(path)
    assert calls == [[2, 3, 5], [5]]                    # lần 2 chỉ xin trang 5
    assert [p.page for p in parsed.pages] == [1, 2, 3, 4, 5]
    assert parsed.total_pages == 5 and parsed.diagnostics["failed_pages"] == []


def test_ocr_abort_when_all_models_exhausted_reports_reason(cfg, tmp_path, monkeypatch):
    def boom(images):
        raise llm_router.AllModelsUnavailable("quota_daily: hết 20 lượt/ngày", {})

    monkeypatch.setattr(DP, "_ocr_llm_batch", boom)
    with pytest.raises(DP.OCRIncomplete) as exc:
        DP.parse_file(_scan_doc(tmp_path))
    assert "quota_daily" in str(exc.value) and "xong 0/3" in str(exc.value)


def _vn_text_png() -> bytes:
    import io

    from PIL import Image, ImageDraw, ImageFont

    from helpers import find_font

    im = Image.new("L", (1240, 600), 255)
    ImageDraw.Draw(im).multiline_text((80, 80), "Điều 5. Hạn mức mua sắm\nHạn mức là 5 tỷ đồng cho mỗi gói", fill=0,
                                      font=ImageFont.truetype(find_font(), 36), spacing=16)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.skipif(not DP._tesseract_check()[0], reason="Tesseract hoặc dữ liệu ngôn ngữ 'vie' chưa cài (đặt TESSDATA_PREFIX nếu vie nằm ở thư mục riêng)")
def test_tesseract_reads_vietnamese_and_reports_confidence(cfg):
    from helpers import find_font

    if not find_font():
        pytest.skip("không có font Unicode để dựng ảnh thử")
    from src.services.direct_answer import fold

    text, conf = DP._ocr_tesseract_png(_vn_text_png())
    assert "dieu 5" in fold(text) and "5 ty dong" in fold(text)
    assert conf is not None and conf > 60
