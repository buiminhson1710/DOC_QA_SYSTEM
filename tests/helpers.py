"""Tiện ích dựng tài liệu thử."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf", "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def find_font() -> Optional[str]:
    return next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)


def make_text_pdf(path: Path, pages: list[str]) -> Path:
    """PDF có lớp chữ (cần font Unicode để có dấu tiếng Việt)."""
    import pymupdf

    font = find_font()
    assert font, "không có font Unicode"
    doc = pymupdf.open()
    for text in pages:
        pg = doc.new_page()
        pg.insert_font(fontname="uni", fontfile=font)
        pg.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text, fontname="uni", fontsize=9)
    doc.save(str(path))
    return path


def make_ascii_pdf(path: Path, pages: list[Optional[str]]) -> Path:
    """Mỗi phần tử: chuỗi ASCII (trang có chữ) hoặc None (trang scan = chỉ có ảnh, không có chữ)."""
    import pymupdf
    from PIL import Image

    doc = pymupdf.open()
    for text in pages:
        pg = doc.new_page()
        if text is None:
            buf = io.BytesIO()
            Image.new("RGB", (200, 280), (250, 250, 250)).save(buf, "PNG")
            pg.insert_image(pg.rect, stream=buf.getvalue())
        else:
            pg.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text, fontsize=11)
    doc.save(str(path))
    return path


def make_docx(path: Path, lines: list[str]) -> Path:
    from docx import Document

    d = Document()
    for line in lines:
        d.add_paragraph(line)
    d.save(str(path))
    return path
