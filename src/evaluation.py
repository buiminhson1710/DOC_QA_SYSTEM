"""Đánh giá RAG có đáp án chuẩn (chạy trong tiến trình, không cần khoá API).

Phạm vi đo ĐƯỢC ở chế độ này (không phụ thuộc chất lượng LLM):
  - Recall@K / MRR của bằng chứng chuẩn trong các nguồn mà pipeline đưa vào ngữ cảnh gửi AI;
  - độ phủ bằng chứng trong ngữ cảnh, kích thước ngữ cảnh (ước lượng token), độ trễ truy xuất;
  - nhận biết văn bản sửa đổi (nguồn chứa nội dung hiện hành có được đánh dấu "sửa đổi" không);
  - so sánh các cấu hình: vector-only / baseline trước nâng cấp / +hybrid / +ghim điều khoản.
KHÔNG đo được ở đây (cần LLM + embedding thật): độ đúng câu trả lời, groundedness, khả năng từ chối của LLM.
Các chỉ số đó được đo bằng `scripts/eval_rag.py live` trên máy chủ thật.

Embedding dùng ở chế độ này là BỘ GIẢ LẬP băm n-gram ký tự (thay cho model thật) => kết quả đo cơ chế pipeline,
KHÔNG dự đoán chất lượng của embedding thật. Kho tài liệu là dữ liệu TỔNG HỢP (số hiệu hư cấu).
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import random
import re
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from src.services.direct_answer import fold

# ---------------------------------------------------------------------------
# Kho tài liệu tổng hợp
# ---------------------------------------------------------------------------
DOC_A = "101/2030/NĐ-CP"   # văn bản gốc
DOC_B = "202/2031/NĐ-CP"   # sửa đổi, bổ sung một phần DOC_A + bãi bỏ một khoản
DOC_C = "303/2031/NĐ-CP"   # thay thế NĐ 55/2019, bãi bỏ NĐ 77/2018 (chưa nạp)
DOC_D = "15/2031/TT-BYT"   # thông tư độc lập (từ vựng khác)
DOC_F = "404/2030/NĐ-CP"   # bản "scan xấu": chữ bị nhiễu OCR

_A_ARTICLES = [
    ("Phạm vi điều chỉnh", "Nghị định này quy định về quản lý, sử dụng và thanh lý thiết bị y tế tại cơ sở khám bệnh, chữa bệnh công lập."),
    ("Đối tượng áp dụng", "Nghị định này áp dụng đối với cơ sở khám bệnh, chữa bệnh công lập và tổ chức, cá nhân có liên quan đến thiết bị y tế."),
    ("Giải thích từ ngữ", "Thiết bị y tế là máy móc, dụng cụ dùng để chẩn đoán, điều trị hoặc theo dõi người bệnh. Cơ sở sử dụng là đơn vị trực tiếp vận hành thiết bị."),
    ("Phân loại thiết bị y tế", "Thiết bị y tế được chia thành bốn nhóm A, B, C, D theo mức độ rủi ro. Nhóm A có rủi ro thấp nhất, nhóm D có rủi ro cao nhất."),
    ("Hạn mức mua sắm trực tiếp", "Hạn mức mua sắm trực tiếp thiết bị y tế là 5 tỷ đồng cho mỗi gói mua sắm. Vượt hạn mức này phải tổ chức lựa chọn nhà cung cấp theo quy định."),
    ("Thời hạn bảo hành tối thiểu", "Thời hạn bảo hành tối thiểu của thiết bị y tế là 24 tháng kể từ ngày nghiệm thu bàn giao đưa vào sử dụng."),
    ("Kiểm định định kỳ", "Thiết bị y tế nhóm C và nhóm D phải được kiểm định định kỳ 12 tháng một lần bởi tổ chức kiểm định được chỉ định."),
    ("Thanh lý thiết bị", "Hồ sơ đề nghị thanh lý thiết bị y tế phải được cơ quan có thẩm quyền giải quyết trong thời hạn 30 ngày làm việc kể từ ngày nhận đủ hồ sơ."),
    ("Báo cáo sự cố", "1. Cơ sở phải báo cáo sự cố nghiêm trọng trong vòng 24 giờ kể từ khi phát hiện.\n2. Báo cáo gửi đến cơ quan quản lý trực tiếp bằng văn bản điện tử.\n3. Cơ sở được miễn báo cáo đối với sự cố thuộc thiết bị nhóm A."),
    ("Lưu trữ hồ sơ thiết bị", "Hồ sơ thiết bị y tế được lưu trữ tối thiểu 10 năm kể từ ngày thanh lý hoặc ngừng sử dụng thiết bị."),
    ("Xử phạt vi phạm", "1. Phạt tiền từ 20 triệu đồng đến 40 triệu đồng đối với hành vi sử dụng thiết bị y tế chưa kiểm định.\n2. Phạt tiền từ 50 triệu đồng đến 80 triệu đồng đối với hành vi làm giả kết quả kiểm định."),
    ("Nhập khẩu thiết bị", "Giấy phép nhập khẩu thiết bị y tế có hiệu lực 36 tháng kể từ ngày cấp và được gia hạn một lần."),
    ("Đào tạo người vận hành", "Người vận hành thiết bị nhóm D phải hoàn thành khóa đào tạo tối thiểu 16 giờ và được cấp chứng nhận trước khi vận hành."),
    ("Hiệu lực thi hành", "Nghị định này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2031."),
]
_FILLER = " Các cơ sở có trách nhiệm thực hiện đúng quy định tại Nghị định này và chịu sự thanh tra, kiểm tra của cơ quan có thẩm quyền."


def _articles_text(articles: list[tuple[str, str]], start: int = 1) -> list[str]:
    return [f"Điều {i}. {title}\n{body}{_FILLER}" for i, (title, body) in enumerate(articles, start=start)]


def _doc_a_text() -> list[str]:
    head = f"Số: {DOC_A}\nNGHỊ ĐỊNH\nQuy định về quản lý, sử dụng thiết bị y tế tại cơ sở công lập\nChính phủ ban hành Nghị định."
    return [head] + _articles_text(_A_ARTICLES)


_FILLER_TOPICS = [
    "quản lý nhật ký vận hành", "bảo trì phòng ngừa", "an toàn bức xạ", "xử lý chất thải y tế", "kiểm soát nhiễm khuẩn",
    "bảo mật dữ liệu bệnh nhân", "kiểm kê tài sản cố định", "đánh giá nhà cung cấp", "quản lý phụ tùng thay thế", "hiệu chuẩn thiết bị đo",
]


def _doc_b_fillers() -> list[str]:
    """Các điều bổ sung dài, không liên quan câu hỏi, làm văn bản sửa đổi trải ra nhiều chunk."""
    out = []
    for i, topic in enumerate(_FILLER_TOPICS):
        letter = "abcdefghij"[i]
        body = (f"Cơ sở sử dụng thiết bị y tế thực hiện {topic} theo kế hoạch hằng năm, lập hồ sơ và lưu trữ phục vụ kiểm tra. "
                f"Người đứng đầu cơ sở chịu trách nhiệm tổ chức {topic}, phân công cán bộ phụ trách và báo cáo kết quả cho cơ quan quản lý. ") * 2
        out.append(f'{5 + i}. Bổ sung Điều 3{letter} như sau:\n"Điều 3{letter}. Quy định về {topic}\n{body}"')
    return out


def _doc_b_text() -> list[str]:
    return [
        f"Số: {DOC_B}\nNGHỊ ĐỊNH\nSửa đổi, bổ sung một số điều của Nghị định số {DOC_A}\nChính phủ ban hành Nghị định.",
        f"Điều 1. Sửa đổi, bổ sung một số điều của Nghị định số {DOC_A}",
        '1. Sửa đổi, bổ sung Điều 5 như sau:\n"Điều 5. Hạn mức mua sắm trực tiếp\nHạn mức mua sắm trực tiếp thiết bị y tế là 1,5 tỷ đồng cho mỗi gói mua sắm."',
        '2. Sửa đổi, bổ sung Điều 8 như sau:\n"Điều 8. Thanh lý thiết bị\nHồ sơ đề nghị thanh lý thiết bị y tế phải được giải quyết trong thời hạn 15 ngày làm việc kể từ ngày nhận đủ hồ sơ."',
        '3. Sửa đổi, bổ sung khoản 1 Điều 11 như sau:\n"1. Phạt tiền từ 30 triệu đồng đến 60 triệu đồng đối với hành vi sử dụng thiết bị y tế chưa kiểm định."',
        f"4. Bãi bỏ khoản 3 Điều 9 của Nghị định số {DOC_A}.",
    ] + _doc_b_fillers() + [
        '12. Sửa đổi, bổ sung Điều 12 như sau:\n"Điều 12. Nhập khẩu thiết bị\nGiấy phép nhập khẩu thiết bị y tế có hiệu lực 48 tháng kể từ ngày cấp và được gia hạn hai lần."',
        "Điều 2. Hiệu lực thi hành\nNghị định này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2031.",
    ]


def _doc_c_text() -> list[str]:
    arts = [
        ("Phạm vi điều chỉnh", "Nghị định này quy định về quản lý kho thuốc và vắc xin tại cơ sở y tế."),
        ("Điều kiện bảo quản", "Vắc xin phải được bảo quản ở nhiệt độ từ 2 đến 8 độ C và được theo dõi nhiệt độ liên tục."),
        ("Kiểm kê kho", "Kho thuốc phải được kiểm kê 6 tháng một lần và lập biên bản kiểm kê."),
        ("Lưu mẫu", "Thời hạn lưu mẫu thuốc là 3 năm kể từ ngày xuất kho lô thuốc."),
        ("Điều khoản thi hành", "1. Nghị định này thay thế Nghị định số 55/2019/NĐ-CP ngày 12 tháng 6 năm 2019 của Chính phủ về quản lý kho thuốc.\n2. Kể từ ngày Nghị định này có hiệu lực, bãi bỏ Nghị định số 77/2018/NĐ-CP ngày 5 tháng 5 năm 2018 của Chính phủ."),
    ]
    return [f"Số: {DOC_C}\nNGHỊ ĐỊNH\nQuy định về quản lý kho thuốc và vắc xin\nChính phủ ban hành Nghị định."] + _articles_text(arts)


def _doc_d_text() -> list[str]:
    arts = [
        ("Phạm vi điều chỉnh", "Thông tư này hướng dẫn việc kê khai giá trang thiết bị chẩn đoán in vitro."),
        ("Kê khai lại giá", "Thời hạn kê khai lại giá khi có biến động là 45 ngày kể từ ngày điều chỉnh giá."),
        ("Mẫu kê khai", "Cơ sở sử dụng mẫu kê khai số 01 ban hành kèm theo Thông tư này."),
        ("Công khai giá", "Giá kê khai được công khai trên cổng thông tin của cơ quan quản lý trong 5 ngày làm việc."),
        ("Hiệu lực", "Thông tư này có hiệu lực từ ngày 15 tháng 3 năm 2031."),
    ]
    return [f"Số: {DOC_D}\nTHÔNG TƯ\nHướng dẫn kê khai giá trang thiết bị chẩn đoán in vitro\nBộ trưởng Bộ Y tế ban hành Thông tư."] + _articles_text(arts)


_F_ARTICLES = [
    ("Phạm vi", "Nghị định này quy định về quản lý hóa chất xét nghiệm tại phòng xét nghiệm."),
    ("Bảo quản hóa chất", "Hóa chất xét nghiệm phải được bảo quản ở nhiệt độ 15 đến 25 độ C, tránh ánh sáng trực tiếp."),
    ("Hạn dùng sau mở nắp", "Hạn dùng của hóa chất sau khi mở nắp là 28 ngày nếu bảo quản đúng điều kiện."),
    ("Kiểm tra đầu vào", "Hóa chất nhập kho phải được kiểm tra chất lượng đầu vào trong 7 ngày."),
    ("Xử lý hóa chất hết hạn", "Hóa chất hết hạn phải được tiêu hủy trong thời hạn 60 ngày kể từ ngày hết hạn."),
]
_DIACRITIC = {"ă": "a", "â": "a", "ê": "e", "ô": "o", "ơ": "o", "ư": "u", "đ": "d", "á": "a", "à": "a", "ả": "a", "ạ": "a", "ã": "a",
              "é": "e", "è": "e", "ẻ": "e", "ẹ": "e", "í": "i", "ì": "i", "ị": "i", "ó": "o", "ò": "o", "ọ": "o", "ú": "u", "ù": "u", "ụ": "u",
              "ế": "e", "ề": "e", "ể": "e", "ệ": "e", "ố": "o", "ồ": "o", "ộ": "o", "ớ": "o", "ờ": "o", "ợ": "o", "ứ": "u", "ừ": "u", "ự": "u",
              "ấ": "a", "ầ": "a", "ậ": "a", "ắ": "a", "ằ": "a", "ặ": "a", "ý": "y", "ỳ": "y"}


def ocr_noise(text: str, rate: float = 0.4, seed: int = 7) -> str:
    """Mô phỏng OCR xấu: rụng dấu tiếng Việt (xác suất `rate`) và nhầm 'm'->'rn' ở một số chữ. Giữ nguyên chữ số."""
    rng = random.Random(seed)
    out = []
    for ch in text:
        low = ch.lower()
        if low in _DIACRITIC and rng.random() < rate:
            repl = _DIACRITIC[low]
            out.append(repl.upper() if ch.isupper() else repl)
        elif low == "m" and rng.random() < 0.05:
            out.append("rn")
        else:
            out.append(ch)
    return "".join(out)


def _doc_f_text() -> list[str]:
    return [ocr_noise(t) for t in
            [f"Số: {DOC_F}\nNGHỊ ĐỊNH\nQuy định về quản lý hóa chất xét nghiệm\nChính phủ ban hành Nghị định."] + _articles_text(_F_ARTICLES)]


CORPUS = {
    "NĐ 101-2030 (gốc).docx": _doc_a_text,
    "NĐ 202-2031 (sửa đổi).docx": _doc_b_text,
    "NĐ 303-2031 (thay thế-bãi bỏ).docx": _doc_c_text,
    "TT 15-2031-TT-BYT.docx": _doc_d_text,
    "NĐ 404-2030 (scan xấu).docx": _doc_f_text,
}


def build_corpus(directory: Path) -> list[Path]:
    """Tạo các file DOCX (không cần font hệ thống) trong `directory`."""
    from docx import Document

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, builder in CORPUS.items():
        doc = Document()
        for page in builder():
            for line in page.split("\n"):
                doc.add_paragraph(line)
        path = directory / name
        doc.save(str(path))
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Bộ câu hỏi chuẩn. evidence: chuỗi nguyên văn chứa đáp án (so khớp không phân biệt dấu/hoa thường).
# stale: giá trị CŨ đã bị sửa đổi (không được là nguồn duy nhất). answerable=False: tài liệu không có thông tin.
# ---------------------------------------------------------------------------
GOLDEN: list[dict[str, Any]] = [
    # --- tra cứu điều/khoản ---
    {"id": "A1", "category": "lookup", "q": "Thời hạn bảo hành tối thiểu của thiết bị y tế theo Nghị định 101/2030/NĐ-CP là bao lâu?", "evidence": ["bảo hành tối thiểu của thiết bị y tế là 24 tháng"]},
    {"id": "A2", "category": "lookup", "q": "Thiết bị nhóm C và D phải kiểm định định kỳ bao lâu một lần?", "evidence": ["kiểm định định kỳ 12 tháng một lần"]},
    {"id": "A3", "category": "lookup", "q": "Hồ sơ thiết bị y tế phải lưu trữ tối thiểu bao nhiêu năm?", "evidence": ["lưu trữ tối thiểu 10 năm"]},
    {"id": "A4", "category": "lookup", "q": "Thiết bị y tế được chia thành mấy nhóm theo mức độ rủi ro?", "evidence": ["bốn nhóm A, B, C, D"]},
    {"id": "A5", "category": "lookup", "q": "Người vận hành thiết bị nhóm D phải đào tạo tối thiểu mấy giờ?", "evidence": ["đào tạo tối thiểu 16 giờ"]},
    {"id": "A6", "category": "lookup", "q": "Điều 7 Nghị định 101/2030/NĐ-CP quy định thế nào về kiểm định?", "evidence": ["kiểm định định kỳ 12 tháng một lần"]},
    {"id": "A7", "category": "lookup", "q": "Báo cáo sự cố nghiêm trọng phải gửi trong bao lâu?", "evidence": ["báo cáo sự cố nghiêm trọng trong vòng 24 giờ"]},
    {"id": "A8", "category": "lookup", "q": "Vắc xin phải bảo quản ở nhiệt độ nào?", "evidence": ["nhiệt độ từ 2 đến 8 độ C"]},
    # --- văn bản đã bị sửa đổi: phải lấy nội dung hiện hành ---
    {"id": "M1", "category": "amended", "q": "Hạn mức mua sắm trực tiếp thiết bị y tế hiện nay là bao nhiêu?", "evidence": ["là 1,5 tỷ đồng cho mỗi gói mua sắm"], "stale": ["là 5 tỷ đồng cho mỗi gói mua sắm"]},
    {"id": "M2", "category": "amended", "q": "Điều 5 Nghị định 101/2030/NĐ-CP về hạn mức mua sắm trực tiếp quy định thế nào?", "evidence": ["là 1,5 tỷ đồng cho mỗi gói mua sắm"], "stale": ["là 5 tỷ đồng cho mỗi gói mua sắm"]},
    {"id": "M3", "category": "amended", "q": "Thanh lý thiết bị y tế phải được giải quyết trong thời hạn bao nhiêu ngày làm việc?", "evidence": ["giải quyết trong thời hạn 15 ngày làm việc"], "stale": ["giải quyết trong thời hạn 30 ngày làm việc"]},
    {"id": "M4", "category": "amended", "q": "Mức phạt khi sử dụng thiết bị y tế chưa kiểm định là bao nhiêu?", "evidence": ["từ 30 triệu đồng đến 60 triệu đồng"], "stale": ["từ 20 triệu đồng đến 40 triệu đồng"]},
    {"id": "M5", "category": "amended", "q": "Điều 8 Nghị định 101/2030/NĐ-CP được sửa đổi như thế nào?", "evidence": ["giải quyết trong thời hạn 15 ngày làm việc"], "stale": []},
    {"id": "M7", "category": "amended_long", "q": "Điều 12 Nghị định 101/2030/NĐ-CP quy định thế nào?", "evidence": ["có hiệu lực 48 tháng kể từ ngày cấp"], "stale": ["có hiệu lực 36 tháng kể từ ngày cấp"]},
    {"id": "M8", "category": "amended_long", "q": "Giấy phép nhập khẩu thiết bị y tế hiện hành có hiệu lực bao lâu?", "evidence": ["có hiệu lực 48 tháng kể từ ngày cấp"], "stale": ["có hiệu lực 36 tháng kể từ ngày cấp"]},
    {"id": "M9", "category": "amended_long", "q": "Theo Điều 12 Nghị định 101/2030/NĐ-CP, giấy phép nhập khẩu được gia hạn mấy lần?", "evidence": ["được gia hạn hai lần"], "stale": ["được gia hạn một lần"]},
    {"id": "M6", "category": "amended", "q": "Cơ sở có còn được miễn báo cáo sự cố thiết bị nhóm A không?", "evidence": ["Bãi bỏ khoản 3 Điều 9"], "stale": []},
    # --- nhiều văn bản ---
    {"id": "X1", "category": "multi_doc", "q": "So sánh thời hạn kê khai lại giá và thời hạn bảo hành tối thiểu của thiết bị y tế", "evidence": ["kê khai lại giá khi có biến động là 45 ngày", "bảo hành tối thiểu của thiết bị y tế là 24 tháng"]},
    {"id": "X2", "category": "multi_doc", "q": "Nêu thời hạn lưu mẫu thuốc và thời hạn lưu trữ hồ sơ thiết bị y tế", "evidence": ["lưu mẫu thuốc là 3 năm", "lưu trữ tối thiểu 10 năm"]},
    {"id": "X3", "category": "multi_doc", "q": "Thời hạn công khai giá kê khai và thời hạn xử lý hồ sơ thanh lý thiết bị", "evidence": ["công khai trên cổng thông tin của cơ quan quản lý trong 5 ngày làm việc", "giải quyết trong thời hạn 15 ngày làm việc"]},
    # --- quan hệ văn bản ---
    {"id": "R1", "category": "relation", "q": "Nghị định nào bãi bỏ Nghị định số 77/2018/NĐ-CP?", "evidence": ["bãi bỏ Nghị định số 77/2018/NĐ-CP"]},
    {"id": "R2", "category": "relation", "q": "Nghị định 303/2031/NĐ-CP thay thế văn bản nào?", "evidence": ["thay thế Nghị định số 55/2019/NĐ-CP"]},
    {"id": "R3", "category": "relation", "q": "Nghị định nào sửa đổi, bổ sung Nghị định 101/2030/NĐ-CP?", "evidence": ["Sửa đổi, bổ sung một số điều của Nghị định số 101/2030/NĐ-CP"]},
    # --- văn bản chất lượng OCR kém ---
    {"id": "O1", "category": "ocr_noisy", "q": "Hạn dùng của hóa chất xét nghiệm sau khi mở nắp là bao lâu?", "evidence": ["sau khi mở nắp là 28 ngày"]},
    {"id": "O2", "category": "ocr_noisy", "q": "Hóa chất xét nghiệm hết hạn phải tiêu hủy trong bao nhiêu ngày?", "evidence": ["tiêu hủy trong thời hạn 60 ngày"]},
    {"id": "O3", "category": "ocr_noisy", "q": "Hóa chất xét nghiệm bảo quản ở nhiệt độ bao nhiêu?", "evidence": ["nhiệt độ 15 đến 25 độ C"]},
    # --- diễn đạt khác, không nêu số điều ---
    {"id": "P1", "category": "paraphrase", "q": "Tổ chức nào được chỉ định để kiểm định máy móc chẩn đoán, và tần suất ra sao?", "evidence": ["kiểm định định kỳ 12 tháng một lần"]},
    {"id": "P2", "category": "paraphrase", "q": "Sau khi mở nắp, thuốc thử dùng được thêm mấy ngày?", "evidence": ["sau khi mở nắp là 28 ngày"]},
    {"id": "P3", "category": "paraphrase", "q": "Khi nào thì cơ sở khám chữa bệnh phải thông báo tai nạn thiết bị cho cơ quan quản lý?", "evidence": ["báo cáo sự cố nghiêm trọng trong vòng 24 giờ"]},
    # --- không có dữ liệu trong kho ---
    {"id": "N1", "category": "no_data", "q": "Mức thuế thu nhập cá nhân đối với tiền lương là bao nhiêu phần trăm?", "evidence": [], "answerable": False},
    {"id": "N2", "category": "no_data", "q": "Điều kiện hưởng bảo hiểm thất nghiệp theo luật hiện hành?", "evidence": [], "answerable": False},
    {"id": "N3", "category": "no_data", "q": "Quy định về xây dựng nhà ở xã hội tại đô thị loại đặc biệt", "evidence": [], "answerable": False},
]


# ---------------------------------------------------------------------------
# Embedding giả lập (n-gram ký tự băm) — chỉ dùng để chạy pipeline không cần khoá API
# ---------------------------------------------------------------------------
def proxy_vector(text: str, dim: int) -> list[float]:
    t = " " + fold(text) + " "
    vec = [0.0] * dim
    for i in range(len(t) - 2):
        h = int(hashlib.md5(t[i : i + 3].encode()).hexdigest()[:8], 16)
        vec[h % dim] += 1.0 if (h >> 12) & 1 else -1.0 if (h >> 13) & 1 else 0.6
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def make_proxy_aembedding(dim: int):
    async def aembedding(model: str, input: list[str], dimensions: Optional[int] = None, **_: Any) -> Any:  # noqa: A002
        return SimpleNamespace(
            data=[{"embedding": proxy_vector(t, dim), "index": i} for i, t in enumerate(input)],
            usage=SimpleNamespace(prompt_tokens=sum(max(1, len(t) // 3) for t in input)),
        )
    return aembedding


# ---------------------------------------------------------------------------
# Chấm điểm
# ---------------------------------------------------------------------------
CONFIGS: dict[str, dict[str, bool]] = {
    "vector_only": {"hybrid_search": False, "pin_articles": False, "expand_amendments": False},
    "baseline_prev": {"hybrid_search": False, "pin_articles": False, "expand_amendments": True},   # ≈ hành vi trước nâng cấp
    "hybrid": {"hybrid_search": True, "pin_articles": False, "expand_amendments": True},
    "full": {"hybrid_search": True, "pin_articles": True, "expand_amendments": True},
}
KS = (1, 3, 5)


def _norm(text: str) -> str:
    return fold(text)


def score_question(q: dict[str, Any], sources: list[dict[str, Any]], ranked_hits: Optional[list[str]] = None) -> dict[str, Any]:
    """Hai nhóm chỉ số, KHÔNG gộp:
      - bước TRUY XUẤT (`ranked_hits`: nội dung các chunk theo thứ hạng độ liên quan, trước khi mở rộng/ghim): Recall@K, MRR;
      - bước NGỮ CẢNH (`sources`: những gì thực sự gửi cho AI): phủ bằng chứng, có giá trị cũ mà thiếu giá trị mới, ...
    Thứ tự trong ngữ cảnh không phải thứ hạng độ liên quan (văn bản sửa đổi được xếp trước) nên không dùng để tính Recall@K."""
    evidence = [_norm(e) for e in q.get("evidence", [])]
    stale = [_norm(e) for e in q.get("stale", [])]
    texts = [_norm(s["content"]) for s in sources]
    out: dict[str, Any] = {"id": q["id"], "category": q["category"], "n_sources": len(sources)}
    if evidence:
        hit_texts = [_norm(t) for t in (ranked_hits if ranked_hits is not None else [s["content"] for s in sources])]
        found_rank = {e: next((i for i, t in enumerate(hit_texts, 1) if e in t), None) for e in evidence}
        out["evidence_rank"] = found_rank
        for k in KS:
            out[f"recall@{k}"] = sum(1 for r in found_rank.values() if r is not None and r <= k) / len(evidence)
        first = min((r for r in found_rank.values() if r is not None), default=None)
        out["mrr"] = 1.0 / first if first else 0.0
        ctx_found = [any(e in t for t in texts) for e in evidence]
        out["context_hit"] = all(ctx_found)
        out["context_recall"] = sum(ctx_found) / len(evidence)
    if stale:
        has_stale = any(any(s in t for t in texts) for s in stale)
        current_marked = any(e in t and src["is_amendment"] for e in evidence for t, src in zip(texts, sources))
        out["stale_in_context"] = has_stale
        out["current_marked_amendment"] = current_marked
        # lỗi nghiêm trọng: chỉ có giá trị cũ, không có giá trị hiện hành
        out["stale_without_current"] = has_stale and not out.get("context_hit", False)
    return out


def _agg(rows: list[dict[str, Any]], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    return round(sum(float(v) for v in vals) / len(vals), 3) if vals else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [f"recall@{k}" for k in KS] + ["mrr", "context_hit", "context_recall", "stale_in_context", "current_marked_amendment", "stale_without_current"]
    return {"n": len(rows), **{k: _agg(rows, k) for k in keys}}


async def evaluate_config(engine: Any, name: str, flags: dict[str, bool], top_k: int = 6) -> dict[str, Any]:
    from src.services.rag_engine import _Timer

    original = engine._s
    engine._s = dataclasses.replace(original, **flags)
    engine._lex = None
    rows: list[dict[str, Any]] = []
    unanswerable: list[dict[str, Any]] = []
    latencies: list[float] = []
    ctx_tokens: list[int] = []
    try:
        for q in GOLDEN:
            t0 = time.perf_counter()
            tm = _Timer()
            prep = await engine._prepare(q["q"], top_k, tm)
            latencies.append((time.perf_counter() - t0) * 1000)
            context = engine._build_context(prep["sources"])
            ctx_tokens.append(math.ceil(len(context) / 3))
            if q.get("answerable", True):
                ranked = [h.payload.get("content", "") for h in prep["hits"]]
                rows.append({**score_question(q, prep["sources"], ranked), "context_chars": len(context), "top_vector_score": round(prep["top_score"], 4)})
            else:
                unanswerable.append({"id": q["id"], "top_vector_score": prep["top_score"], "n_sources": len(prep["sources"])})
    finally:
        engine._s = original
        engine._lex = None
    by_cat: dict[str, Any] = {}
    for cat in sorted({r["category"] for r in rows}):
        by_cat[cat] = aggregate([r for r in rows if r["category"] == cat])
    return {
        "config": name, "flags": flags, "overall": aggregate(rows), "by_category": by_cat,
        "latency_ms": {"p50": round(statistics.median(latencies), 1), "p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 1)},
        "est_context_tokens": {"mean": round(statistics.mean(ctx_tokens)), "max": max(ctx_tokens)},
        "unanswerable": unanswerable, "answerable_top_vector_score": [r["top_vector_score"] for r in rows], "per_question": rows,
    }


async def run_inprocess(top_k: int = 6, workdir: Optional[Path] = None, configs: Optional[dict[str, dict[str, bool]]] = None) -> dict[str, Any]:
    """Nạp kho tổng hợp vào Qdrant nhúng (embedding giả lập) rồi chấm các cấu hình."""
    import os

    import litellm

    from src.config import get_settings
    from src.services.qdrant_service import QdrantService
    from src.services.rag_engine import RAGEngine
    from src.services.relations_service import RelationsService
    from src.services.security_guard import SecurityGuard

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="docqa_eval_"))
    saved = {k: os.environ.get(k) for k in ("DATA_DIR", "QDRANT_PATH", "EMBEDDING_DIM")}
    os.environ.update({"DATA_DIR": str(tmp), "QDRANT_PATH": str(tmp / "qdrant"), "EMBEDDING_DIM": "256"})
    get_settings.cache_clear()
    settings = get_settings()
    real_embedding = litellm.aembedding
    litellm.aembedding = make_proxy_aembedding(settings.embedding_dim)
    try:
        qdrant = QdrantService(settings)
        relations = RelationsService(qdrant, settings)
        engine = RAGEngine(qdrant, relations, SecurityGuard(settings), settings)
        paths = build_corpus(tmp / "docs")
        ingested = [await engine.ingest_file(p) for p in paths]
        results = {}
        for name, flags in (configs or CONFIGS).items():
            results[name] = await evaluate_config(engine, name, flags, top_k)
        report = {
            "note": "Embedding GIẢ LẬP (băm n-gram ký tự) + kho tổng hợp: đo cơ chế pipeline, không dự đoán chất lượng embedding thật.",
            "corpus": [{"doc": r["doc_number"], "chunks": r["chunks"], "ingest": r["ingest_status"], "relations": r["relations"]} for r in ingested],
            "n_questions": len(GOLDEN), "top_k": top_k, "configs": results,
            "not_measured_here": ["độ đúng câu trả lời", "groundedness của LLM", "khả năng từ chối của LLM", "token thật của nhà cung cấp"],
        }
        try:
            qdrant.client.close()
        except Exception:  # noqa: BLE001
            pass
        return report
    finally:
        litellm.aembedding = real_embedding
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_settings.cache_clear()


def format_markdown(report: dict[str, Any]) -> str:
    lines = [f"> {report['note']}", "",
             "| Cấu hình | R@1 | R@3 | R@5 | MRR | Phủ bằng chứng (ngữ cảnh) | Chỉ có giá trị cũ | Mới được đánh dấu sửa đổi | Token ngữ cảnh TB | p50 ms |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["configs"].items():
        o = r["overall"]
        am = r["by_category"].get("amended", {})
        lines.append(f"| {name} | {o['recall@1']} | {o['recall@3']} | {o['recall@5']} | {o['mrr']} | {o['context_hit']} | "
                     f"{am.get('stale_without_current')} | {am.get('current_marked_amendment')} | {r['est_context_tokens']['mean']} | {r['latency_ms']['p50']} |")
    cats = sorted({c for r in report["configs"].values() for c in r["by_category"]})
    lines += ["", "**Theo nhóm — Recall@3 (truy xuất) / Phủ bằng chứng (ngữ cảnh)**", "", "| Nhóm | " + " | ".join(report["configs"]) + " |", "|---|" + "---|" * len(report["configs"])]
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(f"{r['by_category'][c]['recall@3']} / {r['by_category'][c]['context_hit']}" for r in report["configs"].values()) + " |")
    return "\n".join(lines)
