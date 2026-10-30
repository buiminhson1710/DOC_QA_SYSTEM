"""Trả lời TRỰC TIẾP không cần LLM (0 token) cho các câu hỏi tra cứu có cấu trúc rõ ràng.

Chỉ nhận diện các dạng câu hỏi mà câu trả lời đúng là một *trích nguyên văn* hoặc một *dữ kiện đã có trong đồ thị quan hệ*:
  - article:             "Điều 18 Nghị định 214/2025/NĐ-CP quy định thế nào?"        -> trích nguyên văn Điều 18
  - article_amendments:  "Điều 18 Nghị định 214 được sửa đổi như thế nào?"           -> trích phần sửa đổi trong NĐ sửa đổi
  - status:              "Nghị định 53/2013/NĐ-CP còn hiệu lực không?"               -> tra đồ thị quan hệ
  - affects:             "Nghị định 359/2026 bãi bỏ những nghị định nào?"            -> tra đồ thị quan hệ
  - list:                "Hệ thống đang có những văn bản nào?"                       -> liệt kê văn bản đã nạp
Mọi câu cần suy luận / giải thích / so sánh / điều kiện / yes-no ("có được... không", "tóm tắt", "thủ tục"...) KHÔNG đi vào
đây mà chuyển sang LLM. Khi không chắc chắn, trả về None (nhường cho LLM) chứ không đoán.
"""
from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

_DOC_WORD_TO_TYPE = {
    "nghi dinh": "Nghị định", "thong tu": "Thông tư", "quyet dinh": "Quyết định",
    "luat": "Luật", "nghi quyet": "Nghị quyết",
}

# Từ cho thấy câu hỏi cần suy luận/giải thích -> không trả lời trực tiếp
_NEEDS_REASONING = (
    "tom tat", "giai thich", "so sanh", "phan tich", "khac gi", "khac nhau", "giong", "vi du", "co duoc", "co phai",
    "co can", "co bi", "khong", "dieu kien", "thu tuc", "trinh tu", "truong hop", "bao nhieu", "may",
    "thoi han", "thoi gian", "khi nao", "bao gio", "ai", "phai lam", "ap dung cho", "co nghia", "hieu the nao",
)
_ASKS_CONTENT = (
    "quy dinh", "noi dung", "noi gi", "the nao", "ra sao", "la gi", "cho xem", "hien thi", "xem", "doc", "trich",
    "nguyen van", "gom nhung gi", "bao gom", "nhu the nao", "nhu sau", "ghi gi",
)
_AMEND_WORDS = ("sua doi", "bo sung", "thay doi", "bi sua", "duoc sua")
_STATUS_WORDS = (
    "con hieu luc", "het hieu luc", "con ap dung", "trang thai", "da bi", "bi bai bo", "bi thay the", "bi sua doi",
    "da duoc sua doi", "duoc sua doi boi", "sua doi boi", "bai bo boi", "thay the boi", "boi van ban nao",
    "boi nghi dinh nao", "van ban nao sua", "van ban nao bai bo", "van ban nao thay the", "ai sua doi", "ai bai bo",
)
_DATE_WORDS = ("tu ngay", "khi nao", "bao gio", "ngay nao", "thoi diem", "ngay co hieu luc", "hieu luc tu", "co hieu luc tu")
_AFFECT_VERBS = ("bai bo", "thay the", "sua doi", "bo sung")
_LISTING = ("nhung", "cac", "nao", "gi")
_DETAIL_WORDS = ("dieu nao", "khoan nao", "diem nao", "noi dung", "cu the", "chi tiet", "nhu the nao", "ra sao")
_LIST_SCOPE = ("he thong", "da nap", "da tai len", "hien co", "dang co", "kho")
_LIST_OBJ = ("van ban", "tai lieu")
_LIST_ASK = ("nhung", "cac", "danh sach", "liet ke", "nao")
_LIST_EXCLUDE = ("quy dinh ve", "noi dung", "dieu ", "tom tat")

_FULL_REF = re.compile(r"(?<![\d/])(\d{1,4})\s*/\s*(\d{4})(?:\s*/\s*([a-z][a-z0-9\-]*))?")
_SHORT_REF = re.compile(r"\b(nghi dinh|thong tu|quyet dinh|luat|nghi quyet)\s*(?:so\s*)?(\d{1,4})\b(?!\s*/)")


_HAS_CACHE: dict[int, re.Pattern[str]] = {}


def _has(q: str, words: tuple[str, ...]) -> bool:
    """True nếu q chứa bất kỳ cụm nào trong words (khớp theo ranh giới từ)."""
    pat = _HAS_CACHE.get(id(words))
    if pat is None:
        pat = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in words) + r")(?!\w)")
        _HAS_CACHE[id(words)] = pat
    return bool(pat.search(q))


def fold(text: str) -> str:
    """Bỏ dấu tiếng Việt, hạ chữ thường, chuẩn hoá khoảng trắng."""
    text = unicodedata.normalize("NFC", text or "").lower().replace("đ", "d")
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s/\-]", " ", text)).strip()


@dataclass
class Intent:
    kind: str  # article | article_amendments | status | affects | list | amendments_list
    doc: Optional[str] = None
    article: Optional[str] = None
    related_doc: Optional[str] = None


def _known_docs(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mọi số hiệu hệ thống biết tới: văn bản đã nạp + văn bản được nhắc tới trong quan hệ."""
    known: dict[str, dict[str, Any]] = {}
    for rec in records:
        known[rec["doc_number"]] = {"type": rec.get("doc_type"), "ingested": True}
    for rec in records:
        for rel in rec.get("relations", []):
            known.setdefault(rel["target"], {"type": None, "ingested": False})
    return known


known_docs = _known_docs


def resolve_docs(q: str, known: dict[str, dict[str, Any]]) -> list[str]:
    """Tìm các văn bản được nhắc tới trong câu hỏi (q đã fold). Trả về danh sách số hiệu duy nhất."""
    found: list[str] = []

    def add(number: str) -> None:
        if number not in found:
            found.append(number)

    for m in _FULL_REF.finditer(q):
        num, year, suffix = m.group(1), m.group(2), m.group(3)
        prefix = f"{int(num)}/{year}/"
        for doc in known:
            folded = fold(doc)
            if folded.startswith(prefix) and (not suffix or folded.endswith(suffix)):
                add(doc)
    for m in _SHORT_REF.finditer(q):
        word, num = m.group(1), str(int(m.group(2)))
        want = _DOC_WORD_TO_TYPE[word]
        matches = [d for d, info in known.items() if d.split("/")[0] == num and info["type"] in (want, None)]
        if len(matches) == 1:  # chỉ nhận khi KHÔNG mơ hồ
            add(matches[0])

    # Nhận diện chuỗi số hiệu đi liền hoặc nhắc kèm (ví dụ: "nghị định 214 và 349", "214, 349")
    for doc in known:
        num = doc.split("/")[0]
        if num.isdigit() and len(num) >= 2:
            if re.search(rf"\b{num}\b", q):
                if found or any(w in q for w in ("nghi dinh", "thong tu", "luat", "quyet dinh", "dieu", "khoan", "van ban")):
                    add(doc)

    return found


def detect_intent(question: str, records: list[dict[str, Any]]) -> Optional[Intent]:
    q = fold(question)
    known = _known_docs(records)
    docs = resolve_docs(q, known)
    articles = set(re.findall(r"\bdieu\s+(\d+[a-z]?)\b", q))
    # Câu hỏi về hạn mức giá trị chỉ định thầu
    if (
        not docs
        and not articles
        and "chi dinh thau" in q
        and any(
            term in q
            for term in (
                "bao nhieu",
                "gia tri",
                "han muc",
                "nguong",
                "khong qua",
                "muc tien",
            )
        )
        and known.get("214/2025/NĐ-CP", {}).get("ingested")
    ):
        return Intent(
            kind="thresholds",
            doc="214/2025/NĐ-CP",
            article="78",
        )    
    # Câu hỏi chủ đề chung về chỉ định thầu
    if (
        not docs
        and not articles
        and "chi dinh thau" in q
        and any(
            term in q
            for term in (
                "truong hop",
                "ap dung",
                "quy dinh",
                "dieu kien",
                "nao",
                "bao gom",
            )
        )
        and known.get("214/2025/NĐ-CP", {}).get("ingested")
    ):
        return Intent(
            kind="article_topic",
            doc="214/2025/NĐ-CP",
            article="78",
        )
    if (not docs and not articles and len(q) < 100 and _has(q, _LIST_SCOPE) and _has(q, _LIST_OBJ)
            and _has(q, _LIST_ASK) and not any(w in q for w in _LIST_EXCLUDE)):
        return Intent("list")

        # Liệt kê nội dung văn bản sửa đổi một văn bản gốc
    if (
        len(docs) == 2
        and _has(q, _AMEND_WORDS)
        and (
            "noi dung nao" in q
            or "nhung gi" in q
            or "nhung dieu nao" in q
            or "liet ke" in q
            or _has(q, _LISTING)
        )
    ):
        amendment_doc, original_doc = docs[0], docs[1]
        return Intent(
            kind="amendments_list",
            doc=original_doc,
            related_doc=amendment_doc,
        )

    if len(docs) != 1:
        return None
    doc = docs[0]

    if len(articles) == 1:
        art = next(iter(articles))
        if not known[doc]["ingested"]:
            return None
        if _has(q, _AMEND_WORDS):
            return Intent("article_amendments", doc, art)
        if _has(q, _ASKS_CONTENT) and not _has(q, _NEEDS_REASONING):
            return Intent("article", doc, art)
        return None

    if articles:  # nhiều điều -> để LLM
        return None

    if _has(q, _DATE_WORDS):
        return None
    if _has(q, _STATUS_WORDS):
        return Intent("status", doc)
    if (
        _has(q, _AFFECT_VERBS)
        and _has(q, _LISTING)
        and not _has(q, _DETAIL_WORDS)
        and known[doc]["ingested"]
    ):
        return Intent("affects", doc)
    return None


# ---------------------------------------------------------------------------
# Ghép chunk -> văn bản liền mạch, trích điều khoản
# ---------------------------------------------------------------------------
def _overlap(prev: str, nxt: str, max_k: int = 400, min_k: int = 20) -> int:
    for k in range(min(max_k, len(prev), len(nxt)), min_k - 1, -1):
        if prev.endswith(nxt[:k]):
            return k
    return 0


def merge_chunks(chunks: list[dict[str, Any]]) -> tuple[str, list[int], list[Optional[int]]]:
    """Ghép chunk (đã sắp theo chunk_index), bỏ phần chồng lấn. Trả về (văn bản, offset bắt đầu từng chunk, trang)."""
    text = ""
    starts: list[int] = []
    pages: list[Optional[int]] = []
    for c in chunks:
        content = c.get("content", "")
        if not text:
            starts.append(0)
            text = content
        else:
            k = _overlap(text, content)
            starts.append(len(text) - k if k else len(text) + 1)
            text += content[k:] if k else "\n" + content
        pages.append(c.get("page"))
    return text, starts, pages


def page_at(offset: int, starts: list[int], pages: list[Optional[int]]) -> Optional[int]:
    i = max(0, bisect.bisect_right(starts, offset) - 1)
    return pages[i] if pages else None


def _heading_re(number: Optional[str] = None) -> re.Pattern[str]:
    num = re.escape(number) if number else r"[\dó§a-z]+"
    return re.compile(rf"(?mi)^[\s\-_—|l1I*~`]*(?:Đi[ềêèe]u|ĐI[ỀÊÈE]U)\s+({num})\s*[,.:\s\-—]")


def extract_article(text: str, number: str) -> Optional[tuple[int, int]]:
    """Tìm khoảng [start, end) của 'Điều N' (chọn đoạn dài nhất để bỏ qua mục lục)."""
    any_heading = _heading_re()
    best: Optional[tuple[int, int]] = None
    for m in _heading_re(number).finditer(text):
        nxt = any_heading.search(text, m.end())
        end = nxt.start() if nxt else len(text)
        if best is None or (end - m.start()) > (best[1] - best[0]):
            best = (m.start(), end)
    if best and best[1] - best[0] < 40:  # chỉ là dòng mục lục
        return None
    return best


_ITEM_START = re.compile(r"(?m)^[ \t]*(\d+)\.\s+(?=(?:Sửa đổi|Bổ sung|Bãi bỏ|Thay thế|Sửa|Bỏ)\b)")
_ARTICLE_START = re.compile(r"(?m)^[ \t]*Điều\s+\d+[a-z]?\s*[.:]")


def extract_amendment_blocks(
    text: str,
    number: str,
    max_len: int = 6000,
) -> list[tuple[int, str]]:
    """Ưu tiên lấy đúng điều của văn bản sửa đổi nhắm tới Điều gốc."""
    ref = re.compile(
        rf"(?i)đi[ềêèe]u\s+{re.escape(number)}\b(?![\d/])"
    )

    # 1. Tách các điều cấp cao trong văn bản sửa đổi.
    article_matches = list(_ARTICLE_START.finditer(text))
    matched_articles: list[tuple[int, str]] = []

    for i, match in enumerate(article_matches):
        start = match.start()
        end = (
            article_matches[i + 1].start()
            if i + 1 < len(article_matches)
            else len(text)
        )
        block = text[start:end].strip()

        # Tiêu đề kết thúc trước mục đánh số đầu tiên,
        # ví dụ "1. Sửa đổi, bổ sung ...".
        first_item = _ITEM_START.search(block)
        heading = block[:first_item.start()] if first_item else block[:350]
        heading_folded = fold(heading)

        has_amend_action = any(
            word in heading_folded
            for word in (
                "sua doi",
                "bo sung",
                "bai bo",
                "thay the",
            )
        )

        # Chỉ nhận khi chính tiêu đề điều sửa đổi nhắc tới
        # số Điều cần tìm, không chỉ nội dung bên trong nhắc tới.
        if has_amend_action and ref.search(heading):
            content = block[:max_len]
            if len(block) > max_len:
                content += "…"
            matched_articles.append((start, content))

    # Có tiêu đề khớp thì trả về kết quả ưu tiên này,
    # không trộn các đoạn chỉ tình cờ nhắc đến Điều cần tìm.
    if matched_articles:
        return matched_articles

    # 2. Fallback cho dạng sửa đổi mà tiêu đề không nêu rõ Điều gốc.
    starts = [m.start() for m in _ITEM_START.finditer(text)]
    boundaries = sorted(
        set(
            starts
            + [m.start() for m in _ARTICLE_START.finditer(text)]
            + [len(text)]
        )
    )

    blocks: list[tuple[int, str]] = []

    for start in starts:
        end = next(
            (boundary for boundary in boundaries if boundary > start),
            len(text),
        )
        block = text[start:end].strip()

        if ref.search(block[:260]):
            content = block[:max_len]
            if len(block) > max_len:
                content += "…"
            blocks.append((start, content))

    return blocks
