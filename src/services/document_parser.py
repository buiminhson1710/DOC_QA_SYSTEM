"""Bóc tách văn bản PDF/DOCX, trích số hiệu (doc_number) và quan hệ pháp lý."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import litellm

try:
    import pymupdf as fitz  # pymupdf >= 1.24.3
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.services.logger_service import get_logger

logger = get_logger("parser")


class ParseError(Exception):
    """Không đọc/bóc tách được tài liệu."""


class OCRIncomplete(ParseError):
    """OCR chưa xong toàn bộ trang scan. Kết quả đã làm được nằm trong cache nên chạy lại sẽ TIẾP TỤC phần còn thiếu."""


# ---------------------------------------------------------------------------
# Số hiệu văn bản
# ---------------------------------------------------------------------------
# Ví dụ: 214/2025/NĐ-CP, 22/2023/QH15, 18/2016/NĐ-CP
NUM_BODY = r"\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐ]{1,6}\d{0,2}(?:-[A-ZĐ]{1,8})*"
DOC_NUM_RE = re.compile(rf"(?<![\d/])({NUM_BODY})")
_HEADER_NUM_RE = re.compile(rf"S[ốo]\s*:?\s*({NUM_BODY})")

_DOC_TYPES = {
    "NĐ": "Nghị định",
    "TT": "Thông tư",
    "QĐ": "Quyết định",
    "QH": "Luật",
    "NQ": "Nghị quyết",
    "UBTVQH": "Pháp lệnh",
}


def normalize_doc_number(raw: str) -> str:
    """'214 / 2025 / ND-CP' -> '214/2025/NĐ-CP'."""
    compact = re.sub(r"\s+", "", raw)
    parts = compact.split("/")
    if len(parts) != 3:
        return compact.upper()
    number, year, suffix = parts
    suffix = suffix.upper()
    suffix = re.sub(r"^(N|Q)D", lambda m: m.group(1) + "Đ", suffix)  # ND-CP -> NĐ-CP
    try:
        number = str(int(number))
    except ValueError:
        pass
    return f"{number}/{year}/{suffix}"


def doc_type_of(doc_number: str) -> Optional[str]:
    parts = doc_number.split("/")
    if len(parts) != 3:
        return None
    prefix = re.split(r"[-\d]", parts[2], maxsplit=1)[0]
    return _DOC_TYPES.get(prefix)


def extract_doc_number(first_page_text: str, filename: str) -> str:
    """Ưu tiên 'Số: ...' ở phần đầu trang 1 -> số hiệu đầu tiên -> tên file."""
    head = unicodedata.normalize("NFC", first_page_text)[:2500]
    m = _HEADER_NUM_RE.search(head)
    if m:
        return normalize_doc_number(m.group(1))
    m = DOC_NUM_RE.search(head)
    if m:
        return normalize_doc_number(m.group(1))
    # Fallback từ tên file: "Nghi dinh 214_2025_ND-CP PDF.pdf"
    stem = Path(filename).stem.upper().replace("_", "/")
    m = DOC_NUM_RE.search(stem)
    if m:
        return normalize_doc_number(m.group(1))
    return re.sub(r"\W+", "-", Path(filename).stem).strip("-") or "UNKNOWN"


_TITLE_RE = re.compile(
    r"(NGHỊ ĐỊNH|THÔNG TƯ|LUẬT|QUYẾT ĐỊNH|NGHỊ QUYẾT)\s+(.{5,300}?)\s+(?=Căn cứ|CHÍNH PHỦ|THỦ TƯỚNG|BỘ TRƯỞNG|QUỐC HỘI)"
)


def extract_title(first_page_text: str) -> Optional[str]:
    flat = " ".join(unicodedata.normalize("NFC", first_page_text)[:2500].split())
    m = _TITLE_RE.search(flat)
    return f"{m.group(1).title()} {m.group(2).strip()}" if m else None


# ---------------------------------------------------------------------------
# Quan hệ pháp lý
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Relation:
    type: str      # "amends" | "repeals" | "replaces"
    target: str    # số hiệu văn bản bị tác động
    evidence: str  # đoạn câu làm căn cứ
    scope: str = ""  # "partial" (sửa đổi/bổ sung một phần) | "partial_repeal" (bãi bỏ một số điều khoản) | "full" (thay thế/bãi bỏ toàn bộ)


_DOC_WORDS = r"(?i:Nghị định|Thông tư(?:\s+liên tịch)?|Luật|Quyết định|Nghị quyết|Pháp lệnh)"

# "Sửa đổi, bổ sung một số điều của Nghị định số 214/2025/NĐ-CP"
_AMEND_RE = re.compile(
    rf"(?P<trig>(?i:sửa đổi(?:,?\s*(?:và\s*)?bổ sung)?))(?P<between>[^.;]{{0,250}}?)"
    rf"{_DOC_WORDS}\s+(?:số\s+)?(?P<num>{NUM_BODY})"
)
_CHAPTER_AMEND_RE = re.compile(
    rf"(?i:chương\s+[ivxlcdm]+)[^.;\n]{{0,100}}?(?i:sửa đổi)[^.;\n]{{0,150}}?{_DOC_WORDS}\s+(?:số\s+)?(?P<num>{NUM_BODY})"
)
# "thay thế Nghị định số ..."
_REPLACE_RE = re.compile(
    rf"(?i:thay thế)\s+(?:(?i:các)\s+)?{_DOC_WORDS}\s+(?:số\s+)?(?P<num>{NUM_BODY})"
)
# "bãi bỏ [các] Nghị định sau/số ..." (bãi bỏ toàn bộ văn bản)
_REPEAL_FULL_RE = re.compile(
    rf"(?i:bãi bỏ)\s+(?:(?i:toàn bộ|các)\s+)*(?:{_DOC_WORDS}|(?i:văn bản))"
)
# "bãi bỏ khoản 3 Điều 5 của Nghị định số ..." (bãi bỏ một phần -> coi là sửa đổi)
_REPEAL_PART_RE = re.compile(
    rf"(?i:bãi bỏ)[^.;]{{0,250}}?(?i:của)\s+{_DOC_WORDS}\s+(?:số\s+)?(?P<num>{NUM_BODY})"
)
# Ranh giới câu / điều: ". Chữ hoa" (không tính "1. Nghị định") hoặc đầu "Điều N."
_CLAUSE_END_RE = re.compile(r"(?<!\d)\.\s+(?=[A-ZĐ])|\s(?=Điều\s+\d+\.\s)")


def _evidence(text: str, pos: int, length: int = 220) -> str:
    return text[pos : pos + length].strip()


def _clause_from(text: str, start: int, max_len: int = 800) -> str:
    window = text[start : start + max_len]
    m = _CLAUSE_END_RE.search(window)
    return window[: m.start()] if m else window


def extract_relations(text: str, own_doc_number: str) -> list[Relation]:
    """Quét các mẫu câu: sửa đổi/bổ sung, thay thế, bãi bỏ văn bản khác."""
    flat = unicodedata.normalize("NFC", re.sub(r"\s+", " ", text))
    found: dict[tuple[str, str], tuple[str, str]] = {}

    def add(rel_type: str, raw_num: str, pos: int, scope: str) -> None:
        target = normalize_doc_number(raw_num)
        if target == own_doc_number:
            return
        found.setdefault((rel_type, target), (_evidence(flat, pos), scope))
        
    for m in _CHAPTER_AMEND_RE.finditer(flat):
        add("amends", m.group("num"), m.start(), "partial")
        # Fallback khi OCR nhận sai dấu trong tiêu đề chương.
            # Fallback OCR: xử lý tiêu đề chương bị OCR sai hoặc mất chữ "Chương".
    def _fold_relation_text(value: str) -> str:
        normalized = unicodedata.normalize("NFD", value).casefold()
        normalized = "".join(
            ch for ch in normalized
            if unicodedata.category(ch) != "Mn"
        )
        return normalized.replace("đ", "d")

    number_pattern = (
        r"\d{1,4}\s*/\s*\d{4}\s*/\s*[a-z]{1,6}\d{0,2}"
        r"(?:-[a-z]{1,8})*"
    )
    relation_pattern = re.compile(
        rf"\bnghi\s*dinh\s*so\s*[-:]?\s*({number_pattern})",
        re.IGNORECASE,
    )

    def _add_from_heading(heading: str, evidence_pos: int) -> None:
        normalized = _fold_relation_text(heading)

        # Xét từng cụm "sửa đổi", không để một lần khớp
        # nuốt mất cụm sửa đổi tiếp theo.
        for trigger in re.finditer(r"\bsua\s+doi\b", normalized):
            window = normalized[trigger.end():trigger.end() + 700]
            target_match = relation_pattern.search(window)

            if not target_match:
                continue

            between = window[:target_match.start()]
            # Không lấy văn bản chỉ được nhắc trong lịch sử sửa đổi.
            if re.search(r"\bboi\b", between):
                continue

            add(
                "amends",
                target_match.group(1),
                evidence_pos,
                "partial",
            )
            break

    # Chương I có thể bị OCR thành một dòng gạch ngang,
    # vì vậy kiểm tra phần đầu trước Điều 1.
    first_article = re.search(r"(?i)\bđiều\s+1\s*[.:]", flat)
    prelude_end = first_article.start() if first_article else min(len(flat), 5000)
    prelude = flat[:prelude_end]

    if prelude:
        _add_from_heading(
            prelude,
            max(0, prelude_end - 450),
        )

    # Chỉ chấp nhận ký tự giống số La Mã để tránh nhầm "chương trình".
    chapter_matches = list(
        re.finditer(r"(?i)\bchương\s+([ivxlcdmïh]+)(?=\s|$)", flat)
    )

    for chapter in chapter_matches:
        rest = flat[chapter.end():]
        next_article = re.search(
            r"(?i)\bđiều\s+\d+[a-z]?\s*[.:]",
            rest,
        )

        heading_end = (
            chapter.end() + next_article.start()
            if next_article
            else min(len(flat), chapter.start() + 1800)
        )
        heading = flat[chapter.start():heading_end]

        _add_from_heading(heading, chapter.start())
    # 1) Sửa đổi, bổ sung
    for m in _AMEND_RE.finditer(flat):
        prev = flat[max(0, m.start() - 20) : m.start()].lower()
        if "được" in prev or "bị" in prev:  # "đã được sửa đổi, bổ sung bởi Luật ..." -> bỏ
            continue
        if m.group("between").strip().lower().startswith(("bởi", "theo", "bằng")):
            continue
        add("amends", m.group("num"), m.start(), "partial")

    # 2) Thay thế
    for m in _REPLACE_RE.finditer(flat):
        add("replaces", m.group("num"), m.start(), "full")

    # 3) Bãi bỏ toàn bộ: lấy mọi số hiệu trong mệnh đề (có thể liệt kê nhiều văn bản)
    full_starts: set[int] = set()
    for m in _REPEAL_FULL_RE.finditer(flat):
        full_starts.add(m.start())
        clause = _clause_from(flat, m.start())
        for num in DOC_NUM_RE.findall(clause):
            add("repeals", num, m.start(), "full")

    # 4) Bãi bỏ một phần -> amends
    for m in _REPEAL_PART_RE.finditer(flat):
        if m.start() in full_starts:
            continue
        add("amends", m.group("num"), m.start(), "partial_repeal")

    return [Relation(t, target, ev, scope) for (t, target), (ev, scope) in found.items()]


# ---------------------------------------------------------------------------
# Đọc file & chia chunk
# ---------------------------------------------------------------------------
@dataclass
class PageText:
    page: Optional[int]
    text: str


@dataclass
class ParsedDocument:
    filename: str
    doc_number: str
    doc_type: Optional[str]
    title: Optional[str]
    pages: list[PageText]
    relations: list[Relation] = field(default_factory=list)
    total_pages: Optional[int] = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


@dataclass
class Chunk:
    index: int
    content: str
    page: Optional[int]
    article: Optional[str]


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# OCR cho PDF scan: Tesseract (local, không tốn quota) hoặc model vision (theo lô nhiều trang / lượt gọi)
# ---------------------------------------------------------------------------
_SCAN_TEXT_THRESHOLD = 80   # trang có ảnh và < 80 ký tự text (vd. chỉ số trang) được coi là trang scan
_PAGE_MARK = re.compile(r"={2,}\s*TRANG\s+(\d+)\s*={2,}", re.I)
_OCR_BATCH_PROMPT = (
    "Dưới đây là {n} ảnh trang của một văn bản pháp luật tiếng Việt (theo thứ tự: {labels}). "
    "Hãy trích xuất TOÀN BỘ chữ trong từng ảnh, giữ nguyên số điều/khoản/điểm và bảng biểu; "
    "không tóm tắt, không bình luận. Với MỖI ảnh, in một dòng `===TRANG <số>===` (đúng số trang đã nêu) "
    "rồi tới nội dung của trang đó."
)


def _settings():
    from src.config import get_settings
    return get_settings()


def _file_key(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _cache_file(key: str, engine: str, page: int) -> Path:
    return _settings().ocr_cache_dir / key / engine / f"{page}.txt"


def _cache_get(key: str, engine: str, page: int) -> str:
    # Bản OCR bằng AI chính xác hơn Tesseract nên cũng được dùng lại khi đang chọn Tesseract.
    for eng in dict.fromkeys([engine, "llm"]):
        f = _cache_file(key, eng, page)
        if f.exists():
            text = f.read_text(encoding="utf-8").strip()
            if text:
                return text
    return ""


def _cache_put(key: str, engine: str, page: int, text: str, conf: Optional[float] = None) -> None:
    f = _cache_file(key, engine, page)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")
    if conf is not None:
        f.with_suffix(".conf").write_text(str(conf), encoding="utf-8")


def _cache_conf(key: str, engine: str, page: int) -> Optional[float]:
    for eng in dict.fromkeys([engine, "llm"]):
        f = _cache_file(key, eng, page).with_suffix(".conf")
        if f.exists():
            try:
                return float(f.read_text(encoding="utf-8"))
            except ValueError:
                return None
    return None


_WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
)


def _resolve_tesseract_cmd() -> str:
    """TESSERACT_CMD trong .env > tesseract trong PATH > đường dẫn cài đặt mặc định trên Windows."""
    configured = _settings().tesseract_cmd
    if configured and (os.path.isfile(configured) or shutil.which(configured)):
        return configured
    if shutil.which("tesseract"):
        return "tesseract"
    for candidate in _WINDOWS_TESSERACT_PATHS:
        if os.path.isfile(candidate):
            return candidate
    return configured if configured else ""


@lru_cache(maxsize=1)
def _tesseract_check() -> tuple[bool, str]:
    """(sẵn sàng?, mô tả) — mô tả giải thích chính xác vì sao không dùng được Tesseract."""
    s = _settings()
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError:
        return False, "chưa cài thư viện: chạy `python -m pip install pytesseract pillow`"
    cmd = _resolve_tesseract_cmd()
    if not cmd:
        return False, "không tìm thấy tesseract.exe: hãy cài Tesseract-OCR hoặc đặt TESSERACT_CMD trong .env"
    pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as exc:  # noqa: BLE001
        return False, f"không chạy được Tesseract tại '{cmd}': {str(exc)[:120]}"
    try:
        langs = set(pytesseract.get_languages(config=""))
    except Exception as exc:  # noqa: BLE001
        return False, f"không đọc được danh sách ngôn ngữ: {str(exc)[:120]}"
    missing = set(s.ocr_lang.split("+")) - langs
    if missing:
        return False, (
            f"thiếu dữ liệu ngôn ngữ {sorted(missing)} (đang có {sorted(langs)}): "
            "tải vie.traineddata bỏ vào thư mục tessdata của Tesseract hoặc đặt TESSDATA_PREFIX"
        )
    return True, f"Tesseract {version} tại '{cmd}', ngôn ngữ {s.ocr_lang}"


def _tesseract_ready() -> bool:
    return _tesseract_check()[0]


def ocr_status() -> dict[str, str]:
    """Trạng thái OCR (hiển thị ở /api/health để tự chẩn đoán)."""
    s = _settings()
    ok, detail = _tesseract_check()
    if s.ocr_engine == "llm":
        engine = "llm"
    elif s.ocr_engine == "tesseract":
        engine = "tesseract" if ok else "lỗi (Tesseract chưa sẵn sàng)"
    else:
        engine = "tesseract" if ok else "llm (tự chuyển vì Tesseract chưa sẵn sàng — tốn quota AI)"
    return {"configured": s.ocr_engine, "engine_in_use": engine, "tesseract": detail}


def _pick_engine() -> str:
    s = _settings()
    if s.ocr_engine == "tesseract":
        ok, detail = _tesseract_check()
        if not ok:
            raise ParseError(f"OCR_ENGINE=tesseract nhưng Tesseract chưa dùng được: {detail}.")
        return "tesseract"
    if s.ocr_engine == "llm":
        return "llm"
    ok, detail = _tesseract_check()  # auto: ưu tiên chạy local (miễn phí, không giới hạn)
    if not ok:
        logger.warning("OCR_ENGINE=auto: không dùng được Tesseract (%s) -> OCR bằng AI, sẽ tốn quota", detail)
    return "tesseract" if ok else "llm"


def _render_png(page, dpi: int, page_num: int) -> bytes:
    try:
        return page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72)).tobytes("png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không render được trang %d để OCR: %s", page_num, exc)
        return b""


def _ocr_tesseract_png(png: bytes) -> tuple[str, Optional[float]]:
    """OCR một trang bằng Tesseract. Trả về (văn bản, độ tin cậy trung bình 0-100 của các từ nhận dạng được)."""
    import pytesseract
    from PIL import Image
    s = _settings()
    pytesseract.pytesseract.tesseract_cmd = _resolve_tesseract_cmd() or "tesseract"
    with Image.open(io.BytesIO(png)) as im:
        data = pytesseract.image_to_data(im, lang=s.ocr_lang, config="--oem 1 --psm 6", output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    order: list[tuple[int, int, int]] = []
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(word)
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= 0:
            confs.append(conf)
    out: list[str] = []
    prev: Optional[tuple[int, int, int]] = None
    for key in order:
        if prev is not None and key[:2] != prev[:2]:
            out.append("")  # sang đoạn mới
        out.append(" ".join(lines[key]))
        prev = key
    return "\n".join(out).strip(), (round(sum(confs) / len(confs), 1) if confs else None)


def _split_pages(text: str, expected: list[int]) -> dict[int, str]:
    parts = _PAGE_MARK.split(text)
    out: dict[int, str] = {}
    for i in range(1, len(parts) - 1, 2):
        n, body = int(parts[i]), parts[i + 1].strip()
        if n in expected and body:
            out[n] = body
    if not out and len(expected) == 1 and text.strip():
        out[expected[0]] = text.strip()
    return out


def _ocr_llm_batch(images: dict[int, bytes]) -> dict[int, str]:
    """OCR nhiều trang trong MỘT lượt gọi (tiết kiệm số request / hạn mức theo ngày)."""
    from src.services import llm_router
    s = _settings()
    nums = sorted(images)
    content: list[dict] = [{"type": "text", "text": _OCR_BATCH_PROMPT.format(n=len(nums), labels=", ".join(f"TRANG {n}" for n in nums))}]
    for n in nums:
        b64 = base64.b64encode(images[n]).decode("utf-8")
        content.append({"type": "text", "text": f"Ảnh của TRANG {n}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    models = [s.ocr_model or s.generation_model, *s.generation_fallback_models]
    resp, used = llm_router.complete(
        models, messages=[{"role": "user", "content": content}],
        max_tokens=min(16000, 2500 * len(nums)), timeout=180,
    )
    text = (resp.choices[0].message.content or "").strip()
    pages = _split_pages(text, nums)
    logger.info("OCR trang %s via %s: nhận %d/%d trang", nums, used, len(pages), len(nums))
    return pages


def _ocr_pages(path: Path, doc, needed: list[int]) -> tuple[dict[int, str], list[int], str, str, dict[int, Optional[float]]]:
    """OCR các trang scan. Trả về (kết quả theo trang, trang thất bại, lý do dừng)."""
    from src.services import llm_router
    s = _settings()
    key = _file_key(path)

    # Ưu tiên kiểm tra bộ nhớ đệm đã có sẵn trước khi đòi hỏi cài đặt OCR engine
    for candidate_engine in ("tesseract", "llm"):
        cached_results: dict[int, str] = {}
        cached_confs: dict[int, Optional[float]] = {}
        for n in needed:
            c = _cache_get(key, candidate_engine, n)
            if c:
                cached_results[n] = c
                cached_confs[n] = _cache_conf(key, candidate_engine, n)
        if len(cached_results) == len(needed):
            logger.info("OCR '%s': dùng lại toàn bộ %d trang scan từ bộ nhớ đệm (%s)", path.name, len(needed), candidate_engine)
            return cached_results, [], "", candidate_engine, cached_confs

    engine = _pick_engine()

    results: dict[int, str] = {}
    confs: dict[int, Optional[float]] = {}
    todo: list[int] = []
    for n in needed:
        cached = _cache_get(key, engine, n)
        if cached:
            results[n] = cached
            confs[n] = _cache_conf(key, engine, n)
        else:
            todo.append(n)
    if results:
        logger.info("OCR '%s': dùng lại %d trang đã có trong bộ nhớ đệm, còn %d trang", path.name, len(results), len(todo))
    abort = ""
    if not todo:
        return results, [], abort, engine, confs

    logger.info("OCR '%s' bằng %s: %d trang", path.name, engine, len(todo))
    if engine == "tesseract":
        wave = max(1, s.ocr_workers) * 2
        with ThreadPoolExecutor(max_workers=max(1, s.ocr_workers)) as pool:
            for i in range(0, len(todo), wave):
                group = todo[i : i + wave]
                pngs = {n: _render_png(doc[n - 1], 300, n) for n in group}  # render tuần tự (pymupdf không thread-safe)
                futs = {n: pool.submit(_ocr_tesseract_png, png) for n, png in pngs.items() if png}
                for n, fut in futs.items():
                    try:
                        text, conf = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Tesseract lỗi trang %d: %s", n, exc)
                        continue
                    if text:
                        results[n] = text
                        confs[n] = conf
                        _cache_put(key, engine, n, text, conf)
                logger.info("OCR '%s': xong %d/%d trang", path.name, min(i + wave, len(todo)), len(todo))
    else:
        models = [s.ocr_model or s.generation_model, *s.generation_fallback_models]
        per = max(1, s.ocr_pages_per_request)
        batches = [todo[i : i + per] for i in range(0, len(todo), per)]
        with ThreadPoolExecutor(max_workers=max(1, s.ocr_workers)) as pool:
            for w in range(0, len(batches), max(1, s.ocr_workers)):
                if not llm_router.available(models):
                    abort = "mọi model OCR đều đang hết quota/quá tải"
                    break
                wave = batches[w : w + max(1, s.ocr_workers)]
                jobs = []
                for batch in wave:
                    imgs = {n: _render_png(doc[n - 1], s.ocr_dpi, n) for n in batch}
                    jobs.append((batch, pool.submit(_ocr_llm_batch, {n: b for n, b in imgs.items() if b})))
                for batch, fut in jobs:
                    try:
                        got = fut.result()
                    except llm_router.AllModelsUnavailable as exc:
                        abort = str(exc)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("OCR lô %s lỗi: %s", batch, exc)
                        continue
                    for n, text in got.items():
                        results[n] = text
                        _cache_put(key, engine, n, text)
                if abort:
                    break
    failed = [n for n in todo if n not in results]
    return results, failed, abort, engine, confs


_GARBLE_SYMBOLS = re.compile(r"[|\\~^_`{}<>]")
_LEGIT_MIXED = re.compile(r"^(?:\d+[a-zđ]?|[a-zđ]\d?|\d+[./\-]\S*|\S*\d+/\S*|\(?[a-zđ0-9]\)|\d+(?:\.\d+)*[a-zđ%]*)$", re.I)


def garble_ratio(text: str) -> float:
    """Tỉ lệ 'từ' có dấu hiệu OCR hỏng (lẫn chữ-số bất thường, ký tự lạ). 0 = sạch, > 0.08 đáng nghi."""
    tokens = text.split()
    if not tokens:
        return 1.0
    bad = 0
    for tok in tokens:
        core = tok.strip(".,;:()[]\"'“”")
        if not core:
            continue
        if _GARBLE_SYMBOLS.search(core):
            bad += 1
        elif len(core) >= 3 and any(c.isalpha() for c in core) and any(c.isdigit() for c in core) and not _LEGIT_MIXED.match(core):
            bad += 1
    return round(bad / len(tokens), 3)


def normalize_article_number(raw: str) -> str:
    """Chuẩn hoá số điều và sửa lỗi OCR phổ biến: 4ó -> 46, §4 -> 84, l/I -> 1."""
    s = raw.strip()
    s = s.replace("ó", "6").replace("§", "8")
    s = re.sub(r"(?<=\d)[lI]", "1", s)
    s = re.sub(r"^[lI](?=\d)", "1", s)
    return s


# Nhận diện tiêu đề Điều: chịu được ký tự rác OCR ở đầu dòng (-, _, l, |), lỗi không dấu (Dieu), dấu ngăn cách (., :, ,, -, space)
_ARTICLE_HEADING_RE = re.compile(
    r"(?mi)^[\s\-_—|l1I*~`]*((?:Đi[ềêèe]u|ĐI[ỀÊÈE]U)\s+([\dó§a-z]+))\s*[,.:\s\-—]"
)


def article_coverage(text: str) -> dict[str, object]:
    """Các 'Điều N' tìm thấy và những số bị thiếu trong dãy 1..max (dấu hiệu OCR/ngắt trang làm mất tiêu đề hoặc mất trang)."""
    found_nums: set[int] = set()
    for m in _ARTICLE_HEADING_RE.finditer(text):
        num_str = normalize_article_number(m.group(2))
        if num_str.isdigit():
            found_nums.add(int(num_str))
    found = sorted(found_nums)
    if not found:
        return {"found": 0, "max": None, "missing": []}
    missing = sorted(set(range(1, found[-1] + 1)) - set(found))
    return {"found": len(found), "max": found[-1], "missing": missing[:60], "missing_count": len(missing)}


def _read_pdf(path: Path) -> tuple[list[PageText], dict]:
    """Trích text; trang scan (không có/ rất ít text nhưng có ảnh) được OCR. Hỗ trợ PDF scan một phần."""
    pages: list[PageText] = []
    s = _settings()
    diag: dict = {"engine": "text", "ocr_pages": [], "pages": {}}
    try:
        with fitz.open(path) as doc:
            diag["pages_total"] = len(doc)
            ocr_needed: list[int] = []
            for i, page in enumerate(doc, start=1):
                text = _clean(page.get_text("text"))
                is_scan = page.get_images(full=True) and len(text) < _SCAN_TEXT_THRESHOLD
                if is_scan:
                    ocr_needed.append(i)
                elif text:
                    pages.append(PageText(i, text))

            if ocr_needed:
                results, failed, abort, used_engine, confs = _ocr_pages(path, doc, ocr_needed)
                diag["engine"] = used_engine if len(ocr_needed) == diag["pages_total"] else f"text+{used_engine}"
                diag["ocr_pages"] = sorted(ocr_needed)
                diag["failed_pages"] = sorted(failed)
                for n in results:
                    diag["pages"][n] = {"source": "ocr", "conf": confs.get(n)}
                for n, text in results.items():
                    cleaned = _clean(text)
                    if cleaned:
                        pages.append(PageText(n, cleaned))
                if failed:
                    st = ocr_status()
                    msg = (
                        f"OCR chưa hoàn tất '{path.name}' (engine: {st['engine_in_use']}; Tesseract: {st['tesseract']}): "
                        f"xong {len(results)}/{len(ocr_needed)} trang scan, "
                        f"còn thiếu {len(failed)} trang (vd. {failed[:8]}). "
                        + (f"Lý do: {abort[:200]}. " if abort else "")
                        + "Kết quả đã lưu (storage/ocr_cache): tải lại file sau khi quota được làm mới, "
                        "hoặc dùng OCR_ENGINE=tesseract (chạy local, không tốn quota) — hệ thống chỉ OCR phần còn thiếu."
                    )
                    if not s.ocr_allow_partial:
                        raise OCRIncomplete(msg)
                    logger.warning("%s (OCR_ALLOW_PARTIAL=true -> vẫn nạp phần đã đọc được)", msg)
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Không đọc được PDF '{path.name}': {exc}") from exc

    pages.sort(key=lambda p: p.page or 0)  # giữ đúng thứ tự trang sau khi OCR
    for pg in pages:
        info = diag["pages"].setdefault(pg.page, {"source": "text", "conf": None})
        info["chars"] = len(pg.text)
        info["garble"] = garble_ratio(pg.text)
    diag["empty_pages"] = [n for n in range(1, (diag.get("pages_total") or 0) + 1) if n not in {p.page for p in pages}]
    min_conf = _settings().ocr_min_confidence
    diag["low_conf_pages"] = sorted(n for n, i in diag["pages"].items() if i.get("conf") is not None and i["conf"] < min_conf)
    diag["garbled_pages"] = sorted(n for n, i in diag["pages"].items() if i.get("garble", 0) > 0.08)
    return pages, diag


def _read_docx(path: Path) -> list[PageText]:
    try:
        doc = DocxDocument(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Không đọc được DOCX '{path.name}': {exc}") from exc
    text = _clean("\n".join(parts))
    return [PageText(None, text)] if text else []


def _number_from_filename(filename: str) -> Optional[str]:
    m = DOC_NUM_RE.search(Path(filename).stem.upper().replace("_", "/"))
    return normalize_doc_number(m.group(1)) if m else None


def _fix_with_filename(doc_number: str, filename: str) -> str:
    """Số hiệu trích từ trang scan có thể sai ký tự (NĐ -> NB). Nếu tên file có cùng số + năm thì tin tên file."""
    fn = _number_from_filename(filename)
    if fn and fn != doc_number:
        a, b = fn.split("/"), doc_number.split("/")
        if len(a) == 3 and len(b) == 3 and a[0] == b[0] and a[1] == b[1]:
            logger.warning("Số hiệu '%s' (từ nội dung) khác tên file -> dùng '%s'", doc_number, fn)
            return fn
    return doc_number


def _pdf_page_count(path: Path) -> Optional[int]:
    if path.suffix.lower() != ".pdf":
        return None
    try:
        with fitz.open(path) as doc:
            return len(doc)
    except Exception:  # noqa: BLE001
        return None


def parse_file(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    diag: dict = {}
    if suffix == ".pdf":
        pages, diag = _read_pdf(path)
    elif suffix == ".docx":
        pages = _read_docx(path)
    else:
        raise ParseError(f"Định dạng không được hỗ trợ: {suffix}")

    if not pages:
        raise ParseError(f"'{path.name}' không có văn bản trích xuất được (có thể là PDF scan, cần OCR).")

    first = pages[0].text
    doc_number = _fix_with_filename(extract_doc_number(first, path.name), path.name)
    parsed = ParsedDocument(
        filename=path.name,
        doc_number=doc_number,
        doc_type=doc_type_of(doc_number),
        title=extract_title(first),
        pages=pages,
        total_pages=diag.get("pages_total") or _pdf_page_count(path),
        diagnostics=diag,
    )
    parsed.relations = extract_relations(parsed.full_text, doc_number)
    parsed.diagnostics["articles"] = article_coverage(parsed.full_text)
    logger.info(
        "Parsed %s | doc_number=%s | pages=%d | relations=%s",
        path.name, doc_number, len(pages), [(r.type, r.target) for r in parsed.relations],
    )
    return parsed


_ARTICLE_RE = _ARTICLE_HEADING_RE


def chunk_document(parsed: ParsedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chia theo trang, ưu tiên ngắt tại 'Điều N'; gán nhãn Điều cho từng chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\nĐiều ", "\n\n", "\n", ". ", "; ", " ", ""],
        keep_separator=True,
    )
    chunks: list[Chunk] = []
    current_article: Optional[str] = None
    for page in parsed.pages:
        for piece in splitter.split_text(page.text):
            content = piece.strip()
            if not content:
                continue
            matches = list(_ARTICLE_RE.finditer(content))
            if matches:
                first_num = normalize_article_number(matches[0].group(2))
                art_label = f"Điều {first_num}"
                label = art_label if (matches[0].start() < 30 or current_article is None) else current_article
                last_num = normalize_article_number(matches[-1].group(2))
                current_article = f"Điều {last_num}"
            else:
                label = current_article
            chunks.append(Chunk(index=len(chunks), content=content, page=page.page, article=label))
    return chunks
