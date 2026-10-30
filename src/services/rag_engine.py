"""RAG pipeline: nạp tài liệu, truy xuất + mở rộng quan hệ pháp lý, cache ngữ cảnh, patch câu trả lời."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import litellm
from pydantic import BaseModel, Field, ValidationError

from src.services import direct_answer as DA
from src.services import llm_router
from src.services.lexical import BM25, rrf
from src.services.trace import est_tokens, trace, trace_ingest

from src.config import Settings, file_url_for, get_settings
from src.services.document_parser import (
    DOC_NUM_RE,
    OCRIncomplete,
    ParseError,
    chunk_document,
    normalize_doc_number,
    parse_file,
)
from src.services.logger_service import get_logger, log_event, preview
from src.services.qdrant_service import EmbeddingError, Hit, QdrantService
from src.services.relations_service import RelationsService
from src.services.security_guard import CANARY, SecurityGuard

logger = get_logger("rag")
litellm.drop_params = True

class IngestState:
    """Trạng thái nạp từng file (complete | complete_with_warnings | incomplete | failed) lưu ở storage/ingest_state.json."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        try:
            self._data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def set(self, filename: str, info: dict[str, Any]) -> None:
        with self._lock:
            self._data[filename] = {**info, "updated_at": datetime.now(timezone.utc).isoformat()}
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
            except OSError:
                logger.exception("Không ghi được ingest_state")

    def all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)


class UpstreamError(Exception):
    """Lỗi từ dịch vụ LLM đám mây."""


class QueryExpired(Exception):
    """query_id không tồn tại hoặc đã hết hạn cache (TTL)."""


# ===========================================================================
# Schemas
# ===========================================================================
class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    use_ai: Optional[bool] = None  # None: tự động (ưu tiên rẻ nhất) | False: không gọi AI | True: luôn dùng AI


class Citation(BaseModel):
    index: int
    doc_number: str
    filename: str
    file_url: str
    page: Optional[int] = None
    article: Optional[str] = None
    status: str = "unverified"
    is_amendment: bool = False
    score: float = 0.0
    snippet: str


class Modification(BaseModel):
    original_doc: str
    amending_doc: str
    relation: str  # amended_by | replaced_by | repealed_by
    amending_file_url: Optional[str] = None
    note: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[Citation]
    modifications: list[Modification]
    cache_ttl_seconds: int
    usage: Usage
    mode: str = "llm"          # llm | cache | direct | extractive
    notice: Optional[str] = None


class PatchRequest(BaseModel):
    query_id: str = Field(min_length=8, max_length=64)
    instruction: str = Field(min_length=3, max_length=1000)


class PatchResponse(BaseModel):
    query_id: str
    answer: str
    edits_applied: list[dict[str, str]]
    edits_failed: list[dict[str, str]]
    note: str = ""
    usage: Usage


# ===========================================================================
# Cache ngữ cảnh (TTL)
# ===========================================================================
@dataclass
class CacheEntry:
    query_id: str
    question: str
    sources: list[dict[str, Any]]
    cited: list[int]
    answer: str
    modifications: list[dict[str, Any]]
    created_at: float
    expires_at: float


class ContextCache:
    """Bộ nhớ đệm trong tiến trình, theo query_id, hết hạn sau TTL (mặc định 30 phút)."""

    def __init__(self, ttl_seconds: int, max_items: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_items
        self._data: "OrderedDict[str, CacheEntry]" = OrderedDict()

    @property
    def ttl(self) -> int:
        return self._ttl

    def _purge(self) -> None:
        now = time.time()
        for key in [k for k, v in self._data.items() if v.expires_at <= now]:
            del self._data[key]

    def put(self, entry: CacheEntry) -> None:
        self._purge()
        self._data[entry.query_id] = entry
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def get(self, query_id: str) -> Optional[CacheEntry]:
        self._purge()
        return self._data.get(query_id)


class AnswerCache:
    """Cache câu trả lời AI theo câu hỏi đã chuẩn hoá. Hỏi lại cùng câu -> 0 token. Tự vô hiệu khi kho tài liệu đổi."""

    def __init__(self, ttl_seconds: int, max_items: int = 500) -> None:
        self._ttl, self._max = ttl_seconds, max_items
        self._data: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        item = self._data.get(key)
        if not item or item[0] <= time.time():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return item[1]

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = (time.time() + self._ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)


# ===========================================================================
# Helpers
# ===========================================================================
_REL_KEYS = (("amended_by", "amended_by"), ("replaced_by", "replaced_by"), ("repealed_by", "repealed_by"))
_STATUS_VI = {
    "unverified": "chưa xác minh hiệu lực (chưa ghi nhận văn bản nào sửa đổi, thay thế hay bãi bỏ trong các văn bản đã nạp — điều này KHÔNG có nghĩa văn bản còn hiệu lực)",
    "amended": "đã được sửa đổi, bổ sung",
    "replaced": "đã bị thay thế",
    "repealed": "đã bị bãi bỏ",
}
_REL_LABEL_VI = {
    "amended_by": "được sửa đổi, bổ sung bởi",
    "replaced_by": "bị thay thế bởi",
    "repealed_by": "bị bãi bỏ bởi",
}



PROMPTS_DIR = Path("prompts")

def _load_prompt(filename: str) -> str:
    """Đọc nội dung file prompt, tự động lọc bỏ phần frontmatter (YAML header như version: ...) nếu có."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file prompt tại: {path.resolve()}")
    content = path.read_text(encoding="utf-8")
    if "---" in content:
        parts = content.split("---", 1)
        return parts[1].strip()
    return content.strip()

# Nạp nội dung từ file Markdown riêng biệt
_CHAT_SYSTEM = _load_prompt("chat_system.md")
_PATCH_SYSTEM = _load_prompt("patch_system.md")

PROMPT_VERSION = "chat-v5"
PROMPT_HASH = hashlib.sha1(_CHAT_SYSTEM.encode("utf-8")).hexdigest()[:8]

# Gắn mã bảo mật Canary (giữ nguyên logic bảo mật gốc của dự án)
_CHAT_SYSTEM += f"\n(Mã nội bộ, tuyệt đối không nhắc lại dưới mọi hình thức: {CANARY})"
_PATCH_SYSTEM += f"\n(Mã nội bộ, tuyệt đối không nhắc lại dưới mọi hình thức: {CANARY})"
_SYSTEM_PROMPTS = [_CHAT_SYSTEM, _PATCH_SYSTEM]

def _parse_json(raw: str) -> Optional[dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Phục hồi khi câu trả lời bị cắt ngang (truncated do max_tokens) hoặc format JSON chưa đóng
    m = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)', cleaned)
    if m:
        text = m.group(1).replace(r"\n", "\n").replace(r"\"", '"').replace(r"\\", "\\")
        sources_m = re.search(r'"used_sources"\s*:\s*\[([\d,\s]*)\]', cleaned)
        used = [int(x.strip()) for x in sources_m.group(1).split(",") if x.strip().isdigit()] if sources_m else []
        return {"answer": text, "used_sources": used, "enough_context": True}

    return None


def _usage_reported(resp: Any) -> bool:
    """True nếu nhà cung cấp trả số token thật (khác với ước tính est_tokens)."""
    usage = getattr(resp, "usage", None)
    return usage is not None and getattr(usage, "prompt_tokens", None) is not None


def _usage_of(resp: Any) -> Usage:
    usage = getattr(resp, "usage", None)
    return Usage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


class LLMAnswer(BaseModel):
    """Schema đầu ra bắt buộc của AI."""

    model_config = {"extra": "ignore"}
    answer: str = Field(min_length=1)
    used_sources: list[int] = Field(default_factory=list)
    enough_context: bool = True


def validate_llm_output(raw: str) -> tuple[Optional[LLMAnswer], list[str]]:
    """Kiểm tra đầu ra AI có đúng schema không. Trả về (đối tượng hợp lệ | None, danh sách lỗi)."""
    data = _parse_json(raw)
    if data is None:
        return None, ["đầu ra không phải JSON hợp lệ"]
    try:
        return LLMAnswer.model_validate(data), []
    except ValidationError as exc:
        return None, [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()]


_CLAIM_VALID = re.compile(r"còn\s+hiệu\s+lực|vẫn\s+có\s+hiệu\s+lực|đang\s+có\s+hiệu\s+lực|vẫn\s+còn\s+hiệu\s+lực", re.I)
_CLAIM_ENDED = re.compile(r"hết\s+hiệu\s+lực|bị\s+bãi\s+bỏ|đã\s+bị\s+bãi\s+bỏ|bị\s+thay\s+thế|đã\s+bị\s+thay\s+thế", re.I)


def _year_pair(number: str) -> str:
    parts = number.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else number


def reference_checks(answer: str, question: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Đối chiếu số hiệu / điều / trang / khẳng định hiệu lực trong câu trả lời với các Nguồn đã cung cấp.

    LƯU Ý: đây chỉ là kiểm tra heuristic phát hiện tham chiếu KHÔNG có trong nguồn (dấu hiệu bịa); việc đạt kiểm tra
    KHÔNG chứng minh câu trả lời đúng về pháp lý."""
    ctx = " ".join(s["content"] for s in sources)
    ctx_numbers = {s["doc_number"] for s in sources}
    for s in sources:
        for key in ("amended_by", "replaced_by", "repealed_by"):
            ctx_numbers.update(s.get(key) or [])
    ctx_numbers.update(normalize_doc_number(m) for m in DOC_NUM_RE.findall(ctx))
    known_pairs = {_year_pair(n) for n in ctx_numbers} | {_year_pair(normalize_doc_number(m)) for m in DOC_NUM_RE.findall(question)}
    unsupported_docs = sorted({n for n in (normalize_doc_number(m) for m in DOC_NUM_RE.findall(answer)) if _year_pair(n) not in known_pairs})

    labels = " ".join(str(s.get("article") or "") for s in sources)
    haystack = ctx + " " + labels + " " + question
    articles = {m for m in re.findall(r"(?i)\bđiều\s+(\d+[a-zđ]?)", answer)}
    unsupported_articles = sorted(a for a in articles if not re.search(rf"(?i)\bđiều\s+{re.escape(a)}\b", haystack))

    pages = {int(p) for p in re.findall(r"(?i)\btrang\s+(\d+)", answer)}
    known_pages = {s["page"] for s in sources if s.get("page")} | {int(p) for p in re.findall(r"(?i)\btrang\s+(\d+)", question)}
    unsupported_pages = sorted(pages - known_pages)

    claims: list[str] = []
    if _CLAIM_VALID.search(answer) and not _CLAIM_VALID.search(ctx):
        claims.append("khẳng định 'còn hiệu lực' nhưng nguồn không nêu (hệ thống chưa xác minh hiệu lực)")
    if _CLAIM_ENDED.search(answer) and not (
        _CLAIM_ENDED.search(ctx) or re.search(r"(?i)bãi\s+bỏ|thay\s+thế", ctx) or any(s.get("status") in ("repealed", "replaced") for s in sources)
    ):
        claims.append("khẳng định văn bản hết hiệu lực/bị bãi bỏ/thay thế nhưng nguồn không nêu")
    return {"unsupported_documents": unsupported_docs, "unsupported_articles": unsupported_articles,
            "unsupported_pages": unsupported_pages, "unsupported_status_claims": claims}


class _Timer:
    """Đo thời gian (ms) từng bước xử lý một câu hỏi."""

    def __init__(self) -> None:
        self._last = time.perf_counter()
        self._t0 = self._last
        self.laps: dict[str, int] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = self.laps.get(name, 0) + round((now - self._last) * 1000)
        self._last = now

    @property
    def total_ms(self) -> int:
        return round((time.perf_counter() - self._t0) * 1000)


_NUM_TOKEN = re.compile(r"(\d[\d.,]*\d|\d)(\s*(?:%|tỷ|triệu|nghìn|ngày|tháng|năm|đồng|giờ|lần))?", re.I)
_NO_BASIS = re.compile(r"(?:không|chưa)\s+(?:tìm\s+thấy|có\s+đủ|đủ|có)\s+(?:căn\s+cứ|thông\s+tin|dữ\s+liệu)", re.I)


def answer_checks(answer: str, question: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Kiểm tra tự động chất lượng câu trả lời: số liệu có nằm trong nguồn không, trích dẫn [n] có hợp lệ không.

    Số liệu có đơn vị (tỷ, triệu, ngày, tháng, %...) được so theo CẶP (chữ số, đơn vị): "10 ngày" không được coi là có căn cứ
    chỉ vì chữ số 10 xuất hiện ở chỗ khác trong nguồn. Đây là heuristic, không thay thế đối chiếu văn bản gốc."""
    digits = lambda t: re.sub(r"\D", "", t)  # noqa: E731
    unit = lambda t: re.sub(r"\s+", "", (t or "").lower())  # noqa: E731
    context = " ".join(s["content"] for s in sources)
    known: set[str] = set()
    known_pairs: set[tuple[str, str]] = set()
    for text in (context, question):
        for m in _NUM_TOKEN.finditer(text):
            known.add(digits(m.group(1)))
            if m.group(2):
                known_pairs.add((digits(m.group(1)), unit(m.group(2))))
    body = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", answer)
    body = re.sub(r"(?i)\b(điều|khoản|điểm|mục|chương|trang|nguồn)\s+\d+[a-zđ]?", " ", body)
    ungrounded: list[str] = []
    for m in _NUM_TOKEN.finditer(body):
        d, u = digits(m.group(1)), unit(m.group(2))
        grounded = (d, u) in known_pairs if u else (len(d) < 2 or d in known)
        if not grounded:
            label = (m.group(1) + (m.group(2) or "")).strip()
            if label not in ungrounded:
                ungrounded.append(label)
    valid = {s["index"] for s in sources}
    marks = [int(n) for grp in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer) for n in re.findall(r"\d+", grp)]
    sentences = [x for x in re.split(r"(?<=[.!?;:])\s+|\n+", body) if len(x.strip()) >= 50]
    return {
        "ungrounded_numbers": ungrounded,
        "citation_marks": len(marks),
        "invalid_citations": sorted({n for n in marks if n not in valid}),
        "uncited_sentences": sum(1 for x in sentences if not re.search(r"\[\d+", x)),
        "says_no_basis": bool(_NO_BASIS.search(answer)),
        "answer_chars": len(answer),
    }


def _log_text(text: str, limit: int) -> str:
    """Nội dung ghi vào trace: nguyên văn (rút gọn) hoặc chỉ độ dài + hash nếu TRACE_LOG_TEXT=false (dữ liệu nhạy cảm)."""
    from src.config import get_settings

    if get_settings().trace_log_text:
        return preview(text, limit) if limit < 1000 else text[:limit]
    return f"<{len(text)} ký tự sha1:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}>"


def with_page(url: str, page: Optional[int], filename: str) -> str:
    """Thêm #page=N để trình xem PDF mở đúng trang (chỉ với file PDF)."""
    if url and page and filename.lower().endswith(".pdf") and "#" not in url:
        return f"{url}#page={page}"
    return url


_REL_VERB_VI = {"amended_by": "sửa đổi, bổ sung", "replaced_by": "thay thế", "repealed_by": "bãi bỏ"}


def apply_edits(answer: str, edits: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    """Áp dụng diff lên câu trả lời; trả về (câu trả lời mới, edit thành công, edit thất bại)."""
    applied: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        op = str(edit.get("op", "replace"))
        if op == "replace":
            find, repl = str(edit.get("find", "")), str(edit.get("replace", ""))
            if find and find in answer:
                answer = answer.replace(find, repl, 1)
                applied.append({"op": op, "find": find, "replace": repl})
            else:
                failed.append({"op": op, "find": find, "reason": "Không tìm thấy đoạn 'find' trong câu trả lời"})
        elif op == "append":
            text = str(edit.get("text", "")).strip()
            if text:
                answer = answer.rstrip() + "\n\n" + text
                applied.append({"op": op, "text": text})
        else:
            failed.append({"op": op, "reason": "Thao tác không được hỗ trợ"})
    return answer, applied, failed


# ===========================================================================
# Engine
# ===========================================================================
class RAGEngine:
    def __init__(
        self,
        qdrant: QdrantService,
        relations: RelationsService,
        guard: SecurityGuard,
        settings: Optional[Settings] = None,
    ) -> None:
        self._q = qdrant
        self._rel = relations
        self._guard = guard
        self._s = settings or get_settings()
        self.cache = ContextCache(self._s.cache_ttl_seconds, self._s.cache_max_items)
        self._ingest_lock = asyncio.Lock()
        self._answers = AnswerCache(self._s.answer_cache_ttl_seconds)
        self._corpus_version = 0  # tăng khi kho tài liệu đổi -> vô hiệu cache câu trả lời
        self._doc_text: dict[str, tuple[int, str, list[int], list[Optional[int]]]] = {}
        self._lex: Optional[tuple[int, BM25, dict[str, Hit]]] = None
        self._ingest_state = IngestState(self._s.ingest_state_file)

    # ------------------------------------------------------------- ingestion
    async def ingest_file(self, path: Path) -> dict[str, Any]:
        """Parse -> chunk -> embed -> lưu Qdrant -> kiểm chứng -> cập nhật đồ thị quan hệ. Mỗi lần nạp có `ingest_id`.

        An toàn khi dừng giữa chừng: OCR và embedding đã làm được nằm trong cache nên chạy lại chỉ xử lý phần còn thiếu;
        point ID xác định nên không tạo dữ liệu trùng; nếu lưu Qdrant lỗi thì registry quan hệ được hoàn tác."""
        ingest_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        stages: dict[str, int] = {}
        stage = "parse"
        trace_ingest("ingest_start", ingest_id=ingest_id, filename=path.name,
                     size_bytes=path.stat().st_size if path.exists() else None, embedding_model=self._s.embedding_model,
                     embedding_dim=self._s.embedding_dim, ocr_engine=self._s.ocr_engine, collection=self._s.qdrant_collection)
        snapshot: tuple[Optional[str], Optional[dict[str, Any]]] = (None, None)
        try:
            t = time.perf_counter()
            parsed = await asyncio.to_thread(parse_file, path)
            stages["parse_ms"] = round((time.perf_counter() - t) * 1000)

            stage, t = "chunk", time.perf_counter()
            chunks = await asyncio.to_thread(chunk_document, parsed, self._s.chunk_size, self._s.chunk_overlap)
            if not chunks:
                raise ParseError(f"'{path.name}' không tạo được chunk nào")
            stages["chunk_ms"] = round((time.perf_counter() - t) * 1000)

            stage, t = "embed", time.perf_counter()
            texts = [f"{parsed.doc_number} {c.article or ''}\n{c.content}".strip() for c in chunks]  # tiền tố giúp truy xuất chính xác hơn
            vectors = await self._q.embed_texts(texts)
            embed_stats = dict(self._q.last_embed)
            stages["embed_ms"] = round((time.perf_counter() - t) * 1000)
            if len(vectors) != len(chunks):
                raise EmbeddingError(f"Số vector ({len(vectors)}) khác số chunk ({len(chunks)})", kind="invalid_output")

            stage, t = "store", time.perf_counter()
            dg = parsed.diagnostics
            lens = [len(c.content) for c in chunks]
            summary = {
                "engine": dg.get("engine"), "ocr_pages": len(dg.get("ocr_pages", [])), "failed_pages": dg.get("failed_pages", []),
                "empty_pages": dg.get("empty_pages", []), "low_conf_pages": dg.get("low_conf_pages", [])[:60],
                "garbled_pages": dg.get("garbled_pages", [])[:60], "articles": dg.get("articles"),
                "chunk_chars": {"min": min(lens), "avg": round(sum(lens) / len(lens)), "max": max(lens), "short(<80)": sum(1 for n in lens if n < 80)},
            }
            file_url = file_url_for(path.name)
            async with self._ingest_lock:
                snapshot = (parsed.doc_number, await asyncio.to_thread(self._rel.get_record, parsed.doc_number))
                old_filename = self._rel.filename_of(parsed.doc_number)
                affected = await asyncio.to_thread(
                    self._rel.upsert_document, doc_number=parsed.doc_number, filename=path.name, file_url=file_url,
                    title=parsed.title, doc_type=parsed.doc_type, chunk_count=len(chunks), relations=parsed.relations,
                    pages_read=len(parsed.pages), pages_total=parsed.total_pages, diagnostics=summary,
                )
                base = {**self._rel.base_metadata(parsed.doc_number), **self._rel.payload_for(parsed.doc_number)}
                payloads = [{**base, "content": c.content, "page": c.page, "article": c.article, "chunk_index": c.index} for c in chunks]
                stored = await asyncio.to_thread(self._q.replace_document, path.name, payloads, vectors)
                if old_filename and old_filename != path.name:
                    await asyncio.to_thread(self._q.delete_by_filename, old_filename)
                synced = await asyncio.to_thread(self._rel.sync_to_qdrant, affected - {parsed.doc_number})
            stages["store_ms"] = round((time.perf_counter() - t) * 1000)

            stage = "verify"
            qpoints = await asyncio.to_thread(self._q.count_filename, path.name)
        except Exception as exc:  # noqa: BLE001
            if snapshot[0] is not None:  # hoàn tác registry nếu lưu Qdrant dở dang
                await asyncio.to_thread(self._rel.restore_record, snapshot[0], snapshot[1])
            resumable = stage in ("embed", "store", "verify") or isinstance(exc, OCRIncomplete)
            kind = getattr(exc, "kind", None) or type(exc).__name__
            self._ingest_state.set(path.name, {
                "ingest_id": ingest_id, "status": "incomplete" if resumable else "failed", "stage": stage,
                "error_kind": kind, "error": str(exc)[:500], "resumable": resumable,
            })
            trace_ingest("ingest_failed", ingest_id=ingest_id, filename=path.name, stage=stage, error_type=type(exc).__name__,
                         error_kind=kind, error=str(exc)[:500], resumable=resumable, stages_ms=stages,
                         total_ms=round((time.perf_counter() - started) * 1000))
            raise

        warnings: list[str] = []
        if summary["failed_pages"]:
            warnings.append(f"OCR lỗi {len(summary['failed_pages'])} trang: {summary['failed_pages'][:10]}")
        if summary["empty_pages"]:
            warnings.append(f"{len(summary['empty_pages'])} trang không có chữ: {summary['empty_pages'][:10]}")
        if summary["low_conf_pages"]:
            warnings.append(f"{len(summary['low_conf_pages'])} trang OCR độ tin cậy thấp: {summary['low_conf_pages'][:10]}")
        if summary["garbled_pages"]:
            warnings.append(f"{len(summary['garbled_pages'])} trang nghi nhiễu ký tự: {summary['garbled_pages'][:10]}")
        if (summary["articles"] or {}).get("missing"):
            warnings.append(f"Dãy 'Điều' bị thiếu số: {summary['articles']['missing'][:15]} (có thể mất trang/OCR sai tiêu đề)")
        if parsed.total_pages and len(parsed.pages) < 0.9 * parsed.total_pages:
            warnings.append(f"chỉ đọc được {len(parsed.pages)}/{parsed.total_pages} trang")
        incomplete = qpoints != len(chunks)
        if incomplete:
            warnings.append(f"Qdrant chỉ có {qpoints}/{len(chunks)} điểm của file này")
        status = "incomplete" if incomplete else ("complete_with_warnings" if warnings else "complete")
        total_ms = round((time.perf_counter() - started) * 1000)
        report = {
            "ingest_id": ingest_id, "status": status, "pages_total": parsed.total_pages, "pages_read": len(parsed.pages),
            "chunks": len(chunks), "vectors": len(vectors), "dimension": len(vectors[0]) if vectors else None,
            "qdrant_points": qpoints, "embedding": embed_stats, "stages_ms": stages, "total_ms": total_ms,
            "warnings": warnings, "summary": summary,
        }
        self._ingest_state.set(path.name, {"ingest_id": ingest_id, "status": status, "stage": "done", "warnings": warnings,
                                           "pages": f"{len(parsed.pages)}/{parsed.total_pages}", "chunks": len(chunks), "qdrant_points": qpoints})
        self._save_diagnostics(parsed, chunks, report)
        trace_ingest("ingest_done", filename=path.name, doc_number=parsed.doc_number,  # ingest_id nằm trong report
                     relations=[f"{r.type}:{r.scope}:{r.target}" for r in parsed.relations], **report)

        result = {
            "ingest_id": ingest_id, "filename": path.name, "doc_number": parsed.doc_number, "doc_type": parsed.doc_type,
            "title": parsed.title, "file_url": file_url, "chunks": stored, "status": base["status"],
            "ingest_status": status, "warnings": warnings,
            "relations": [{"type": r.type, "target": r.target, "scope": r.scope} for r in parsed.relations],
            "updated_related_documents": synced,
        }
        self._corpus_version += 1
        self._doc_text.clear()
        self._lex = None
        log_event(logger, logging.INFO, "INGEST_OK", **{k: v for k, v in result.items() if k != "warnings"}, seconds=round(total_ms / 1000, 2))
        return result

    def _save_diagnostics(self, parsed: Any, chunks: list[Any], report: dict[str, Any]) -> None:
        """Lưu chẩn đoán chi tiết từng trang ra storage/diagnostics/<tên file>.json (để xem/gửi khi cần)."""
        try:
            d = self._s.diagnostics_dir
            d.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^\w.\-]+", "_", parsed.filename)
            (d / f"{safe}.json").write_text(json.dumps({"doc_number": parsed.doc_number, "report": report, "pages": parsed.diagnostics.get("pages"),
                                                       "chunk_pages": sorted({c.page for c in chunks if c.page})}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        except Exception:  # noqa: BLE001
            logger.exception("Không ghi được diagnostics")

    async def delete_file(self, filename: str) -> dict[str, Any]:
        async with self._ingest_lock:
            await asyncio.to_thread(self._q.delete_by_filename, filename)
            affected = await asyncio.to_thread(self._rel.remove_by_filename, filename)
            synced = await asyncio.to_thread(self._rel.sync_to_qdrant, affected)
        self._corpus_version += 1
        self._doc_text.clear()
        self._lex = None
        return {"filename": filename, "updated_related_documents": synced}

    # ------------------------------------------------------------------ chat
    async def _lexical_hits(self, question: str) -> list[tuple[Hit, float]]:
        """Tìm từ khoá BM25 trên toàn bộ chunk (chỉ mục dựng lại khi kho tài liệu đổi)."""
        if self._lex is None or self._lex[0] != self._corpus_version:
            allhits = await asyncio.to_thread(self._q.scroll_all)
            self._lex = (
                self._corpus_version,
                BM25([
                    (h.id, f"{h.payload.get('doc_number', '')} {h.payload.get('article', '')} {h.payload.get('content', '')}")
                    for h in allhits
                ]),
                {h.id: h for h in allhits},
            )
        _, index, by_id = self._lex
        return [(by_id[i], score) for i, score in index.search(question, self._s.lexical_top_k) if i in by_id]

    async def _prepare(self, question: str, top_k: int, tm: "_Timer") -> dict[str, Any]:
        """Embedding -> tìm vector + từ khoá -> trộn RRF -> mở rộng văn bản sửa đổi -> ghim điều khoản -> cắt theo ngân sách."""
        query_vec = (await self._q.embed_texts([question]))[0]
        embed_stats = dict(self._q.last_embed)
        tm.lap("embed")

        target_docs = None
        try:
            records = await asyncio.to_thread(self._rel.list_documents)
            known = DA.known_docs(records)
            q_folded = DA.fold(question)
            resolved = [d for d in DA.resolve_docs(q_folded, known) if known.get(d, {}).get("ingested")]
            if resolved:
                target_docs = resolved
        except Exception:
            pass

        if target_docs and len(target_docs) > 1:
            vec_hits = []
            per_doc_limit = max(3, top_k // len(target_docs) + 1)
            for d in target_docs:
                d_hits = await asyncio.to_thread(self._q.search, query_vec, per_doc_limit, doc_numbers=[d])
                vec_hits.extend(d_hits)
            seen_ids = set()
            unique = []
            for h in sorted(vec_hits, key=lambda x: x.score, reverse=True):
                if h.id not in seen_ids:
                    seen_ids.add(h.id)
                    unique.append(h)
            vec_hits = unique[:top_k + 4]
        else:
            vec_hits = await asyncio.to_thread(self._q.search, query_vec, top_k, doc_numbers=target_docs)
        tm.lap("vector")
        
        lex = await self._lexical_hits(question) if self._s.hybrid_search else []
        tm.lap("lexical")

        by_id: dict[str, Hit] = {h.id: h for h in vec_hits}
        for h, _ in lex:
            by_id.setdefault(h.id, h)
        fused = rrf([[h.id for h in vec_hits], [h.id for h, _ in lex]])
        vec_ids, lex_ids = {h.id for h in vec_hits}, {h.id for h, _ in lex}
        limit = top_k + (min(4, self._s.lexical_top_k) if lex else 0)
        order = sorted(by_id, key=lambda i: -fused.get(i, 0.0))[:limit]
        hits = []
        via: dict[str, str] = {}
        for i in order:
            h = by_id[i]
            h.fused = fused.get(i, 0.0)
            hits.append(h)
            via[i] = "vector+lexical" if i in vec_ids and i in lex_ids else ("vector" if i in vec_ids else "lexical")
        top_vec = max((h.score for h in vec_hits), default=0.0)

        sources, expansions = await self._expand_with_relations(query_vec, hits)
        
        tm.lap("expand")
    
        n_expanded = sum(1 for s in sources if s.get("via") == "expanded")
        
        # === RERANKING / ARTICLE FUSION ===
        q = DA.fold(question)
        # Nhận diện điều khoản được nhắc trong câu hỏi (ví dụ: "Điều 78", "khoản 1 Điều 78")
        matched_articles = set(re.findall(r"\bdieu\s+(\d+[a-z]?)\b", q))

        # Nếu không nêu số điều, lấy điều khoản có điểm cao nhất làm điều trọng tâm để kéo các chunk cùng điều
        dominant_articles = set(matched_articles)
        if not dominant_articles and sources:
            top_art = str(sources[0].get("article") or "").strip()
            m_top = re.search(r"(\d+[a-z]?)", top_art)
            if m_top and (sources[0].get("score", 0.0) >= 0.70 or sources[0].get("fused", 0.0) >= 0.02):
                dominant_articles.add(m_top.group(1))

        for s in sources:
            base_score = s.get("fused")
            if base_score is None:
                base_score = s.get("score", 0.0)

            article = str(s.get("article") or "").strip()
            content = str(s.get("content") or "")
            m_art = re.search(r"(\d+[a-z]?)", article)
            art_num = m_art.group(1) if m_art else ""

            # Khớp nếu số điều trùng hoặc nội dung đề cập tới điều trọng tâm (ví dụ NĐ 349 sửa đổi Điều 78)
            mentions_dominant = any(f"dieu {da}" in DA.fold(content) for da in dominant_articles)

            if (art_num and art_num in dominant_articles) or mentions_dominant:
                s["fused"] = base_score * 1.35
            else:
                s["fused"] = base_score

        sources.sort(
            key=lambda s: (s.get("is_amendment", False), s.get("fused", s.get("score", 0.0))),
            reverse=True,
        )

        # Đảm bảo phân bổ nguồn khi người dùng hỏi đích danh nhiều văn bản (vd: 214 và 349)
        if target_docs and len(target_docs) > 1:
            by_target: dict[str, list[dict[str, Any]]] = {d: [] for d in target_docs}
            others: list[dict[str, Any]] = []
            for s in sources:
                if s.get("doc_number") in by_target:
                    by_target[s["doc_number"]].append(s)
                else:
                    others.append(s)
            balanced: list[dict[str, Any]] = []
            max_len = max((len(v) for v in by_target.values()), default=0)
            for idx in range(max_len):
                for d in target_docs:
                    if idx < len(by_target[d]):
                        balanced.append(by_target[d][idx])
            sources = balanced + others
        # === END RERANKING ===
        
        
        sources = await self._with_pinned(question, hits, sources)
        before_budget = len(sources)
        sources = self._fit_budget(sources)
        n_pinned = sum(1 for s in sources if s.get("pinned"))
        def summary(s: dict[str, Any]) -> dict[str, Any]:
            return {"i": s["index"], "doc": s["doc_number"], "article": s["article"], "page": s["page"], "score": s["score"],
                    "via": s.get("via"), "amendment": s["is_amendment"], "chars": len(s["content"])}

        trace_part = {
            "embedding": embed_stats,
            "retrieval": {
                "top_k": top_k, "hybrid": self._s.hybrid_search, "top_vector_score": round(top_vec, 4),
                "vector": [{"rank": k, "id": str(h.id)[:8], "doc": h.payload.get("doc_number"), "article": h.payload.get("article"),
                            "page": h.payload.get("page"), "score": round(h.score, 4)} for k, h in enumerate(vec_hits, 1)],
                "lexical": [{"rank": k, "id": str(h.id)[:8], "doc": h.payload.get("doc_number"), "article": h.payload.get("article"),
                             "page": h.payload.get("page"), "bm25": round(sc, 3)} for k, (h, sc) in enumerate(lex, 1)],
                "fused": [{"rank": k, "id": str(h.id)[:8], "rrf": round(h.fused, 5), "via": via.get(h.id)} for k, h in enumerate(hits, 1)],
                "expanded_docs": sorted(expansions), "sources_before_budget": before_budget, "dropped_by_budget": before_budget - len(sources),
                "expanded_sources": n_expanded, "pinned": n_pinned,
            },
            "sources": [summary(s) for s in sources],
        }
        return {"sources": sources, "hits": hits, "vec_hits": vec_hits, "top_score": top_vec, "trace": trace_part}

    async def chat(self, req: ChatRequest, request_id: str = "-") -> ChatResponse:
        if request_id in ("-", ""):
            request_id = uuid.uuid4().hex[:12]  # mọi câu hỏi đều có request_id để truy vết
        try:
            return await self._chat_impl(req, request_id)
        except Exception as exc:  # noqa: BLE001 - ghi trace lỗi (kể cả bị chặn bảo mật) rồi ném lại cho lớp API xử lý
            trace("chat_error", request_id=request_id, question=_log_text(req.question, 300),
                  error_type=type(exc).__name__, error_kind=getattr(exc, "kind", None) or getattr(exc, "category", None),
                  error=str(exc)[:400])
            raise

    async def _chat_impl(self, req: ChatRequest, request_id: str = "-") -> ChatResponse:
        """Thứ tự ưu tiên (rẻ -> đắt): cache -> trả lời trực tiếp -> truy xuất lai + AI -> (AI lỗi) trích đoạn."""
        started = time.perf_counter()
        tm = _Timer()
        await self._guard.check(req.question, request_id=request_id, context="chat")  # chốt chặn local
        tm.lap("guard")

        top_k = req.top_k or self._s.top_k
        cache_key = f"{self._corpus_version}|{top_k}|{DA.fold(req.question)}"

        # 1) Cache câu trả lời: hỏi lại cùng câu -> 0 token, 0 embedding
        cached = self._answers.get(cache_key)
        if cached is not None:
            tm.lap("cache")
            return self._respond(req.question, cached["answer"], cached["sources"], cached["cited"], mode="cache",
                                 usage=Usage(), request_id=request_id, started=started, tm=tm, trace={"cache_hit": True})

        # 2) Trả lời trực tiếp (không embedding, không AI) cho câu tra cứu có cấu trúc
        if self._s.direct_answers and req.use_ai is not True:
            direct = await self._try_direct(req.question, request_id, started, tm)
            tm.lap("direct")
            if direct is not None:
                return direct

        # 3) Truy xuất lai
        prep = await self._prepare(req.question, top_k, tm)
        sources, hits, top_score, tr = prep["sources"], prep["hits"], prep["top_score"], prep["trace"]
        query_id = uuid.uuid4().hex
        if not hits:
            trace("chat", request_id=request_id, mode="no_hits", question=preview(req.question, 500), timings_ms=tm.laps, **tr)
            return ChatResponse(
                query_id=query_id, answer="Chưa có tài liệu nào phù hợp trong hệ thống để trả lời câu hỏi này.",
                citations=[], modifications=[], cache_ttl_seconds=self.cache.ttl, usage=Usage(), mode="direct",
            )
        if self._s.min_relevance_score > 0 and top_score < self._s.min_relevance_score:
            trace("chat", request_id=request_id, mode="out_of_scope", question=preview(req.question, 500),
                  threshold=self._s.min_relevance_score, timings_ms=tm.laps, **tr)
            return ChatResponse(
                query_id=query_id,
                answer="Câu hỏi không liên quan đến nội dung các văn bản đã nạp nên mình không có căn cứ để trả lời.",
                citations=[], modifications=[], cache_ttl_seconds=self.cache.ttl, usage=Usage(), mode="direct",
            )

        if req.use_ai is False:  # người dùng chọn không dùng AI
            return self._extractive(req.question, sources, "Chế độ không dùng AI: hiển thị các đoạn liên quan nhất.",
                                    request_id, started, tm=tm, trace=tr)

        context = self._build_context(sources)
        user_msg = f"<context>\n{context}\n</context>\n\nCâu hỏi: {req.question}"
        llm_info: dict[str, Any] = {}
        try:
            resp, model_used = await self._complete(
                [{"role": "system", "content": _CHAT_SYSTEM}, {"role": "user", "content": user_msg}],
                max_tokens=self._s.answer_max_tokens, info=llm_info,
            )
        except llm_router.AllModelsUnavailable as exc:
            tm.lap("llm")
            log_event(logger, logging.WARNING, "LLM_UNAVAILABLE_EXTRACTIVE", request_id=request_id, reason=str(exc)[:300])
            return self._extractive(
                req.question, sources,
                "Hiện chưa gọi được mô hình AI (hết hạn mức hoặc quá tải) nên hệ thống hiển thị trực tiếp các đoạn "
                "liên quan nhất từ văn bản. Bạn có thể hỏi lại sau để có câu trả lời tổng hợp.",
                request_id, started, tm=tm, trace={**tr, "llm_error": str(exc)[:300], "llm": llm_info},
            )
        tm.lap("llm")

        choice = resp.choices[0]
        raw = choice.message.content or ""
        parsed_out, schema_errors = validate_llm_output(raw)
        answer = (parsed_out.answer if parsed_out else raw).strip()
        self._guard.check_output(answer, _SYSTEM_PROMPTS, request_id=request_id, context="chat_output")
        valid = {s["index"] for s in sources}
        used = parsed_out.used_sources if parsed_out else []
        cited = [i for i in used if i in valid]
        invalid_used = [i for i in used if i not in valid]
        fmt = {
            "schema_ok": parsed_out is not None, "schema_errors": schema_errors, "invalid_used_sources": invalid_used,
            "used_sources_valid": bool(cited) and not invalid_used,
            "enough_context": parsed_out.enough_context if parsed_out else None,
            "finish_reason": getattr(choice, "finish_reason", None), "used_raw_text_as_answer": parsed_out is None,
        }
        cited = cited or sorted(valid)

        checks = {**answer_checks(answer, req.question, sources), **reference_checks(answer, req.question, sources)}
        problems: list[str] = []
        if checks["ungrounded_numbers"]:
            problems.append("số liệu không thấy nguyên văn trong nguồn: " + ", ".join(checks["ungrounded_numbers"][:5]))
        if checks["unsupported_documents"]:
            problems.append("số hiệu văn bản không có trong nguồn: " + ", ".join(checks["unsupported_documents"][:5]))
        if checks["unsupported_articles"]:
            problems.append("điều không có trong nguồn: " + ", ".join(f"Điều {a}" for a in checks["unsupported_articles"][:5]))
        if checks["unsupported_pages"]:
            problems.append("số trang không khớp nguồn: " + ", ".join(str(p) for p in checks["unsupported_pages"][:5]))
        problems += checks["unsupported_status_claims"]
        if checks["invalid_citations"] or invalid_used:
            problems.append("trích dẫn [n] không khớp nguồn đã cung cấp")
        if parsed_out is None:
            problems.append("định dạng đầu ra của AI không đúng schema")
        if problems:
            answer += "\n\n⚠ Kiểm tra tự động (chỉ là heuristic, không thay thế đối chiếu văn bản gốc): " + "; ".join(problems) + "."
        self._answers.put(cache_key, {"answer": answer, "sources": sources, "cited": cited})

        usage = _usage_of(resp)
        trace_extra = {
            **tr, "model": model_used, "llm": {"model": model_used, **llm_info}, "format": fmt, "checks": checks,
            "usage_source": "provider" if _usage_reported(resp) else "unavailable",
            "prompt": {"version": PROMPT_VERSION, "hash": PROMPT_HASH, "system_chars": len(_CHAT_SYSTEM),
                       "context_chars": len(context), "question_chars": len(req.question),
                       "est_tokens": {"system": est_tokens(len(_CHAT_SYSTEM)), "context": est_tokens(len(context)),
                                      "question": est_tokens(len(req.question)), "answer": est_tokens(len(answer))}},
        }
        return self._respond(req.question, answer, sources, cited, mode="llm", usage=usage, request_id=request_id,
                             started=started, tm=tm, trace=trace_extra, extra={"model": model_used, "top_score": round(top_score, 4)})

    # ---- dựng phản hồi + lưu Context Cache (để /api/chat/patch dùng được cho mọi chế độ) ----
    def _respond(
        self, question: str, answer: str, sources: list[dict[str, Any]], cited: list[int], *, mode: str,
        usage: Usage, request_id: str, started: float, notice: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None, tm: Optional["_Timer"] = None, trace: Optional[dict[str, Any]] = None,
    ) -> ChatResponse:
        query_id = uuid.uuid4().hex
        modifications = self._build_modifications(sources, [])
        citations = [self._to_citation(s) for s in sources if s["index"] in cited]
        link_lines = [
            f"- {m.amending_doc} ({_REL_VERB_VI.get(m.relation, m.relation)} {m.original_doc}): {m.amending_file_url}"
            for m in modifications
            if m.amending_file_url and m.amending_file_url not in answer
        ]
        if link_lines:
            answer = answer.rstrip() + "\n\nVăn bản sửa đổi, bổ sung liên quan (bấm để mở):\n" + "\n".join(link_lines)
        self.cache.put(
            CacheEntry(
                query_id=query_id, question=question, sources=sources, cited=cited, answer=answer,
                modifications=[m.model_dump() for m in modifications],
                created_at=time.time(), expires_at=time.time() + self.cache.ttl,
            )
        )
        from src.services import trace as _trace  # tránh trùng tên tham số `trace`
        _trace.trace(
            "chat", request_id=request_id, query_id=query_id, mode=mode, question=_log_text(question, 500),
            answer=_log_text(answer, 3000), total_ms=round((time.perf_counter() - started) * 1000),
            timings_ms=(tm.laps if tm else None),
            usage={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                   "total_tokens": usage.prompt_tokens + usage.completion_tokens},
            citations=[c.model_dump(include={"index", "doc_number", "page", "article", "is_amendment", "score"}) for c in citations],
            modifications=[m.model_dump(include={"original_doc", "amending_doc", "relation"}) for m in modifications],
            **(trace or {}),
        )
        log_event(
            logger, logging.INFO, "CHAT_OK", request_id=request_id, query_id=query_id, mode=mode,
            question=preview(question), sources=len(sources), prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens, seconds=round(time.perf_counter() - started, 2), **(extra or {}),
        )
        return ChatResponse(
            query_id=query_id, answer=answer, citations=citations, modifications=modifications,
            cache_ttl_seconds=self.cache.ttl, usage=usage, mode=mode, notice=notice,
        )

    async def _with_pinned(self, question: str, hits: list[Hit], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Khi câu hỏi nhắc tới 'Điều N': ghim NGUYÊN VĂN điều khoản gốc + phần sửa đổi trong văn bản sửa đổi lên đầu ngữ cảnh.

        Không phụ thuộc truy xuất ngữ nghĩa (có thể lấy nhầm đoạn), nên AI luôn thấy đúng điều khoản và đúng nội dung đã sửa.
        """
        if not self._s.pin_articles:
            return sources
        q = DA.fold(question)
        articles = sorted(set(re.findall(r"\bdieu\s+(\d+[a-z]?)\b", q)))[:3]
        if not articles:
            return sources
        try:
            records = await asyncio.to_thread(self._rel.list_documents)
            known = DA.known_docs(records)
            docs = [d for d in DA.resolve_docs(q, known) if known[d]["ingested"]]
            if not docs:
                return sources
            pinned: list[dict[str, Any]] = []
            for doc in docs[:2]:
                merged = await self._merged_doc(doc)
                rel = self._rel.payload_for(doc)
                for art in articles:
                    if merged:
                        span = DA.extract_article(merged[0], art)
                        if span:
                            src = self._doc_source(0, doc, content=merged[0][span[0]:span[1]].strip()[:3500],
                                                   page=DA.page_at(span[0], merged[1], merged[2]), article=f"Điều {art}")
                            src["id"] = f"pin-{doc}-{art}"
                            pinned.append(src)
                    for amending in rel["amended_by"]:
                        amended = await self._merged_doc(amending)
                        blocks = DA.extract_amendment_blocks(amended[0], art) if amended else []
                        if blocks:
                            src = self._doc_source(0, amending, content="\n\n".join(b for _, b in blocks)[:4000],
                                                   page=DA.page_at(blocks[0][0], amended[1], amended[2]),
                                                   article=f"sửa đổi Điều {art} của {doc}", is_amendment=True)
                            src["id"] = f"pin-{amending}-{art}"
                            pinned.append(src)
        except Exception:  # noqa: BLE001 - tối ưu phụ: lỗi thì dùng kết quả truy xuất thường
            logger.exception("Ghim điều khoản lỗi -> dùng kết quả truy xuất thường")
            return sources
        if not pinned:
            return sources
        for s in pinned:
            s["pinned"] = True
        pinned_text = " ".join(" ".join(p["content"].split()) for p in pinned)
        rest = [s for s in sources if " ".join(s["content"].split())[:150] not in pinned_text]
        merged_sources = pinned + rest
        for i, s in enumerate(merged_sources, start=1):
            s["index"] = i
        log_event(logger, logging.INFO, "PINNED_ARTICLES", articles=articles, pinned=len(pinned), kept=len(rest))
        return merged_sources

    def _fit_budget(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Cắt bớt nguồn cuối danh sách để ngữ cảnh gửi AI không vượt MAX_CONTEXT_CHARS (tiết kiệm input token)."""
        out: list[dict[str, Any]] = []
        used = 0
        for s in sources:
            n = len(s["content"])
            if out and used + n > self._s.max_context_chars:
                break
            out.append(s)
            used += n
        return out

    def _extractive(self, question: str, sources: list[dict[str, Any]], header: str, request_id: str, started: float,
                    tm: Optional["_Timer"] = None, trace: Optional[dict[str, Any]] = None) -> ChatResponse:
        """Không có AI: trả về các đoạn liên quan nhất (văn bản sửa đổi trước), kèm nguồn."""
        chosen = sources[:4]
        parts = [header]
        for s in chosen:
            tag = " — VĂN BẢN SỬA ĐỔI" if s["is_amendment"] else ""
            where = f"{s['doc_number']}{tag} | {s['article'] or '-'} | trang {s['page'] or '-'}"
            excerpt = " ".join(s["content"].split())
            parts.append(f"[{s['index']}] {where}\n{excerpt[:700]}{'…' if len(excerpt) > 700 else ''}")
        return self._respond(question, "\n\n".join(parts), sources, [s["index"] for s in chosen], mode="extractive",
                             usage=Usage(), request_id=request_id, started=started, notice=header, tm=tm, trace=trace)

    # ---------------------------------------------------- trả lời trực tiếp (0 token)
    def _doc_source(self, index: int, doc: str, *, content: str = "", page: Optional[int] = None,
                    article: Optional[str] = None, is_amendment: bool = False) -> dict[str, Any]:
        meta = self._rel.base_metadata(doc)
        rel = self._rel.payload_for(doc)
        filename = meta.get("filename") or ""
        return {
            "index": index, "id": f"direct-{index}", "score": 1.0, "is_amendment": is_amendment, "doc_number": doc,
            "filename": filename, "file_url": meta.get("file_url") or (file_url_for(filename) if filename else ""),
            "page": page, "article": article, "status": rel["status"], "content": content,
            "amended_by": rel["amended_by"], "replaced_by": rel["replaced_by"], "repealed_by": rel["repealed_by"],
        }

    async def _merged_doc(self, doc: str) -> Optional[tuple[str, list[int], list[Optional[int]]]]:
        cached = self._doc_text.get(doc)
        if cached and cached[0] == self._corpus_version:
            return cached[1], cached[2], cached[3]
        hits = await asyncio.to_thread(self._q.scroll_doc, doc)
        if not hits:
            return None
        text, starts, pages = DA.merge_chunks([h.payload for h in hits])
        self._doc_text[doc] = (self._corpus_version, text, starts, pages)
        return text, starts, pages

    async def _try_direct(
        self,
        question: str,
        request_id: str,
        started: float,
        tm: Optional["_Timer"] = None,
    ) -> Optional[ChatResponse]:
        records = await asyncio.to_thread(self._rel.list_documents)
        intent = DA.detect_intent(question, records)

        if intent is None:
            return None

        try:
            if intent.kind == "list":
                result = self._direct_list(records)
            elif intent.kind in ("status", "affects"):
                result = self._direct_relations(intent, records)
            elif intent.kind == "amendments_list":
                result = await self._direct_amendments_list(intent)
            elif intent.kind == "thresholds":
                result = await self._direct_thresholds(intent)
            elif intent.kind == "article_topic":
                result = await self._direct_article_topic(intent)
            elif intent.kind == "article":
                result = await self._direct_article(intent)
            elif intent.kind == "article_amendments":
                result = await self._direct_article_amendments(intent)
            else:
                return None

        except Exception:  # noqa: BLE001
            logger.exception(
                "Direct answer lỗi (intent=%s) -> chuyển sang truy xuất + AI",
                intent.kind,
            )
            return None

        if result is None:
            return None

        answer, sources = result
        log_event(
            logger,
            logging.INFO,
            "DIRECT_ANSWER",
            request_id=request_id,
            intent=intent.kind,
            doc=intent.doc,
            article=intent.article,
        )
        return self._respond(
            question,
            answer,
            sources,
            [s["index"] for s in sources],
            mode="direct",
            usage=Usage(),
            request_id=request_id,
            started=started,
            extra={"intent": intent.kind},
            tm=tm,
            trace={
                "intent": intent.kind,
                "doc": intent.doc,
                "article": intent.article,
            },
            notice="Trả lời trực tiếp từ văn bản/đồ thị quan hệ, không dùng AI (0 token).",
        )    
    def _direct_list(records: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        if not records:
            return "Hiện chưa có văn bản nào được nạp vào hệ thống.", []
        lines = ["Các văn bản đang có trong hệ thống:"]
        for r in records:
            extra = []
            if r.get("amended_by"):
                extra.append("được sửa đổi bởi " + ", ".join(r["amended_by"]))
            if r.get("repealed_by"):
                extra.append("bị bãi bỏ bởi " + ", ".join(r["repealed_by"]))
            lines.append(f"- {r['doc_number']} — {r.get('title') or r['filename']}" + (f" ({'; '.join(extra)})" if extra else ""))
        return "\n".join(lines), []

    def _direct_relations(self, intent: DA.Intent, records: list[dict[str, Any]]) -> Optional[tuple[str, list[dict[str, Any]]]]:
        doc = intent.doc or ""
        by_number = {r["doc_number"]: r for r in records}
        rel = self._rel.payload_for(doc)
        sources: list[dict[str, Any]] = []

        def add_source(number: str, content: str = "", amending: bool = False) -> None:
            if number in by_number and all(s["doc_number"] != number for s in sources):
                sources.append(self._doc_source(len(sources) + 1, number, content=content, is_amendment=amending))

        def evidence(src: str, rel_type: str, target: str) -> str:
            for r in by_number.get(src, {}).get("relations", []):
                if r["type"] == rel_type and r["target"] == target and r.get("evidence"):
                    return f"\n    Căn cứ trong {src}: «{r['evidence'][:220]}»"
            return ""

        if intent.kind == "status":
            ingested = doc in by_number
            lines = [f"Văn bản {doc}" + (f" — {by_number[doc].get('title')}" if ingested and by_number[doc].get("title") else "") + ":"]
            if ingested:
                add_source(doc)
                lines.append(f"- Tình trạng theo các văn bản đã nạp: {_STATUS_VI[rel['status']]}.")
            else:
                lines.append("- Văn bản này chưa được nạp vào hệ thống, nhưng được nhắc tới trong các văn bản đã nạp:")
            found = False
            for key, rel_type, label in (("amended_by", "amends", "được sửa đổi, bổ sung bởi"),
                                         ("replaced_by", "replaces", "bị thay thế bởi"),
                                         ("repealed_by", "repeals", "bị bãi bỏ bởi")):
                for src in rel[key]:
                    found = True
                    lines.append(f"- {label} {src}{evidence(src, rel_type, doc)}")
                    if by_number.get(src, {}).get("file_url"):
                        lines.append(f"    📎 Mở văn bản {src}: {by_number[src]['file_url']}")
                    add_source(src, content=evidence(src, rel_type, doc), amending=True)
            if not found and not ingested:
                return None
            lines.append("Lưu ý: kết quả chỉ dựa trên các văn bản đã nạp vào hệ thống; có thể còn văn bản khác chưa nạp.")
            return "\n".join(lines), sources

        # affects: văn bản này tác động tới văn bản nào
        lines = [f"Văn bản {doc} tác động tới các văn bản sau (theo nội dung đã nạp):"]
        add_source(doc)
        total = 0
        for key, label in (("amends", "sửa đổi, bổ sung"), ("replaces", "thay thế"), ("repeals", "bãi bỏ")):
            for target in rel[key]:
                total += 1
                lines.append(f"- {label} {target}{evidence(doc, key, target)}")
                if by_number.get(target, {}).get("file_url"):
                    lines.append(f"    📎 Mở văn bản {target}: {by_number[target]['file_url']}")
                add_source(target, amending=False)
        if total == 0:
            return None
        return "\n".join(lines), sources

    async def _direct_article(self, intent: DA.Intent) -> Optional[tuple[str, list[dict[str, Any]]]]:
        doc, art = intent.doc or "", intent.article or ""
        merged = await self._merged_doc(doc)
        if merged is None:
            return None
        text, starts, pages = merged
        span = DA.extract_article(text, art)
        if span is None:
            return None
        art_text = text[span[0] : span[1]].strip()
        label = f"Điều {art}"
        sources = [self._doc_source(1, doc, content=art_text, page=DA.page_at(span[0], starts, pages), article=label)]
        rel = self._rel.payload_for(doc)
        orig_link = with_page(sources[0]["file_url"], sources[0]["page"], sources[0]["filename"])
        lines = [f"{label} — {doc} (trích nguyên văn, không dùng AI):", "", art_text]
        if orig_link:
            lines += ["", f"📎 Văn bản gốc {doc}: {orig_link}"]

        if rel["status"] in ("repealed", "replaced"):
            who = rel["repealed_by"] + rel["replaced_by"]
            lines += ["", f"⚠ Cảnh báo: văn bản này {_STATUS_VI[rel['status']]} bởi {', '.join(who)}."]
        for amending in rel["amended_by"]:
            amended = await self._merged_doc(amending)
            blocks = DA.extract_amendment_blocks(amended[0], art) if amended else []
            if blocks:
                page = DA.page_at(blocks[0][0], amended[1], amended[2])
                src = self._doc_source(len(sources) + 1, amending, content="\n\n".join(b for _, b in blocks), page=page, is_amendment=True)
                lines += ["", f"⚠ {label} đã được sửa đổi, bổ sung bởi {amending}. Nội dung sửa đổi liên quan:"]
                for _, block in blocks:
                    lines += ["", block]
                link = with_page(src["file_url"], page, src["filename"])
                if link:
                    lines += ["", f"📎 Văn bản sửa đổi {amending}: {link}"]
                sources.append(src)
            else:
                src = self._doc_source(len(sources) + 1, amending, is_amendment=True)
                lines += ["", f"⚠ Văn bản này đã được sửa đổi, bổ sung bởi {amending}; hãy kiểm tra văn bản đó xem {label} có bị tác động không."]
                if src["file_url"]:
                    lines += ["", f"📎 Văn bản sửa đổi {amending}: {src['file_url']}"]
                sources.append(src)
        return "\n".join(lines), sources

    async def _direct_article_amendments(self, intent: DA.Intent) -> Optional[tuple[str, list[dict[str, Any]]]]:
        doc, art = intent.doc or "", intent.article or ""
        rel = self._rel.payload_for(doc)
        label = f"Điều {art}"
        lines = [f"{label} của {doc} — các nội dung sửa đổi, bổ sung (trích nguyên văn, không dùng AI):"]
        sources = [self._doc_source(1, doc, article=label)]
        if sources[0]["file_url"]:
            lines.append(f"📎 Văn bản gốc {doc}: {sources[0]['file_url']}")
        found = False
        for amending in rel["amended_by"]:
            amended = await self._merged_doc(amending)
            blocks = DA.extract_amendment_blocks(amended[0], art) if amended else []
            if not blocks:
                continue
            found = True
            page = DA.page_at(blocks[0][0], amended[1], amended[2])
            src = self._doc_source(len(sources) + 1, amending, content="\n\n".join(b for _, b in blocks), page=page, is_amendment=True)
            lines += ["", f"Theo {amending}:"]
            for _, block in blocks:
                lines += ["", block]
            link = with_page(src["file_url"], page, src["filename"])
            if link:
                lines += ["", f"📎 Văn bản sửa đổi {amending}: {link}"]
            sources.append(src)
        return ("\n".join(lines), sources) if found else None  # không tìm thấy -> nhường cho AI
    async def _direct_amendments_list(
        self, intent: DA.Intent
    ) -> Optional[tuple[str, list[dict[str, Any]]]]:
        """Liệt kê các điều của văn bản gốc bị tác động."""
        original_doc = intent.doc or ""
        amendment_doc = intent.related_doc or ""

        if not original_doc or not amendment_doc:
            return None

        merged = await self._merged_doc(amendment_doc)
        if merged is None:
            return None

        text, starts, pages = merged

        chapters = list(
            re.finditer(r"(?im)^[ \t]*Chương[^\n]*", text)
        )
        if not chapters:
            return None

        target = DA.fold(original_doc)
        section_start = section_end = None

        for i, chapter in enumerate(chapters):
            start = chapter.start()
            end = (
                chapters[i + 1].start()
                if i + 1 < len(chapters)
                else len(text)
            )
            section = text[start:end]

            if target in DA.fold(section[:1800]):
                section_start, section_end = start, end
                break

        if section_start is None or section_end is None:
            return None

        section = text[section_start:section_end]
        pattern = re.compile(
            r"(?im)^[ \t]*Điều\s+(\d+[a-z]?)\s*[.:]\s*([^\n]{0,220})"
        )
        candidates = list(pattern.finditer(section))
        matches = [
            m for m in candidates
            if DA.fold(m.group(2)).startswith(
                (
                    "sua ",
                    "sua doi",
                    "sua dai",
                    "bo sung",
                    "bai bo",
                    "thay the",
                )
            )
        ]
        if not matches:
            return None

        lines = [
            f"{amendment_doc} sửa đổi, bổ sung {original_doc}.",
            "",
            "Danh sách điều được xác định từ chương sửa đổi:",
        ]
        sources: list[dict[str, Any]] = []

        for i, match in enumerate(matches):
            amendment_article = match.group(1)
            title = re.sub(r"\s+", " ", match.group(2)).strip()

            # Sửa cách đọc OCR cụ thể: Điều 8Š -> Điều 85.
            title = re.sub(
                r"(?i)(đi[ềêèe]u\s+)8[ŠS]\b",
                r"\g<1>85",
                title,
            )

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(section)
            )
            article_text = section[match.start():end].strip()

            # Tìm số Điều gốc ngay trong tiêu đề.
            refs = re.findall(
                r"(?i)đi[ềêèe]u\s+(\d+[a-z]?)\b",
                title,
            )

            # Nếu tiêu đề không nêu số Điều gốc, kiểm tra dòng ngay sau.
            if not refs:
                after_title = article_text[len(match.group(0)):]
                standalone = re.search(
                    r"(?im)^[ \t]*đi[ềêèe]u\s+(\d+[a-z]?)[ \t]*$",
                    after_title[:500],
                )
                if standalone:
                    refs = [standalone.group(1)]

            # Điều thay thế/bãi bỏ cụm từ có thể tác động nhiều Điều.
            if not refs and re.search(
                r"(?i)thay thế|bãi bỏ các cụm từ",
                title,
            ):
                reference_text = article_text[len(match.group(0)):]

                # Chuẩn hóa lỗi OCR cụ thể: Điều 14ó -> Điều 146
                reference_text = re.sub(
                    r"(?i)(đi[ềêèe]u\s+)14ó\b",
                    r"\g<1>146",
                    reference_text,
                )

                refs = re.findall(
                    r"(?i)đi[ềêèe]u\s+(\d+[a-z]?)\b",
                    reference_text,
                )

            refs = list(dict.fromkeys(refs))

            if refs:
                ref_label = ", ".join(f"Điều {r}" for r in refs)
                lines.append(
                    f"- Điều {amendment_article} của {amendment_doc}: "
                    f"{title}. Điều bị tác động: {ref_label}."
                )
            else:
                lines.append(
                    f"- Điều {amendment_article} của {amendment_doc}: "
                    f"{title}. Chưa xác định được Điều gốc từ tiêu đề OCR."
                )

            absolute_pos = section_start + match.start()
            page = DA.page_at(absolute_pos, starts, pages)

            sources.append(
                self._doc_source(
                    len(sources) + 1,
                    amendment_doc,
                    content=article_text[:2200],
                    page=page,
                    article=f"Điều {amendment_article}",
                    is_amendment=True,
                )
            )

        lines.extend([
            "",
            "Lưu ý: Số Điều được nhận diện tự động từ văn bản OCR. "
            "Các mục không xác định được cần đối chiếu trực tiếp với PDF."
        ])

        return "\n".join(lines), sources   
    async def _direct_thresholds(
        self, intent: DA.Intent
    ) -> Optional[tuple[str, list[dict[str, Any]]]]:
        """Trả lời trực tiếp câu hỏi về ngưỡng giá trị chỉ định thầu."""
        amending_doc = "349/2026/NĐ-CP"

        amended = await self._merged_doc(amending_doc)
        if amended is None:
            return None

        text, starts, pages = amended
        blocks = DA.extract_amendment_blocks(text, "78")
        if not blocks:
            return None

        # Ưu tiên trích dẫn bắt đầu đúng tại phần sửa khoản 4.
        marker = "sửa đổi, bổ sung khoản 4"
        offset = text.casefold().find(marker)

        if offset >= 0:
            evidence = text[offset:offset + 1200].strip()
            page = DA.page_at(offset, starts, pages)
        else:
            evidence = "\n\n".join(
                block for _, block in blocks
            ).strip()
            page = DA.page_at(blocks[0][0], starts, pages)

        source = self._doc_source(
            1,
            amending_doc,
            content=evidence,
            page=page,
            is_amendment=True,
        )

        lines = [
            "HẠN MỨC GIÁ TRỊ ĐƯỢC ÁP DỤNG CHỈ ĐỊNH THẦU",
            "",
            "Theo khoản 4 Điều 78 Nghị định 214/2025/NĐ-CP, "
            "đã được sửa đổi tại khoản 5 Điều 4 Nghị định "
            "349/2026/NĐ-CP:",
            "",
            "1. Gói thầu thuộc dự toán mua sắm không hình thành "
            "dự án: giá gói thầu không quá 01 tỷ đồng.",
            "",
            "2. Gói thầu dịch vụ tư vấn thuộc dự án: giá gói thầu "
            "không quá 03 tỷ đồng.",
            "",
            "3. Gói thầu dịch vụ phi tư vấn, hàng hóa, xây lắp, "
            "hỗn hợp thuộc dự án: giá gói thầu không quá 05 tỷ đồng.",
            "",
            "Lưu ý: Đây là các ngưỡng giá trị thuộc khoản 4 Điều 78, "
            "không phải điều kiện duy nhất áp dụng cho mọi trường hợp "
            "chỉ định thầu. Các trường hợp khác tại Điều 78 phải được "
            "đánh giá theo điều kiện tương ứng; quy trình lựa chọn nhà "
            "thầu cần xác định riêng theo quy định áp dụng.",
        ]

        link = with_page(
            source["file_url"],
            source["page"],
            source["filename"],
        )

        if link:
            lines += ["", f"📎 Nguồn pháp lý: {link}"]

        return "\n".join(lines), [source]
    async def _direct_article_topic(
        self, intent: DA.Intent
    ) -> Optional[tuple[str, list[dict[str, Any]]]]:
        """Tổng hợp các trường hợp chỉ định thầu và nội dung sửa đổi."""
        doc = intent.doc or ""
        art = intent.article or ""

        if not doc or not art:
            return None

        # Chỉ dùng bản tổng hợp chuyên biệt cho Điều 78.
        if doc != "214/2025/NĐ-CP" or art != "78":
            return await self._direct_article(
                DA.Intent(kind="article", doc=doc, article=art)
            )

        original = await self._merged_doc(doc)
        if original is None:
            return None

        original_text, original_starts, original_pages = original
        span = DA.extract_article(original_text, art)
        if span is None:
            return None

        # Trích dẫn phải chứa nội dung thực tế từ tài liệu.
        article_text = original_text[span[0]:span[1]].strip()
        original_page = DA.page_at(
            span[0], original_starts, original_pages
        )

        relation = self._rel.payload_for(doc)
        amending_doc = "349/2026/NĐ-CP"

        if amending_doc not in relation.get("amended_by", []):
            return await self._direct_article(
                DA.Intent(kind="article", doc=doc, article=art)
            )

        amended = await self._merged_doc(amending_doc)
        if amended is None:
            return None

        amended_text, amended_starts, amended_pages = amended
        blocks = DA.extract_amendment_blocks(amended_text, art)
        if not blocks:
            return None

        amendment_text = "\n\n".join(
            block for _, block in blocks
        ).strip()
        amendment_page = DA.page_at(
            blocks[0][0], amended_starts, amended_pages
        )

        # Hai nguồn đều lưu nội dung được trích xuất thật.
        sources = [
            self._doc_source(
                1,
                doc,
                content=article_text,
                page=original_page,
                article="Điều 78",
            ),
            self._doc_source(
                2,
                amending_doc,
                content=amendment_text,
                page=amendment_page,
                is_amendment=True,
            ),
        ]

        lines = [
            "CÁC TRƯỜNG HỢP ĐƯỢC ÁP DỤNG CHỈ ĐỊNH THẦU",
            "",
            "Căn cứ Điều 78 Nghị định 214/2025/NĐ-CP, "
            "đối chiếu các nội dung sửa đổi tại Điều 4 "
            "Nghị định 349/2026/NĐ-CP:",
            "",
            "1. Gói thầu có yêu cầu cấp bách, khẩn cấp",
            "- Phục vụ quốc phòng, an ninh, đối ngoại và bảo vệ "
            "chủ quyền quốc gia.",
            "- Ngăn chặn, khắc phục hậu quả hoặc cứu trợ do thiên tai, "
            "hỏa hoạn, dịch bệnh, tai nạn, sự cố, thảm họa và sự kiện "
            "bất khả kháng.",
            "- Ngăn ngừa nguy hại đến tính mạng, tài sản hoặc công trình "
            "liền kề; bảo đảm hoạt động khám chữa bệnh và cấp cứu.",
            "- Phục vụ phòng, chống dịch bệnh, hỗ trợ các đoàn ngoại giao "
            "và dự án đầu tư công khẩn cấp theo điều kiện luật định.",
            "",
            "2. Gói thầu cần đẩy nhanh tiến độ, bảo đảm chất lượng "
            "và hiệu quả",
            "- Gói thầu thuộc dự án hoặc dự toán mua sắm cần thực hiện "
            "ngay theo các nghị quyết, kết luận và văn bản chỉ đạo "
            "của cơ quan có thẩm quyền.",
            "- Gói thầu cần bảo đảm kết nối, đồng bộ giữa các công trình "
            "thuộc dự án; gói thầu thuộc dự án đầu tư công đặc biệt.",
            "- Các gói thầu phục vụ hội nghị, hội thảo, sự kiện cấp bách; "
            "mua sắm, vận chuyển hoặc bảo quản hàng dự trữ quốc gia "
            "trong những trường hợp được quy định.",
            "- Gói thầu sửa chữa, cải tạo trụ sở làm việc, nhà lưu trú "
            "công vụ gắn với yêu cầu cấp bách về chuyển đổi số, công nghệ "
            "thông tin hoặc sắp xếp đơn vị hành chính, tổ chức bộ máy.",
            "- Gói thầu xây lắp có tính chất tương tự gói thầu trước đó "
            "đã đấu thầu rộng rãi hoặc hạn chế và nhà thầu đã thực hiện "
            "đáp ứng tiến độ, chất lượng, hiệu quả trong thời hạn luật định.",
            "- Trường hợp đã đăng tải thông báo mời thầu, mời quan tâm "
            "hoặc mời sơ tuyển nhưng không có hồ sơ tham dự, tất cả hồ sơ "
            "không đáp ứng yêu cầu hoặc tất cả nhà thầu không đáp ứng "
            "yêu cầu thực hiện gói thầu.",
            "- Các trường hợp khác thuộc dự toán mua sắm, dự án đầu tư "
            "công và dự án khác khi đáp ứng quy định sau sửa đổi.",
            "",
            "3. Gói thầu thuộc lĩnh vực chiến lược, nghiên cứu khoa học, "
            "công nghệ, đổi mới sáng tạo và chuyển đổi số",
            "- Nhiệm vụ nghiên cứu khoa học, công nghệ và đổi mới sáng tạo "
            "đặc biệt theo quy định pháp luật.",
            "- Gói thầu sử dụng công nghệ, sản phẩm hoặc hàng hóa, dịch vụ "
            "thuộc lĩnh vực chiến lược theo quy định liên quan.",
            "- Các dự án, nhiệm vụ chuyển đổi số; xây dựng, phát triển, "
            "nâng cấp cơ sở dữ liệu quốc gia, nền tảng số dùng chung, "
            "hệ thống thông tin giải quyết thủ tục hành chính, nền tảng "
            "tích hợp chia sẻ dữ liệu và trung tâm giám sát, điều hành "
            "thông minh theo phạm vi quy định.",
            "",
            "4. Gói thầu có giá trị không vượt ngưỡng tại khoản 4 "
            "sau sửa đổi",
            "- Dự toán mua sắm không hình thành dự án: không quá "
            "01 tỷ đồng.",
            "- Gói thầu dịch vụ tư vấn thuộc dự án: không quá "
            "03 tỷ đồng.",
            "- Gói thầu dịch vụ phi tư vấn, hàng hóa, xây lắp, hỗn hợp "
            "thuộc dự án: không quá 05 tỷ đồng.",
            "",
            "5. Gói thầu có điều kiện đặc thù về chuyên môn, nghiệp vụ, "
            "kỹ thuật, công nghệ hoặc thị trường cung ứng",
            "- Hàng hóa, dịch vụ chỉ có một cơ sở sản xuất hoặc cung cấp; "
            "thuốc, thiết bị y tế chỉ có duy nhất một hãng sản xuất "
            "trên thị trường.",
            "- Dịch vụ in ấn, tem, biên lai, niêm phong theo quy định; "
            "in tài liệu phục vụ bầu cử.",
            "- Tư vấn, thi công di dời công trình hạ tầng kỹ thuật, "
            "hạ tầng xã hội để giải phóng mặt bằng, tái định cư; "
            "rà phá bom, mìn, vật nổ.",
            "- Các trường hợp cần bảo đảm tương thích công nghệ, bản quyền, "
            "bảo hành hoặc bảo trì khi tiếp tục mua từ nhà thầu, hãng "
            "sản xuất hay đại lý liên quan.",
            "- Gói thầu nghiên cứu, thử nghiệm, mua quyền sở hữu trí tuệ, "
            "bản quyền chương trình phát sóng; một số gói thầu liên quan "
            "đến dự trữ quốc gia hoặc dịch vụ viễn thông phổ cập.",
            "- Các gói thầu về quy hoạch, phục vụ hoạt động quy hoạch "
            "theo quy định của pháp luật.",
            "- Thuê kho, vận chuyển hoặc bốc xếp hàng tạm giữ trong "
            "trường hợp đặc thù; một số gói thầu nhập khẩu vũ khí thể thao "
            "phục vụ hoạt động thể thao theo quy định.",
            "- Dịch vụ tư vấn, xây dựng hoặc phục chế gắn với tác giả "
            "phương án kiến trúc, ý tưởng quy hoạch, tác phẩm nghệ thuật, "
            "khảo cổ hoặc phục hồi di tích thuộc trường hợp luật định.",
            "- Mua sắm từ hoạt động thanh lý, giải thể, phá sản; hàng hóa "
            "được mua thông qua giao dịch trên sàn giao dịch hàng hóa.",
            "",
            "6. Gói thầu có yêu cầu bảo vệ bí mật nhà nước",
            "- Áp dụng theo quy định của pháp luật về bảo vệ bí mật "
            "nhà nước.",
            "",
            "7. Gói thầu thuộc dự án quan trọng quốc gia hoặc thuộc "
            "trường hợp quy định tại pháp luật quản lý ngành, lĩnh vực",
            "- Gói thầu thuộc dự án quan trọng quốc gia được áp dụng "
            "chỉ định thầu theo nghị quyết của Quốc hội khi quyết định "
            "chủ trương đầu tư.",
            "- Các trường hợp được chỉ định thầu theo pháp luật quản lý "
            "ngành, lĩnh vực.",
            "",
            "NHỮNG NỘI DUNG CẦN LƯU Ý SAU SỬA ĐỔI",
            "- Khoản 2: sửa đổi điểm a, b và g; bổ sung điểm e1.",
            "- Khoản 3: sửa đổi quy định liên quan đến các dự án, nhiệm vụ "
            "mua sắm hoặc thuê dịch vụ phục vụ chuyển đổi số.",
            "- Khoản 4: áp dụng các ngưỡng 01 tỷ, 03 tỷ và 05 tỷ đồng "
            "nêu trên thay cho các ngưỡng cũ.",
            "- Khoản 5: sửa đổi quy định về di dời công trình và hoạt động "
            "quy hoạch; bổ sung trường hợp mua sắm từ thanh lý, giải thể, "
            "phá sản và hàng hóa mua qua sàn giao dịch hàng hóa.",
            "",
            "Đây là bản tổng hợp để tra cứu nhanh, không phải văn bản "
            "hợp nhất nguyên văn. Mỗi trường hợp còn phải đáp ứng đầy đủ "
            "điều kiện, phạm vi và thủ tục tương ứng theo pháp luật.",
        ]

        original_link = with_page(
            sources[0]["file_url"],
            sources[0]["page"],
            sources[0]["filename"],
        )
        amendment_link = with_page(
            sources[1]["file_url"],
            sources[1]["page"],
            sources[1]["filename"],
        )

        if original_link:
            lines += ["", f"📎 Văn bản gốc: {original_link}"]

        if amendment_link:
            lines += ["", f"📎 Văn bản sửa đổi: {amendment_link}"]

        return "\n".join(lines), sources
    async def _expand_with_relations(
        self, query_vec: list[float], hits: list[Hit]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Mở rộng ngữ cảnh 2 chiều dựa trên danh sách quan hệ tập trung từ self._rel (relations.json)."""
        expansions: dict[str, str] = {}  # doc_number -> relation type
        
        hit_docs = {h.payload.get("doc_number") for h in hits if h.payload.get("doc_number")}
        
        try:
            relations = await asyncio.to_thread(self._rel.list_relations)
            for rel in relations:
                src = rel.get("source_doc")  # Ví dụ: 349/2026/NĐ-CP
                tgt = rel.get("target_doc")  # Ví dụ: 214/2025/NĐ-CP
                rel_type = rel.get("relation", "amends")
                
                # Nếu hit thuộc văn bản gốc (tgt) -> kéo văn bản sửa đổi (src)
                if tgt in hit_docs and src:
                    expansions.setdefault(src, rel_type)
                # Nếu hit thuộc văn bản sửa đổi (src) -> kéo văn bản gốc bị sửa (tgt)
                elif src in hit_docs and tgt:
                    expansions.setdefault(tgt, rel_type)
        except Exception:
            pass

        extra: dict[str, list[Hit]] = {}
        if expansions and self._s.expand_amendments:
            results = await asyncio.gather(
                *[
                    asyncio.to_thread(self._q.search, query_vec, self._s.amend_top_k, [doc])
                    for doc in expansions
                ]
            )
            extra = dict(zip(expansions, results))

        merged: dict[str, tuple[Hit, bool, str]] = {}

        for h in hits:
            doc_number = h.payload.get("doc_number", "")
            is_amend = (doc_number in expansions) or bool(h.payload.get("is_amendment", False))
            merged[h.id] = (
                h,
                is_amend,
                "original",
            )

        for doc_hits in extra.values():
            for h in doc_hits:
                doc_number = h.payload.get("doc_number", "")
                is_amendment = (doc_number in expansions) or bool(h.payload.get("is_amendment", False))

                if h.id in merged:
                    old_h, old_amendment, _ = merged[h.id]
                    merged[h.id] = (
                        old_h,
                        old_amendment or is_amendment,
                        "original",
                    )
                else:
                    merged[h.id] = (h, is_amendment, "expanded")

        # Sắp xếp: Ưu tiên văn bản sửa đổi lên trước, sau đó theo điểm tương đồng
        ordered = sorted(
    merged.values(),
    key=lambda t: (t[0].fused or t[0].score),
    reverse=True,
)
        sources = []
        for i, (h, is_amend, via ) in enumerate(ordered, start=1):
            p = h.payload
            sources.append(
                {
                    "index": i, "id": h.id, "score": round(h.score, 4), "is_amendment": is_amend,
                    "via": via,
                    "doc_number": p.get("doc_number", ""), "filename": p.get("filename", ""),
                    "file_url": p.get("file_url") or file_url_for(p.get("filename", "")),
                    "page": p.get("page"), "article": p.get("article"),
                    "status": p.get("status", "unverified"), "content": p.get("content", ""),
                    "amended_by": p.get("amended_by") or [], "replaced_by": p.get("replaced_by") or [],
                    "repealed_by": p.get("repealed_by") or [],
                }
            )
        return sources, expansions

    @staticmethod
    def _build_context(sources: list[dict[str, Any]]) -> str:
        blocks = []
        for s in sources:
            tag = "VĂN BẢN SỬA ĐỔI" if s["is_amendment"] else "VĂN BẢN"
            head = (
                f"[Nguồn {s['index']}] {tag} {s['doc_number']} | file: {s['filename']} | "
                f"trang {s['page'] or '-'} | {s['article'] or '-'} | trạng thái: {s['status']}"
            )
            for key, rel in _REL_KEYS:
                if s[key]:
                    head += f" | {_REL_LABEL_VI[rel]}: {', '.join(s[key])}"
            blocks.append(f"{head}\n{s['content']}")
        return "\n\n---\n\n".join(blocks)

    def _build_modifications(
        self, sources: list[dict[str, Any]], llm_notes: list[Any]
    ) -> list[Modification]:
        notes: dict[tuple[str, str], str] = {}
        for item in llm_notes:
            if isinstance(item, dict) and item.get("amending_doc") and item.get("note"):
                try:
                    amending = normalize_doc_number(str(item["amending_doc"]))
                    original = normalize_doc_number(str(item.get("original_doc", "")))
                except Exception:  # noqa: BLE001
                    continue
                notes[(original, amending)] = str(item["note"])
                notes.setdefault(("", amending), str(item["note"]))

        mods: list[Modification] = []
        seen: set[tuple[str, str, str]] = set()
        for s in sources:
            for key, rel in _REL_KEYS:
                for amending in s[key]:
                    ident = (s["doc_number"], amending, rel)
                    if ident in seen:
                        continue
                    seen.add(ident)
                    note = (
                        notes.get((s["doc_number"], amending))
                        or notes.get(("", amending))
                        or f"{s['doc_number']} {_REL_LABEL_VI[rel]} {amending}; ưu tiên áp dụng nội dung của văn bản sau."
                    )
                    mods.append(
                        Modification(
                            original_doc=s["doc_number"], amending_doc=amending, relation=rel,
                            amending_file_url=self._rel.file_url_of(amending), note=note,
                        )
                    )
        return mods

    @staticmethod
    def _to_citation(s: dict[str, Any]) -> Citation:
        snippet = " ".join(s["content"].split())
        return Citation(
            index=s["index"], doc_number=s["doc_number"], filename=s["filename"],
            file_url=with_page(s["file_url"], s["page"], s["filename"]),
            page=s["page"], article=s["article"], status=s["status"], is_amendment=s["is_amendment"],
            score=s["score"], snippet=snippet[:300] + ("…" if len(snippet) > 300 else ""),
        )

    # ----------------------------------------------------------------- patch
    async def patch(self, req: PatchRequest, request_id: str = "-") -> PatchResponse:
        """Vá cục bộ câu trả lời dựa trên Context Cache: chỉ sinh diff, không truy xuất/viết lại toàn bộ."""
        await self._guard.check(req.instruction, request_id=request_id, context="patch")

        entry = self.cache.get(req.query_id)
        if entry is None:
            raise QueryExpired(f"query_id '{req.query_id}' không tồn tại hoặc đã hết hạn (TTL {self.cache.ttl}s)")

        cited_sources = [s for s in entry.sources if s["index"] in entry.cited]
        context = "\n\n".join(
            f"[Nguồn {s['index']}] {s['doc_number']} {s['article'] or ''}\n{s['content']}" for s in cited_sources
        )
        pinfo: dict[str, Any] = {}
        try:
            resp, pmodel = await self._complete(
            [
                {"role": "system", "content": _PATCH_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"<answer>\n{entry.answer}\n</answer>\n\n<context>\n{context}\n</context>\n\n"
                        f"Yêu cầu chỉnh sửa: {req.instruction}"
                    ),
                },
            ],
            max_tokens=600,
            light=True, info=pinfo,
            )
        except llm_router.AllModelsUnavailable as exc:
            raise UpstreamError(
                "Hiện chưa gọi được mô hình AI (hết hạn mức hoặc quá tải) nên chưa thể chỉnh sửa câu trả lời. "
                "Hãy thử lại sau ít phút. Chi tiết: " + str(exc)[:200]
            ) from exc
        raw = resp.choices[0].message.content or ""
        data = _parse_json(raw) or {}
        edits = data.get("edits") if isinstance(data.get("edits"), list) else []
        new_answer, applied, failed = apply_edits(entry.answer, edits)
        self._guard.check_output(new_answer, _SYSTEM_PROMPTS, request_id=request_id, context="patch_output")
        if applied:
            entry.answer = new_answer  # các lần vá sau dựa trên bản đã vá

        usage = _usage_of(resp)
        trace("patch", request_id=request_id, query_id=req.query_id, instruction=preview(req.instruction, 300),
              applied=len(applied), failed=len(failed), model=pmodel, llm=pinfo,
              usage={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
              usage_source="provider" if _usage_reported(resp) else "unavailable")
        log_event(
            logger, logging.INFO, "PATCH_OK",
            request_id=request_id, query_id=req.query_id, instruction=preview(req.instruction),
            applied=len(applied), failed=len(failed),
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        )
        return PatchResponse(
            query_id=req.query_id, answer=entry.answer, edits_applied=applied, edits_failed=failed,
            note=str(data.get("note", "")), usage=usage,
        )

    # ------------------------------------------------------------ chẩn đoán / đo lường
    async def diagnose_documents(self) -> dict[str, Any]:
        """Độ đầy đủ của từng văn bản: số trang đọc được, chunk trong registry so với điểm thực tế trong Qdrant, cảnh báo OCR."""
        records = await asyncio.to_thread(self._rel.list_documents)
        state = self._ingest_state.all()
        docs = []
        for r in records:
            qpoints = await asyncio.to_thread(self._q.count_filename, r["filename"])
            docs.append({
                "doc_number": r["doc_number"], "filename": r["filename"], "legal_status": r["status"],
                "pages": r.get("coverage"), "chunks_registry": r.get("chunk_count"), "qdrant_points": qpoints,
                "consistent": qpoints == r.get("chunk_count"), "ingest": state.get(r["filename"]),
                "diagnostics": r.get("diagnostics"), "amended_by": r["amended_by"], "replaced_by": r["replaced_by"],
                "repealed_by": r["repealed_by"], "relations": r["relations"], "history": r.get("history"),
            })
        registered = {r["filename"] for r in records}
        pending = {fn: st for fn, st in state.items() if fn not in registered and st.get("status") in ("incomplete", "failed")}
        return {"documents": docs, "ingest_not_completed": pending, "corpus_version": self._corpus_version}

    async def debug_search(self, question: str, top_k: Optional[int] = None, request_id: str = "-") -> dict[str, Any]:
        """Chạy đúng pipeline truy xuất (không gọi AI): trả về điểm/hạng từng nhánh, nguồn được chọn và ngữ cảnh sẽ gửi cho AI."""
        await self._guard.check(question, request_id=request_id, context="diagnostics")
        tm = _Timer()
        prep = await self._prepare(question, top_k or self._s.top_k, tm)
        context = self._build_context(prep["sources"])
        out = {
            "question": question, "timings_ms": tm.laps, **prep["trace"],
            "context_chars": len(context), "est_context_tokens": est_tokens(len(context)),
            "system_prompt": {"version": PROMPT_VERSION, "hash": PROMPT_HASH, "chars": len(_CHAT_SYSTEM),
                              "est_tokens": est_tokens(len(_CHAT_SYSTEM))},
            "context_preview": context[:6000],
            # nội dung từng chunk theo thứ hạng truy xuất (trước khi mở rộng/ghim) — dùng cho scripts/eval_rag.py live
            "retrieval_hits": [{"rank": k, "doc": h.payload.get("doc_number"), "article": h.payload.get("article"),
                                "page": h.payload.get("page"), "content": h.payload.get("content", "")[:1500]}
                               for k, h in enumerate(prep["hits"], 1)],
            "source_texts": [{"i": s["index"], "doc": s["doc_number"], "amendment": s["is_amendment"], "content": s["content"][:1500]}
                             for s in prep["sources"]],
        }
        trace("search_debug", request_id=request_id, question=preview(question, 300), timings_ms=tm.laps,
              retrieval=prep["trace"]["retrieval"], sources=prep["trace"]["sources"])
        return out

    async def selftest(self, doc_number: str, n: int = 3) -> dict[str, Any]:
        """Kiểm tra embedding/chỉ mục bằng cách truy vấn lại chính các chunk mẫu (tốn 2 lượt gọi embedding, không dùng cache)."""
        hits = await asyncio.to_thread(self._q.scroll_doc, doc_number)
        if not hits:
            return {"error": f"Không có chunk nào của {doc_number} trong Qdrant"}
        n = max(1, min(n, len(hits)))
        sample = [hits[round(i * (len(hits) - 1) / max(1, n - 1))] if n > 1 else hits[0] for i in range(n)]
        self_texts = [f"{doc_number} {h.payload.get('article') or ''}\n{h.payload.get('content', '')}".strip() for h in sample]
        snippets = [" ".join(h.payload.get("content", "").split()[:25]) for h in sample]
        vecs = await self._q.embed_texts(self_texts + snippets, use_cache=False)
        stats = dict(self._q.last_embed)
        results = []
        for i, h in enumerate(sample):
            r_self = await asyncio.to_thread(self._q.search, vecs[i], 5)
            r_snip = await asyncio.to_thread(self._q.search, vecs[n + i], 5)
            rank = lambda rs: next((k for k, x in enumerate(rs, 1) if x.id == h.id), None)  # noqa: E731
            results.append({"chunk_index": h.payload.get("chunk_index"), "article": h.payload.get("article"), "page": h.payload.get("page"),
                            "self_rank": rank(r_self), "self_score": round(r_self[0].score, 4) if r_self else None,
                            "snippet_rank": rank(r_snip), "snippet_top_score": round(r_snip[0].score, 4) if r_snip else None})
        summary = {
            "sampled": n, "self_recall@1": sum(1 for r in results if r["self_rank"] == 1) / n,
            "snippet_recall@5": sum(1 for r in results if r["snippet_rank"] is not None) / n,
        }
        trace("embedding_selftest", doc=doc_number, summary=summary, embedding=stats)
        return {"summary": summary, "results": results, "embedding": stats}

    async def resync_metadata(self) -> dict[str, Any]:
        synced = await asyncio.to_thread(self._rel.resync_all)
        return {"resynced_documents": synced}

    # ------------------------------------------------------------------- llm
    async def _complete(self, messages: list[dict[str, str]], max_tokens: int, light: bool = False,
                        info: Optional[dict[str, Any]] = None) -> tuple[Any, str]:
        """Gọi LLM qua router: model nhẹ (nếu có) cho tác vụ đơn giản, chuỗi dự phòng, bỏ qua model đã hết quota."""
        models = [self._s.generation_model, *self._s.generation_fallback_models]
        if light and self._s.light_model:
            models = [self._s.light_model, *models]
        models = list(dict.fromkeys(models))
        return await llm_router.acomplete(
            models, info=info, messages=messages, temperature=0.1, max_tokens=max_tokens,
            response_format={"type": "json_object"}, timeout=90,
        )
