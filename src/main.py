"""FastAPI: upload, phục vụ file gốc, chat, patch."""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import aiofiles
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.config import config_warnings, get_settings
from src.services import llm_router
from src.services.document_parser import ParseError, ocr_status
from src.services.logger_service import get_logger, log_event, setup_logging
from src.services.qdrant_service import EmbeddingError, QdrantService
from src.services.rag_engine import (
    ChatRequest,
    ChatResponse,
    PatchRequest,
    PatchResponse,
    QueryExpired,
    RAGEngine,
    UpstreamError,
)
from src.services.relations_service import RelationsService
from src.services.security_guard import SecurityGuard, SecurityUnavailable, SecurityViolation

setup_logging()
logger = get_logger("api")
settings = get_settings()

ALLOWED_EXT = {".pdf", ".docx"}
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    qdrant = QdrantService(settings)
    relations = RelationsService(qdrant, settings)
    app.state.engine = RAGEngine(qdrant, relations, SecurityGuard(settings), settings)
    app.state.qdrant = qdrant
    app.state.relations = relations
    try:
        await asyncio.to_thread(qdrant.ensure_collection)
    except Exception as exc:  # noqa: BLE001 - không chặn khởi động; sẽ thử lại khi có request
        logger.error("Qdrant chưa sẵn sàng khi khởi động: %s", exc)
    for warning in config_warnings():
        logger.error("CẤU HÌNH SAI: %s", warning)
    logger.info("Doc QA system started")
    yield


app = FastAPI(title="Legal/Doc RAG System", version="1.0.0", lifespan=lifespan)

# Phục vụ giao diện web
_static_dir = Path(__file__).resolve().parent.parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(_static_dir), html=True), name="static")


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/index.html")


# ---------------------------------------------------------------------------
# Middleware & exception handlers
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
    request.state.request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error | request_id=%s %s %s", request.state.request_id, request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Lỗi hệ thống nội bộ", "request_id": request.state.request_id})
    response.headers["X-Request-ID"] = request.state.request_id
    logger.info(
        "%s %s -> %d (%.0f ms) | request_id=%s",
        request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000,
        request.state.request_id,
    )
    return response


def _error(status: int, code: str, message: str, request: Request, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"detail": {"error": code, "message": message, "request_id": request.state.request_id, **extra}},
    )


@app.exception_handler(SecurityViolation)
async def on_security_violation(request: Request, exc: SecurityViolation) -> JSONResponse:
    return _error(
        400, "SECURITY_VIOLATION",
        "Yêu cầu bị từ chối do vi phạm chính sách an toàn (nghi ngờ Prompt Injection/Jailbreak/lộ System Prompt).",
        request, category=exc.category,
    )


@app.exception_handler(SecurityUnavailable)
async def on_security_unavailable(request: Request, exc: SecurityUnavailable) -> JSONResponse:
    return _error(503, "SECURITY_UNAVAILABLE", str(exc), request)


@app.exception_handler(QueryExpired)
async def on_query_expired(request: Request, exc: QueryExpired) -> JSONResponse:
    return _error(410, "QUERY_EXPIRED", str(exc), request)


@app.exception_handler(ParseError)
async def on_parse_error(request: Request, exc: ParseError) -> JSONResponse:
    return _error(422, "PARSE_ERROR", str(exc), request)


@app.exception_handler(UpstreamError)
async def on_upstream_error(request: Request, exc: UpstreamError) -> JSONResponse:
    return _error(502, "UPSTREAM_ERROR", str(exc), request)


@app.exception_handler(EmbeddingError)
async def on_embedding_error(request: Request, exc: EmbeddingError) -> JSONResponse:
    return _error(502, "EMBEDDING_ERROR", str(exc), request)


@app.exception_handler(ResponseHandlingException)
@app.exception_handler(UnexpectedResponse)
async def on_qdrant_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Qdrant error | request_id=%s | %s", request.state.request_id, exc)
    return _error(503, "VECTOR_DB_UNAVAILABLE", "Không kết nối được Qdrant", request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _engine(request: Request) -> RAGEngine:
    return request.app.state.engine  # type: ignore[no-any-return]


def _safe_filename(name: str | None) -> str:
    base = Path((name or "").replace("\\", "/")).name
    base = re.sub(r"[\x00-\x1f<>:\"|?*]", "", base).strip().strip(".")
    if not base or Path(base).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .pdf hoặc .docx")
    return base


async def _save_upload(file: UploadFile, filename: str) -> Path:
    dest = settings.docs_dir / filename
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"File vượt quá {settings.max_upload_mb}MB")
                await out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return dest


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    qdrant: QdrantService = request.app.state.qdrant
    return {
        "status": "ok",
        "qdrant": await asyncio.to_thread(qdrant.ping),
        # model đang tạm ngưng do hết quota/quá tải (tự khôi phục sau thời gian chờ)
        "llm_cooldown": llm_router.status(),
        "ocr": await asyncio.to_thread(ocr_status),
        "config_warnings": config_warnings(),  # rỗng = cấu hình hợp lệ
    }


@app.post("/api/documents/upload")
async def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Upload 1..n file PDF/DOCX -> lưu storage/docs/ -> nạp vào Qdrant + cập nhật quan hệ."""
    engine = _engine(request)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for upload in files:
        try:
            filename = _safe_filename(upload.filename)
            path = await _save_upload(upload, filename)
            results.append(await engine.ingest_file(path))
        except HTTPException as exc:
            errors.append({"filename": upload.filename or "?", "error": str(exc.detail)})
        except (ParseError, EmbeddingError) as exc:
            errors.append({"filename": upload.filename or "?", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingest failed for %s", upload.filename)
            errors.append({"filename": upload.filename or "?", "error": f"Lỗi nạp tài liệu: {exc}"})
    if errors and not results:
        raise HTTPException(status_code=422, detail={"errors": errors})
    return {"ingested": results, "errors": errors}


@app.post("/api/documents/reindex")
async def reindex_documents(request: Request) -> dict[str, Any]:
    """Nạp lại toàn bộ file đang có trong storage/docs/ (tiện khi copy file thủ công)."""
    engine = _engine(request)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(settings.docs_dir.iterdir()):
        if path.suffix.lower() not in ALLOWED_EXT:
            continue
        try:
            results.append(await engine.ingest_file(path))
        except Exception as exc:  # noqa: BLE001
            log_event(logger, logging.ERROR, "REINDEX_FAIL", filename=path.name, error=str(exc))
            errors.append({"filename": path.name, "error": str(exc)})
    return {"ingested": results, "errors": errors}


@app.get("/api/documents")
async def list_documents(request: Request) -> dict[str, Any]:
    relations: RelationsService = request.app.state.relations
    return {"documents": relations.list_documents()}


@app.delete("/api/documents/{filename}")
async def delete_document(request: Request, filename: str) -> dict[str, Any]:
    safe = _safe_filename(filename)
    result = await _engine(request).delete_file(safe)
    (settings.docs_dir / safe).unlink(missing_ok=True)
    return result


@app.get("/api/files/{filename}")
async def get_file(filename: str, download: bool = Query(False)) -> FileResponse:
    """Mở (inline) hoặc tải (download=true) file gốc."""
    safe = Path(filename).name
    path = (settings.docs_dir / safe).resolve()
    if path.parent != settings.docs_dir.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu")
    return FileResponse(
        path,
        media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        filename=safe,
        content_disposition_type="attachment" if download else "inline",
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    return await _engine(request).chat(body, request_id=request.state.request_id)


@app.post("/api/chat/patch", response_model=PatchResponse)
async def chat_patch(request: Request, body: PatchRequest) -> PatchResponse:
    return await _engine(request).patch(body, request_id=request.state.request_id)


# ---------------------------------------------------------------------------
# Chẩn đoán & đo lường (không gọi AI trừ selftest/embedding)
# ---------------------------------------------------------------------------
class SearchDebugRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SelfTestRequest(BaseModel):
    doc: str = Field(min_length=3, max_length=100)
    n: int = Field(default=3, ge=1, le=10)


@app.get("/api/diagnostics/documents")
async def diag_documents(request: Request) -> dict[str, Any]:
    """Độ đầy đủ từng văn bản: trang đọc được, chunk so với điểm Qdrant, cảnh báo OCR, lịch sử nạp."""
    return await _engine(request).diagnose_documents()


@app.get("/api/diagnostics/ingest")
async def diag_ingest(request: Request) -> dict[str, Any]:
    """Trạng thái nạp từng file (complete / complete_with_warnings / incomplete / failed) — file incomplete có thể chạy lại để tiếp tục."""
    return _engine(request)._ingest_state.all()


@app.post("/api/diagnostics/search")
async def diag_search(request: Request, body: SearchDebugRequest) -> dict[str, Any]:
    """Chạy pipeline truy xuất (KHÔNG gọi AI): điểm/hạng vector + BM25 + RRF, nguồn được chọn, ngữ cảnh sẽ gửi AI."""
    return await _engine(request).debug_search(body.question, body.top_k, request_id=request.state.request_id)


@app.post("/api/diagnostics/selftest")
async def diag_selftest(request: Request, body: SelfTestRequest) -> dict[str, Any]:
    """Kiểm tra embedding/chỉ mục của một văn bản (tốn 2 lượt gọi embedding)."""
    return await _engine(request).selftest(body.doc, body.n)


@app.post("/api/documents/resync")
async def resync_documents(request: Request) -> dict[str, Any]:
    """Đồng bộ lại metadata quan hệ/trạng thái vào Qdrant (chỉ payload, không đụng vector, không re-embedding)."""
    return await _engine(request).resync_metadata()


@app.get("/api/diagnostics/trace")
async def diag_trace(limit: int = Query(50, ge=1, le=500), kind: str | None = None) -> dict[str, Any]:
    """Các dòng trace gần nhất (đã che khoá) — tiện để gửi cho người hỗ trợ."""
    import json as _json

    path = settings.trace_file
    if not path.exists():
        return {"events": []}
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-2000:]:
        try:
            ev = _json.loads(line)
        except ValueError:
            continue
        if kind is None or ev.get("kind") == kind:
            events.append(ev)
    return {"events": events[-limit:]}
