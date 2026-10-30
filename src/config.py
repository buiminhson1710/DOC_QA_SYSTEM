"""Đọc cấu hình từ biến môi trường (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

BASE_DIR: Path = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # nạp OPENAI_API_KEY / GEMINI_API_KEY vào os.environ cho litellm


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    docs_dir: Path
    relations_file: Path
    log_file: Path
    log_max_bytes: int
    log_backup_count: int

    local_model_url: str
    local_model_name: str
    security_threshold: float
    security_fail_closed: bool
    security_offline_block_score: int

    qdrant_host: str
    qdrant_port: int
    qdrant_api_key: str
    qdrant_path: str
    qdrant_collection: str

    embedding_model: str
    embedding_dim: int
    generation_model: str
    generation_fallback_models: tuple[str, ...]
    light_model: str
    ocr_model: str
    ocr_engine: str
    ocr_lang: str
    ocr_pages_per_request: int
    ocr_workers: int
    ocr_dpi: int
    ocr_allow_partial: bool
    ocr_cache_dir: Path
    tesseract_cmd: str
    answer_max_tokens: int
    max_context_chars: int
    answer_cache_ttl_seconds: int
    direct_answers: bool
    hybrid_search: bool
    pin_articles: bool
    expand_amendments: bool
    trace_log_text: bool
    embed_batch_size: int
    embed_timeout: float
    embed_max_retries: int
    embed_cache: bool
    embed_cache_path: Path
    ingest_state_file: Path
    lexical_top_k: int
    ocr_min_confidence: float
    trace_file: Path
    ingest_trace_file: Path
    diagnostics_dir: Path

    public_base_url: str
    max_upload_mb: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    amend_top_k: int
    min_relevance_score: float
    cache_ttl_seconds: int
    cache_max_items: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    DATA_DIR = Path(_env("DATA_DIR", str(BASE_DIR)))  # nơi chứa storage/ và logs/ (mặc định: thư mục dự án)
    return Settings(
        base_dir=BASE_DIR,
        docs_dir=DATA_DIR / "storage" / "docs",
        relations_file=DATA_DIR / "storage" / "relations.json",
        log_file=DATA_DIR / "logs" / "app.log",
        log_max_bytes=10 * 1024 * 1024,  # 10MB / file
        log_backup_count=5,              # 5 bản backup
        local_model_url=_env("LOCAL_MODEL_URL", "http://localhost:11434"),
        local_model_name=_env("LOCAL_MODEL_NAME", "qwen2.5:3b"),
        security_threshold=_env_float("SECURITY_THRESHOLD", 0.6),
        security_fail_closed=_env_bool("SECURITY_FAIL_CLOSED", False),
        security_offline_block_score=_env_int("SECURITY_OFFLINE_BLOCK_SCORE", 2),
        qdrant_host=_env("QDRANT_HOST", "localhost"),
        qdrant_port=_env_int("QDRANT_PORT", 6333),
        qdrant_api_key=_env("QDRANT_API_KEY", ""),
        qdrant_path=_env("QDRANT_PATH", ""),
        qdrant_collection=_env("QDRANT_COLLECTION", "legal_docs"),
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=_env_int("EMBEDDING_DIM", 1536),
        generation_model=_env("GENERATION_MODEL", "gpt-4o-mini"),
        light_model=_env("LIGHT_MODEL", ""),
        ocr_model=_env("OCR_MODEL", ""),
        ocr_engine=_env("OCR_ENGINE", "auto").lower(),
        ocr_lang=_env("OCR_LANG", "vie"),
        ocr_pages_per_request=_env_int("OCR_PAGES_PER_REQUEST", 5),
        ocr_workers=_env_int("OCR_WORKERS", 3),
        ocr_dpi=_env_int("OCR_DPI", 200),
        ocr_allow_partial=_env_bool("OCR_ALLOW_PARTIAL", False),
        ocr_cache_dir=DATA_DIR / "storage" / "ocr_cache",
        tesseract_cmd=_env("TESSERACT_CMD", ""),
        answer_max_tokens=_env_int("ANSWER_MAX_TOKENS", 2500),
        max_context_chars=_env_int("MAX_CONTEXT_CHARS", 8000),
        answer_cache_ttl_seconds=_env_int("ANSWER_CACHE_TTL_SECONDS", 86400),
        direct_answers=_env_bool("DIRECT_ANSWERS", True),
        hybrid_search=_env_bool("HYBRID_SEARCH", True),
        pin_articles=_env_bool("PIN_ARTICLES", True),
        expand_amendments=_env_bool("EXPAND_AMENDMENTS", True),
        trace_log_text=_env_bool("TRACE_LOG_TEXT", True),
        embed_batch_size=max(1, _env_int("EMBED_BATCH_SIZE", 64)),
        embed_timeout=_env_float("EMBED_TIMEOUT", 60.0),
        embed_max_retries=max(0, _env_int("EMBED_MAX_RETRIES", 4)),
        embed_cache=_env_bool("EMBED_CACHE", True),
        embed_cache_path=DATA_DIR / "storage" / "embed_cache.sqlite",
        ingest_state_file=DATA_DIR / "storage" / "ingest_state.json",
        lexical_top_k=_env_int("LEXICAL_TOP_K", 6),
        ocr_min_confidence=_env_float("OCR_MIN_CONFIDENCE", 65.0),
        trace_file=DATA_DIR / "logs" / "trace.jsonl",
        ingest_trace_file=DATA_DIR / "logs" / "ingest.jsonl",
        diagnostics_dir=DATA_DIR / "storage" / "diagnostics",
        generation_fallback_models=tuple(
            m.strip() for m in _env("GENERATION_FALLBACK_MODELS", "").split(",") if m.strip()
        ),
        public_base_url=_env("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/"),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 50),
        chunk_size=_env_int("CHUNK_SIZE", 1200),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        top_k=_env_int("TOP_K", 6),
        amend_top_k=_env_int("AMEND_TOP_K", 4),
        min_relevance_score=_env_float("MIN_RELEVANCE_SCORE", 0.0),
        cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 1800),
        cache_max_items=_env_int("CACHE_MAX_ITEMS", 500),
    )


def file_url_for(filename: str) -> str:
    """URL mở/tải trực tiếp file gốc (đã mã hoá ký tự đặc biệt/khoảng trắng)."""
    return f"{get_settings().public_base_url}/api/files/{quote(filename)}"


def _key_var_for(model: str) -> str:
    m = model.lower()
    if m.startswith("gemini/"):
        return "GEMINI_API_KEY"
    if m.startswith(("ollama/", "ollama_chat/")):
        return ""
    if m.startswith(("anthropic/", "claude")):
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"  # litellm mặc định: model không có tiền tố là của OpenAI


def config_warnings() -> list[str]:
    """Phát hiện lỗi cấu hình hay gặp: khoá API là giá trị mẫu / để trống so với model đang chọn."""
    s = get_settings()
    models = [("EMBEDDING_MODEL", s.embedding_model), ("GENERATION_MODEL", s.generation_model)]
    models += [("GENERATION_FALLBACK_MODELS", m) for m in s.generation_fallback_models]
    models += [(n, m) for n, m in (("LIGHT_MODEL", s.light_model), ("OCR_MODEL", s.ocr_model)) if m]
    warnings: list[str] = []
    for label, model in models:
        var = _key_var_for(model)
        if not var:
            continue
        value = os.getenv(var, "").strip()
        if not value:
            problem = "đang để trống"
        elif "xxxx" in value.lower() or value.lower().startswith("your"):
            problem = "đang là khoá mẫu (chưa điền khoá thật)"
        else:
            continue
        msg = f"{label}={model} cần {var} nhưng {problem}. Sửa trong file .env (hoặc đổi {label} sang model bạn có khoá)."
        if msg not in warnings:
            warnings.append(msg)
    return warnings
