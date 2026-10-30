"""Fixture dùng chung. Mọi test chạy KÍN: không đọc .env của người dùng, không gọi API thật, dữ liệu nằm trong thư mục tạm."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Đặt TRƯỚC khi import src.config (load_dotenv không ghi đè biến đã có) => không phụ thuộc .env thật
if os.environ.get("RUN_LIVE_TESTS") != "1":   # chế độ live dùng cấu hình thật trong .env (vẫn ghi dữ liệu vào Qdrant nhúng tạm, không đụng collection thật)
  os.environ.update({
      "EMBEDDING_MODEL": "gemini/test-embedding", "EMBEDDING_DIM": "64", "GENERATION_MODEL": "gemini/test-main",
      "GENERATION_FALLBACK_MODELS": "gemini/test-fallback", "LIGHT_MODEL": "", "OCR_MODEL": "", "GEMINI_API_KEY": "test-key-not-real-0000000000",
      "OPENAI_API_KEY": "", "OCR_ENGINE": "llm", "LOCAL_MODEL_URL": "http://127.0.0.1:9", "SECURITY_FAIL_CLOSED": "false",
      "QDRANT_COLLECTION": "legal_docs_test", "EMBED_MAX_RETRIES": "3", "EMBED_CACHE": "true", "MIN_RELEVANCE_SCORE": "0",
      "HYBRID_SEARCH": "true", "PIN_ARTICLES": "true", "EXPAND_AMENDMENTS": "true", "DIRECT_ANSWERS": "true",
      "TRACE_LOG_TEXT": "true", "ANSWER_CACHE_TTL_SECONDS": "3600", "OCR_PAGES_PER_REQUEST": "5", "OCR_WORKERS": "2",
      "PUBLIC_BASE_URL": "http://test.local",
})


def reset_global_state() -> None:
    from src import config
    from src.services import document_parser, llm_router, trace

    config.get_settings.cache_clear()
    llm_router.reset()
    document_parser._tesseract_check.cache_clear()
    for lg in list(trace._loggers.values()):
        for h in list(lg.handlers):
            h.close()
            lg.removeHandler(h)
    trace._loggers.clear()


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Settings trỏ vào thư mục tạm (DATA_DIR) + Qdrant nhúng riêng cho test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    reset_global_state()
    from src import config

    yield config.get_settings()
    reset_global_state()


class FakeProvider:
    """Thay litellm.aembedding / litellm.acompletion: ghi lại mọi lời gọi và cho phép giả lập lỗi từng bước."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.embed_calls: list[list[str]] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.sleeps: list[float] = []
        self.embed_behavior: Optional[Callable[["FakeProvider", list[str]], Any]] = None
        self.chat_behavior: Optional[Callable[["FakeProvider", str, list[dict[str, Any]]], Any]] = None
        self.report_usage = True

    async def aembedding(self, model: str, input: list[str], dimensions: Optional[int] = None, **kw: Any) -> Any:  # noqa: A002
        from src.evaluation import proxy_vector

        self.embed_calls.append(list(input))
        if self.embed_behavior:
            outcome = self.embed_behavior(self, input)
            if isinstance(outcome, Exception):
                raise outcome
            if outcome is not None:
                return outcome
        return SimpleNamespace(
            data=[{"embedding": proxy_vector(t, self.dim), "index": i} for i, t in enumerate(input)],
            usage=SimpleNamespace(prompt_tokens=sum(max(1, len(t) // 3) for t in input)),
        )

    @staticmethod
    def reply(content: str, prompt_tokens: Optional[int] = 1000, completion_tokens: int = 60, finish: str = "stop") -> Any:
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens) if prompt_tokens is not None else None
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)], usage=usage)

    async def acompletion(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        if str(model).startswith("ollama"):
            raise ConnectionError("ollama không chạy (giả lập)")
        self.chat_calls.append({"model": model, "messages": messages, "kwargs": kw})
        if self.chat_behavior:
            outcome = self.chat_behavior(self, model, messages)
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, str):
                return self.reply(outcome, prompt_tokens=1000 if self.report_usage else None)
            return outcome
        return self.reply('{"answer": "Theo [1] quy định như trong nguồn.", "used_sources": [1], "enough_context": true}',
                          prompt_tokens=1000 if self.report_usage else None)


@pytest.fixture
def provider(cfg, monkeypatch) -> FakeProvider:
    import litellm

    fp = FakeProvider(cfg.embedding_dim)

    async def fake_sleep(seconds: float) -> None:
        fp.sleeps.append(seconds)

    monkeypatch.setattr(litellm, "aembedding", fp.aembedding)
    monkeypatch.setattr(litellm, "acompletion", fp.acompletion)
    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    return fp


@pytest.fixture
def qdrant(cfg):
    from src.services.qdrant_service import QdrantService

    svc = QdrantService(cfg)
    yield svc
    try:
        svc.client.close()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def engine(cfg, qdrant, provider):
    from src.services.rag_engine import RAGEngine
    from src.services.relations_service import RelationsService
    from src.services.security_guard import SecurityGuard

    rel = RelationsService(qdrant, cfg)
    return RAGEngine(qdrant, rel, SecurityGuard(cfg), cfg)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
