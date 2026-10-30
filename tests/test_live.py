"""Test dùng dịch vụ THẬT. Mặc định bị bỏ qua. Bật:  RUN_LIVE_TESTS=1 pytest tests/test_live.py -m live
(đọc khoá từ .env thật; dữ liệu ghi vào Qdrant nhúng tạm, không đụng collection của bạn; tốn một ít token/quota).
RUN_LIVE_E2E=1 để chạy thêm kịch bản đầu-cuối trên kho tổng hợp (tốn nhiều hơn)."""
from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="đặt RUN_LIVE_TESTS=1 và khoá API thật trong .env để chạy"),
]


def test_live_embedding_returns_configured_dimension(cfg):
    from src.services.qdrant_service import QdrantService

    svc = QdrantService(cfg)
    vecs = asyncio.run(svc.embed_texts(["Hạn mức mua sắm trực tiếp", "Thời hạn bảo hành tối thiểu"], use_cache=False))
    assert len(vecs) == 2 and all(len(v) == cfg.embedding_dim for v in vecs), "EMBEDDING_DIM không khớp model thật"
    assert svc.last_embed["tokens_source"] in ("provider", "unavailable")


def test_live_generation_follows_output_schema(cfg):
    from src.services import llm_router
    from src.services.rag_engine import _CHAT_SYSTEM, validate_llm_output

    ctx = "[Nguồn 1] VĂN BẢN 101/2030/NĐ-CP | Điều 5\nHạn mức mua sắm trực tiếp là 5 tỷ đồng."
    resp, _ = asyncio.run(llm_router.acomplete(
        [cfg.generation_model], messages=[{"role": "system", "content": _CHAT_SYSTEM}, {"role": "user", "content": f"<context>\n{ctx}\n</context>\n\nCâu hỏi: Hạn mức là bao nhiêu?"}],
        temperature=0.1, max_tokens=cfg.answer_max_tokens, response_format={"type": "json_object"}, timeout=90))
    out, errors = validate_llm_output(resp.choices[0].message.content)
    assert out is not None, errors
    assert "5 tỷ" in out.answer and 1 in out.used_sources


def test_live_refuses_when_context_lacks_evidence(cfg):
    from src.services import llm_router
    from src.services.rag_engine import _CHAT_SYSTEM, answer_checks, validate_llm_output

    ctx = "[Nguồn 1] VĂN BẢN 101/2030/NĐ-CP | Điều 5\nHạn mức mua sắm trực tiếp là 5 tỷ đồng."
    resp, _ = asyncio.run(llm_router.acomplete(
        [cfg.generation_model], messages=[{"role": "system", "content": _CHAT_SYSTEM}, {"role": "user", "content": f"<context>\n{ctx}\n</context>\n\nCâu hỏi: Mức thuế thu nhập cá nhân là bao nhiêu?"}],
        temperature=0.1, max_tokens=cfg.answer_max_tokens, response_format={"type": "json_object"}, timeout=90))
    out, _ = validate_llm_output(resp.choices[0].message.content)
    assert out is not None and (out.enough_context is False or answer_checks(out.answer, "", [{"index": 1, "content": ctx}])["says_no_basis"])


@pytest.mark.skipif(os.environ.get("RUN_LIVE_E2E") != "1", reason="đặt thêm RUN_LIVE_E2E=1 để chạy đầu-cuối (tốn token)")
def test_live_end_to_end_on_synthetic_corpus(cfg):
    from pathlib import Path

    from src.evaluation import build_corpus
    from src.services.qdrant_service import QdrantService
    from src.services.rag_engine import ChatRequest, RAGEngine
    from src.services.relations_service import RelationsService
    from src.services.security_guard import SecurityGuard

    q = QdrantService(cfg)
    eng = RAGEngine(q, RelationsService(q, cfg), SecurityGuard(cfg), cfg)

    async def go():
        for p in build_corpus(Path(cfg.docs_dir)):
            await eng.ingest_file(p)
        return await eng.chat(ChatRequest(question="Hạn mức mua sắm trực tiếp thiết bị y tế hiện nay là bao nhiêu?"), "live-1")

    resp = asyncio.run(go())
    assert "1,5" in resp.answer and resp.citations
