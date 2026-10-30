"""Test scripts/analyze_trace.py trên trace THẬT do pipeline ghi ra."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

from test_pipeline_integration import AMENDED_Q, ask, ingest_corpus, run
from src.evaluation import DOC_A

_spec = importlib.util.spec_from_file_location("analyze_trace", Path(__file__).resolve().parents[1] / "scripts" / "analyze_trace.py")
AT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AT)


def test_percentiles_and_stats():
    assert AT.pct([1, 2, 3, 4, 100], 0.5) == 3 and AT.pct([], 0.5) is None
    assert AT.stats([10, 20])["mean"] == 15.0


def test_analyzer_summarizes_real_trace(engine, cfg, provider):
    run(ingest_corpus(engine, cfg))
    ask(engine, AMENDED_Q, rid="a1")
    ask(engine, AMENDED_Q, rid="a2")                                                  # cache
    run(engine.chat.__self__.chat(__import__("src.services.rag_engine", fromlist=["ChatRequest"]).ChatRequest(question=f"Điều 5 Nghị định {DOC_A} quy định thế nào?"), "a3"))
    provider.chat_behavior = lambda p, m, msgs: "không phải json"
    ask(engine, "Thời hạn bảo hành tối thiểu của thiết bị y tế là bao lâu?", rid="a4")
    chat = AT.analyze_chat(AT.read_events(cfg.trace_file))
    assert chat["requests"] == 4 and chat["modes"] == {"llm": 2, "cache": 1, "direct": 1}
    assert chat["cache_hit_rate"] == 0.25
    assert chat["tokens"]["usage_from_provider"] == 2 and chat["tokens"]["prompt_total"] == 2000
    assert chat["tokens"]["tokens_per_char (hiệu chỉnh ước lượng)"] > 0
    assert chat["output_quality"]["schema_ok"] == "1/2" and chat["step_ms"]["embed"]["n"] >= 2
    ing = AT.analyze_ingest(AT.read_events(cfg.ingest_trace_file))
    assert ing["done"] == 5 and ing["failed"] == 0 and ing["vectors_vs_points"] == [] and ing["embedding"]["api_texts"] > 0
    json.dumps(AT.render({"chat": chat, "ingest": ing}))
