#!/usr/bin/env python
"""Tóm tắt logs/trace.jsonl và logs/ingest.jsonl: token, độ trễ, truy xuất, chất lượng đầu ra, lỗi, ingest.

Dùng:  python scripts/analyze_trace.py [--logs logs] [--json]
Gửi kết quả (hoặc chính các file .jsonl — đã che khoá) khi cần hỗ trợ chẩn đoán."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_events(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.parent.glob(path.name + "*"), key=lambda p: p.stat().st_mtime)  # gồm cả file xoay vòng .1 .2
    out: list[dict[str, Any]] = []
    for f in files:
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def pct(values: Iterable[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))], 1)


def stats(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": pct(values, 0.5), "p95": pct(values, 0.95), "max": max(values) if values else None,
            "mean": round(statistics.mean(values), 1) if values else None}


def analyze_chat(events: list[dict[str, Any]]) -> dict[str, Any]:
    chats = [e for e in events if e.get("kind") == "chat"]
    errors = [e for e in events if e.get("kind") == "chat_error"]
    by_mode = Counter(e.get("mode") for e in chats)
    lat: dict[str, list[float]] = {}
    steps: dict[str, list[float]] = {}
    for e in chats:
        lat.setdefault(e.get("mode", "?"), []).append(e.get("total_ms", 0))
        for step, ms in (e.get("timings_ms") or {}).items():
            steps.setdefault(step, []).append(ms)
    llm = [e for e in chats if e.get("mode") == "llm"]
    provider = [e for e in llm if e.get("usage_source") == "provider"]
    ratios = []
    for e in provider:
        pr = e.get("prompt") or {}
        chars = pr.get("system_chars", 0) + pr.get("context_chars", 0) + pr.get("question_chars", 0)
        if chars and e["usage"]["prompt_tokens"]:
            ratios.append(e["usage"]["prompt_tokens"] / chars)
    fmt = [e.get("format") or {} for e in llm]
    checks = [e.get("checks") or {} for e in llm]
    flag = lambda key: sum(1 for c in checks if c.get(key))  # noqa: E731
    scores = [e["retrieval"]["top_vector_score"] for e in chats if (e.get("retrieval") or {}).get("top_vector_score") is not None]
    lexical_only = sum(1 for e in chats for s in (e.get("sources") or []) if s.get("via") == "lexical")
    total_sources = sum(len(e.get("sources") or []) for e in chats)
    retries = [e for e in llm if (e.get("llm") or {}).get("attempts", 1) > 1]
    return {
        "requests": len(chats), "errors": dict(Counter(f"{e.get('error_type')}:{e.get('error_kind')}" for e in errors)), "modes": dict(by_mode),
        "cache_hit_rate": round(by_mode.get("cache", 0) / len(chats), 3) if chats else None,
        "latency_ms_by_mode": {m: stats(v) for m, v in lat.items()}, "step_ms": {k: stats(v) for k, v in steps.items()},
        "tokens": {
            "llm_requests": len(llm), "usage_from_provider": len(provider), "usage_unavailable": len(llm) - len(provider),
            "prompt_total": sum(e["usage"]["prompt_tokens"] for e in provider), "completion_total": sum(e["usage"]["completion_tokens"] for e in provider),
            "prompt_per_request": stats([e["usage"]["prompt_tokens"] for e in provider]),
            "completion_per_request": stats([e["usage"]["completion_tokens"] for e in provider]),
            "tokens_per_char (hiệu chỉnh ước lượng)": round(statistics.median(ratios), 3) if ratios else None,
            "est_context_tokens": stats([((e.get("prompt") or {}).get("est_tokens") or {}).get("context", 0) for e in llm]),
        },
        "retrieval": {"top_vector_score": stats(scores), "lexical_only_sources": f"{lexical_only}/{total_sources}",
                      "dropped_by_budget_requests": sum(1 for e in chats if (e.get("retrieval") or {}).get("dropped_by_budget", 0) > 0)},
        "output_quality": {
            "schema_ok": f"{sum(1 for f in fmt if f.get('schema_ok'))}/{len(fmt)}", "finish_reason_length (cắt cụt)": sum(1 for f in fmt if f.get("finish_reason") == "length"),
            "enough_context_false": sum(1 for f in fmt if f.get("enough_context") is False),
            "ungrounded_numbers": flag("ungrounded_numbers"), "unsupported_documents": flag("unsupported_documents"),
            "unsupported_articles": flag("unsupported_articles"), "unsupported_pages": flag("unsupported_pages"),
            "unsupported_status_claims": flag("unsupported_status_claims"), "invalid_citations": flag("invalid_citations"),
            "says_no_basis": flag("says_no_basis"),
        },
        "llm_retries": {"requests_with_retry": len(retries), "error_kinds": dict(Counter(x["kind"] for e in retries for x in (e.get("llm") or {}).get("errors", [])))},
    }


def analyze_ingest(events: list[dict[str, Any]]) -> dict[str, Any]:
    done = [e for e in events if e.get("kind") == "ingest_done"]
    failed = [e for e in events if e.get("kind") == "ingest_failed"]
    return {
        "done": len(done), "failed": len(failed), "status": dict(Counter(e.get("status") for e in done)),
        "failed_by_stage_kind": dict(Counter(f"{e.get('stage')}:{e.get('error_kind')}" for e in failed)),
        "resumable_failures": sum(1 for e in failed if e.get("resumable")),
        "warnings": dict(Counter(w.split(":")[0].split(" ")[0] + " " + " ".join(w.split()[1:3]) for e in done for w in e.get("warnings", []))),
        "embedding": {"retries": sum((e.get("embedding") or {}).get("retries", 0) for e in done),
                      "from_cache_texts": sum((e.get("embedding") or {}).get("from_cache", 0) for e in done),
                      "api_texts": sum((e.get("embedding") or {}).get("api_texts", 0) for e in done)},
        "vectors_vs_points": [{"file": e.get("filename"), "chunks": e.get("chunks"), "vectors": e.get("vectors"), "qdrant_points": e.get("qdrant_points")}
                              for e in done if e.get("chunks") != e.get("qdrant_points") or e.get("chunks") != e.get("vectors")],
        "total_ms": stats([e.get("total_ms", 0) for e in done]),
    }


def render(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--json", action="store_true", help="in JSON thuần")
    args = ap.parse_args()
    logs = Path(args.logs)
    report = {"chat": analyze_chat(read_events(logs / "trace.jsonl")), "ingest": analyze_ingest(read_events(logs / "ingest.jsonl"))}
    print(render(report))


if __name__ == "__main__":
    main()
