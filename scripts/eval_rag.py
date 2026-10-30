#!/usr/bin/env python
"""Đánh giá RAG.

  python scripts/eval_rag.py inprocess [--out eval]          # kho tổng hợp + embedding GIẢ LẬP, không cần khoá API
  python scripts/eval_rag.py export-corpus --dir tmp_corpus  # xuất kho tổng hợp (DOCX) để nạp vào một collection thử riêng
  python scripts/eval_rag.py live --url http://localhost:8000 --golden eval/golden_real_template.json [--out eval]
      # đo trên máy chủ thật (embedding + LLM thật): Recall@K/MRR, đúng/sai theo từ khoá, citation, groundedness (heuristic),
      # từ chối khi thiếu dữ liệu, token và độ trễ. TỐN token/quota của bạn.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import GOLDEN, KS, build_corpus, format_markdown, run_inprocess, score_question  # noqa: E402
from src.services.direct_answer import fold  # noqa: E402


def http(method: str, url: str, body: dict | None = None, timeout: int = 180) -> tuple[Any, dict[str, str], float]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "null"), dict(resp.headers), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        return {"http_status": exc.code, "body": exc.read().decode()[:500]}, dict(exc.headers), (time.perf_counter() - t0) * 1000


def pctl(vals: list[float], q: float) -> float | None:
    vals = sorted(vals)
    return round(vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))], 1) if vals else None


def run_live(url: str, golden: list[dict[str, Any]], top_k: int = 6) -> dict[str, Any]:
    rows, lat, toks = [], [], []
    trace, _, _ = http("GET", f"{url}/api/diagnostics/trace?limit=500&kind=chat")
    for q in golden:
        row: dict[str, Any] = {"id": q["id"], "category": q.get("category")}
        dbg, _, _ = http("POST", f"{url}/api/diagnostics/search", {"question": q["q"], "top_k": top_k})
        if "retrieval_hits" in dbg:
            ranked = [h["content"] for h in dbg["retrieval_hits"]]
            sources = [{"content": s["content"], "is_amendment": s["amendment"]} for s in dbg["source_texts"]]
            row.update(score_question(q, sources, ranked))
            row["top_vector_score"] = dbg["retrieval"]["top_vector_score"]
        resp, headers, ms = http("POST", f"{url}/api/chat", {"question": q["q"], "top_k": top_k})
        lat.append(ms)
        if "answer" not in resp:
            row["error"] = resp
            rows.append(row)
            continue
        answer_f = fold(resp["answer"])
        row["mode"] = resp.get("mode")
        row["tokens"] = resp["usage"]["prompt_tokens"] + resp["usage"]["completion_tokens"]
        toks.append(row["tokens"])
        cited_docs = {c["doc_number"] for c in resp.get("citations", [])}
        if q.get("answerable", True):
            inc = [fold(x) for x in q.get("answer_includes", [])]
            row["answer_includes_ok"] = all(x in answer_f for x in inc) if inc else None
            stale = [fold(x) for x in q.get("stale_answer_values", [])]
            row["stale_only"] = bool(stale and any(x in answer_f for x in stale) and inc and not all(x in answer_f for x in inc))
            exp_docs = set(q.get("expect_docs", []))
            row["citation_doc_ok"] = (exp_docs <= cited_docs) if exp_docs else None
        row["citations_have_links"] = all(c.get("file_url") for c in resp.get("citations", [])) if resp.get("citations") else None
        # kiểm tra tự động ghi trong trace (heuristic) theo request_id
        rid = headers.get("X-Request-ID") or headers.get("x-request-id")
        ev = next((e for e in reversed(http("GET", f"{url}/api/diagnostics/trace?limit=500&kind=chat")[0].get("events", [])) if e.get("request_id") == rid), None)
        if ev and ev.get("checks") is not None:
            c = ev["checks"]
            row["grounded_heuristic"] = not (c.get("ungrounded_numbers") or c.get("unsupported_documents") or c.get("unsupported_articles") or c.get("unsupported_pages") or c.get("invalid_citations"))
            row["says_no_basis"] = bool(c.get("says_no_basis")) or (ev.get("format") or {}).get("enough_context") is False
        if not q.get("answerable", True):
            row["refused"] = row.get("says_no_basis", False) or "chưa đủ căn cứ" in answer_f or "khong tim thay can cu" in answer_f or "khong lien quan" in answer_f
        rows.append(row)

    def rate(key: str, subset: list[dict[str, Any]]) -> Any:
        vals = [r[key] for r in subset if r.get(key) is not None]
        return f"{sum(1 for v in vals if v)}/{len(vals)}" if vals else "n/a"

    ans = [r for r in rows if r.get("category") != "no_data" and "error" not in r]
    nod = [r for r in rows if r.get("category") == "no_data"]
    return {
        "retrieval": {f"recall@{k}": round(statistics.mean([r[f"recall@{k}"] for r in rows if f"recall@{k}" in r]), 3) for k in KS if any(f"recall@{k}" in r for r in rows)}
        | {"mrr": round(statistics.mean([r["mrr"] for r in rows if "mrr" in r]), 3) if any("mrr" in r for r in rows) else None},
        "answers": {"answer_includes_ok": rate("answer_includes_ok", ans), "stale_only (lỗi nghiêm trọng)": rate("stale_only", ans),
                    "citation_doc_ok": rate("citation_doc_ok", ans), "citations_have_links": rate("citations_have_links", rows),
                    "grounded_heuristic": rate("grounded_heuristic", rows), "false_refusal": rate("says_no_basis", ans),
                    "refusal_when_no_data": rate("refused", nod)},
        "cost": {"tokens_per_question": {"p50": pctl(toks, 0.5), "p95": pctl(toks, 0.95), "total": sum(toks)},
                 "latency_ms": {"p50": pctl(lat, 0.5), "p95": pctl(lat, 0.95)}},
        "errors": [r for r in rows if "error" in r], "per_question": rows,
        "caveat": "Từ khoá/heuristic không phải bằng chứng đúng về pháp lý; hãy đọc lại các câu trả lời quan trọng với văn bản gốc.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inprocess"); a.add_argument("--out", default="eval"); a.add_argument("--top-k", type=int, default=6)
    b = sub.add_parser("export-corpus"); b.add_argument("--dir", required=True)
    c = sub.add_parser("live"); c.add_argument("--url", default="http://localhost:8000"); c.add_argument("--golden", required=True)
    c.add_argument("--out", default="eval"); c.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()
    if args.cmd == "export-corpus":
        for p in build_corpus(Path(args.dir)):
            print(p)
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "inprocess":
        report = asyncio.run(run_inprocess(top_k=args.top_k))
        (out / "report_inprocess.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "report_inprocess.md").write_text(format_markdown(report), encoding="utf-8")
        print(format_markdown(report))
    else:
        golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        golden = [g for g in golden if not str(g.get("id", "")).startswith("_")]
        report = run_live(args.url, golden, args.top_k)
        (out / "report_live.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
