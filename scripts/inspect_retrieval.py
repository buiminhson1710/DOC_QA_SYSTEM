#!/usr/bin/env python3
"""Công cụ kiểm tra truy xuất (Retrieval & Chunking inspection) - Không tốn token sinh câu trả lời AI.

Dùng:
  python scripts/inspect_retrieval.py "các trường hợp được chỉ định thầu"
  python scripts/inspect_retrieval.py "Điều 78" --top-k 6
  python scripts/inspect_retrieval.py "quy trình chỉ định thầu rút gọn" --api http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def query_api(base_url: str, question: str, top_k: int | None = None) -> dict:
    url = f"{base_url.rstrip('/')}/api/diagnostics/search"
    payload = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def query_local(question: str, top_k: int | None = None) -> dict:
    from src.config import get_settings
    from src.services.qdrant_service import QdrantService
    from src.services.relations_service import RelationsService
    from src.services.security_guard import SecurityGuard
    from src.services.rag_engine import RAGEngine

    settings = get_settings()
    qdrant = QdrantService(settings)
    relations = RelationsService(qdrant, settings)
    guard = SecurityGuard(settings)
    engine = RAGEngine(qdrant, relations, guard, settings)
    return await engine.debug_search(question, top_k=top_k)


def print_result(data: dict) -> None:
    print("=" * 80)
    print(f"CÂU HỎI KIỂM TRA: {data.get('question')}")
    print("=" * 80)

    timings = data.get("timings_ms", {})
    print(f"⏱️  Thời gian xử lý: {sum(timings.values()):.0f} ms | Chi tiết: {timings}")

    retrieval = data.get("retrieval", {})
    print(f"\n📊 1. KẾT QUẢ TÌM KIẾM VECTOR (Top score: {retrieval.get('top_vector_score')})")
    for v in retrieval.get("vector", []):
        print(f"   [{v.get('rank')}] Score: {v.get('score')} | {v.get('doc')} | {v.get('article')} | Trang {v.get('page')}")

    print(f"\n🔍 2. KẾT QUẢ TỪ KHOÁ BM25")
    for lex in retrieval.get("lexical", []):
        print(f"   [{lex.get('rank')}] BM25: {lex.get('bm25')} | {lex.get('doc')} | {lex.get('article')} | Trang {lex.get('page')}")

    print(f"\n⚡ 3. KẾT HỢP RRF (Reciprocal Rank Fusion)")
    for f in retrieval.get("fused", []):
        print(f"   [{f.get('rank')}] RRF: {f.get('rrf')} | Nhánh: {f.get('via')}")

    sources = data.get("sources", [])
    print(f"\n📦 4. CÁC NGUỒN CHÍNH THỨC ĐƯỢC CHỌN CHO AI ({len(sources)} nguồn - Tổng {data.get('context_chars', 0)} ký tự)")
    for s in sources:
        amend_tag = " [VĂN BẢN SỬA ĐỔI]" if s.get("amendment") else ""
        print(f"\n--- [Nguồn {s.get('i')}]{amend_tag} {s.get('doc')} | {s.get('article')} | Trang {s.get('page')} ({s.get('chars')} ký tự) ---")

    print("\n📝 5. NỘI DUNG TỪNG CHUNK ĐÃ TRÍCH XUẤT:")
    for hit in data.get("retrieval_hits", []):
        print(f"\n--- Chunk #{hit.get('rank')}: {hit.get('doc')} | {hit.get('article')} | Trang {hit.get('page')} ---")
        preview_text = hit.get("content", "").strip()
        if len(preview_text) > 400:
            preview_text = preview_text[:400] + "...\n[xem tiếp trong tài liệu]"
        print(preview_text)

    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra kết quả retrieval và chunking.")
    parser.add_argument("question", help="Câu hỏi cần kiểm tra (ví dụ: 'các trường hợp chỉ định thầu')")
    parser.add_argument("--top-k", type=int, default=4, help="Số lượng nguồn lấy")
    parser.add_argument("--api", default="", help="URL API nếu server đang chạy (ví dụ: http://localhost:8000)")
    args = parser.parse_args()

    if args.api:
        try:
            data = query_api(args.api, args.question, args.top_k)
            print_result(data)
        except Exception as e:
            print(f"Lỗi khi gọi API: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            data = asyncio.run(query_local(args.question, args.top_k))
            print_result(data)
        except Exception as e:
            print(f"Lỗi khi chạy local: {e}", file=sys.stderr)
            print("Gợi ý: Nếu server FastAPI đang chạy, hãy dùng cờ: --api http://localhost:8000")
            sys.exit(1)


if __name__ == "__main__":
    main()
