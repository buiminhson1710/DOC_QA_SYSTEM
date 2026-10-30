"""Quản lý collection Qdrant, embedding (litellm) và tìm kiếm vector."""
from __future__ import annotations

import asyncio
import hashlib
import math
import random
import threading
import time
import uuid
from array import array
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import litellm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config import Settings, get_settings
from src.services.logger_service import get_logger

logger = get_logger("qdrant")
litellm.drop_params = True  # bỏ qua tham số mà provider không hỗ trợ


class EmbeddingError(Exception):
    """Lỗi gọi API embedding. `kind`: quota_daily | rate_limit | overloaded | auth | input | config | invalid_output | error."""

    def __init__(self, message: str, kind: str = "error", retryable: bool = False) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0, wait: float = 0.0, rng: Optional[random.Random] = None) -> float:
    """Exponential backoff + 'equal jitter': d = min(cap, base*2^(attempt-1)); trả về giá trị trong [d/2, d] (>= wait nếu 429 ngắn hạn)."""
    rng = rng or random
    d = min(cap, base * (2 ** (attempt - 1)))
    return max(wait, d / 2 + rng.uniform(0, d / 2))


def classify_embedding_error(exc: Exception) -> str:
    """Phân biệt: quota_daily (không retry), rate_limit/overloaded (retry có giới hạn), auth, input, config."""
    from src.services import llm_router

    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, litellm.AuthenticationError) or any(k in low for k in ("401", "invalid_api_key", "incorrect api key", "api key not valid", "permission_denied")):
        return "auth"
    plan = llm_router.classify(exc)
    if plan.kind in ("quota_daily", "rate_limit", "overloaded"):
        return plan.kind
    if isinstance(exc, litellm.NotFoundError) or "404" in msg or "not found" in low:
        return "config"
    if isinstance(exc, (litellm.BadRequestError, litellm.ContextWindowExceededError)) or "400" in msg or "invalid argument" in low:
        return "input"
    return "error"


class EmbeddingCache:
    """Cache vector theo (model, dim, nội dung) bằng sqlite: ingest dở có thể TIẾP TỤC mà không gọi lại API / không tạo dữ liệu trùng."""

    def __init__(self, path: Any) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        import sqlite3

        pathlib_path = __import__("pathlib").Path(self._path)
        pathlib_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, dim INTEGER, v BLOB)")
        self._db.commit()

    @staticmethod
    def key(model: str, dim: int, text: str) -> str:
        return hashlib.sha1(f"{model}|{dim}|{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: list[str], dim: int) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i : i + 500]
                rows = self._db.execute(f"SELECT k, dim, v FROM emb WHERE k IN ({','.join('?' * len(chunk))})", chunk).fetchall()
                for k, d, blob in rows:
                    if d == dim:
                        arr = array("f")
                        arr.frombytes(blob)
                        out[k] = list(arr)
        return out

    def put_many(self, items: dict[str, list[float]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO emb (k, dim, v) VALUES (?, ?, ?)",
                [(k, len(v), array("f", v).tobytes()) for k, v in items.items()],
            )
            self._db.commit()


@dataclass
class Hit:
    id: str
    score: float
    payload: dict[str, Any]
    fused: float = 0.0  # điểm sau khi trộn vector + từ khoá (RRF); 0 nếu không dùng


def _point_id(filename: str, chunk_index: int) -> str:
    """ID xác định -> nạp lại cùng file sẽ ghi đè thay vì nhân đôi."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{chunk_index}"))


def _match_any(key: str, values: Sequence[str]) -> qm.FieldCondition:
    return qm.FieldCondition(key=key, match=qm.MatchAny(any=list(values)))


def _match(key: str, value: str) -> qm.FieldCondition:
    return qm.FieldCondition(key=key, match=qm.MatchValue(value=value))


class QdrantService:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._s = settings or get_settings()
        if self._s.qdrant_path:
            self.client = QdrantClient(path=self._s.qdrant_path)
        else:
            self.client = QdrantClient(
                host=self._s.qdrant_host,
                port=self._s.qdrant_port,
                api_key=self._s.qdrant_api_key or None,
                timeout=30,
            )
        self._ready = False
        self._lock = threading.Lock()
        self.embed_cache: Optional[EmbeddingCache] = EmbeddingCache(self._s.embed_cache_path) if self._s.embed_cache else None
        self.last_embed: dict[str, Any] = {}  # số liệu lần gọi embedding gần nhất (để đo/ghi trace)

    # ------------------------------------------------------------------ setup
    def ensure_collection(self) -> None:
        """Tạo collection + payload index nếu chưa có (lazy, gọi lại an toàn)."""
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            name = self._s.qdrant_collection
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=qm.VectorParams(size=self._s.embedding_dim, distance=qm.Distance.COSINE),
                )
                logger.info("Created Qdrant collection '%s' (dim=%d)", name, self._s.embedding_dim)
            else:
                try:
                    vectors = self.client.get_collection(name).config.params.vectors
                    size = getattr(vectors, "size", None)
                    if size and size != self._s.embedding_dim:
                        logger.error(
                            "Collection '%s' có dim=%s nhưng EMBEDDING_DIM=%d -> hãy đổi tên collection hoặc dim",
                            name, size, self._s.embedding_dim,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Không kiểm tra được dim của collection: %s", exc)
            for field_name in ("doc_number", "filename"):
                try:
                    self.client.create_payload_index(
                        collection_name=name, field_name=field_name, field_schema=qm.PayloadSchemaType.KEYWORD
                    )
                except Exception as exc:  # noqa: BLE001 (index đã tồn tại / local mode)
                    logger.debug("Payload index '%s': %s", field_name, exc)
            try:
                self.client.create_payload_index(collection_name=name, field_name="chunk_index", field_schema=qm.PayloadSchemaType.INTEGER)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Payload index 'chunk_index': %s", exc)
            self._ready = True

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    # -------------------------------------------------------------- embedding
    async def _embed_batch(self, batch: list[str], stats: dict[str, Any]) -> list[list[float]]:
        """Gọi API cho một batch. Retry CÓ GIỚI HẠN (backoff + jitter) chỉ với lỗi tạm thời; lỗi quota ngày/auth/input không retry."""
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await litellm.aembedding(
                    model=self._s.embedding_model, input=batch, dimensions=self._s.embedding_dim,
                    timeout=self._s.embed_timeout, num_retries=0,
                )
                vecs = [item["embedding"] for item in resp.data]
                if len(vecs) != len(batch):
                    raise EmbeddingError(f"API trả {len(vecs)} vector cho {len(batch)} đầu vào", kind="invalid_output")
                stats["calls"] += 1
                stats["tokens"] += int(getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0)
                return vecs
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001
                kind = classify_embedding_error(exc)
                stats["errors"].append({"kind": kind, "attempt": attempt, "batch": len(batch), "msg": str(exc)[:160]})
                logger.warning("Embedding lỗi [%s] lần %d (batch=%d): %s", kind, attempt, len(batch), str(exc)[:140])
                if kind == "quota_daily":
                    raise EmbeddingError(
                        f"Hết hạn mức embedding theo ngày của '{self._s.embedding_model}' (không retry vì chờ không giúp ích). "
                        "Vector đã tính được đã lưu cache: chạy lại sau khi quota làm mới hoặc đổi model/billing — hệ thống chỉ embed phần còn thiếu.",
                        kind=kind,
                    ) from exc
                if kind == "auth":
                    raise EmbeddingError(
                        f"Khoá API không hợp lệ cho EMBEDDING_MODEL='{self._s.embedding_model}'. Kiểm tra .env: khoá đã điền thật chưa "
                        "và model có đúng nhà cung cấp không (model Gemini phải có tiền tố 'gemini/').", kind=kind,
                    ) from exc
                if kind == "config":
                    raise EmbeddingError(f"Model embedding '{self._s.embedding_model}' không tồn tại/không được hỗ trợ: {str(exc)[:160]}", kind=kind) from exc
                if kind == "input":
                    if len(batch) > 1:  # cô lập đầu vào gây lỗi bằng cách gọi từng văn bản
                        stats["isolated_batches"] = stats.get("isolated_batches", 0) + 1
                        out: list[list[float]] = []
                        for text in batch:
                            out.extend(await self._embed_batch([text], stats))
                        return out
                    raise EmbeddingError(f"Đầu vào không hợp lệ cho embedding (độ dài {len(batch[0])} ký tự): {str(exc)[:160]}", kind=kind) from exc
                if kind == "error":
                    raise EmbeddingError(f"Lỗi embedding không xác định: {str(exc)[:200]}", kind=kind) from exc
                # rate_limit (429 ngắn hạn) / overloaded (503, timeout, mạng): retry có giới hạn
                if attempt > self._s.embed_max_retries:
                    raise EmbeddingError(f"Embedding vẫn lỗi [{kind}] sau {attempt} lần thử: {str(exc)[:160]}", kind=kind, retryable=True) from exc
                from src.services import llm_router
                stats["retries"] += 1
                await asyncio.sleep(backoff_delay(attempt, wait=llm_router.classify(exc).wait if kind == "rate_limit" else 0.0))

    async def embed_texts(self, texts: list[str], batch_size: Optional[int] = None, use_cache: bool = True) -> list[list[float]]:
        """Embed danh sách văn bản. Kiểm tra số lượng/dimension/giá trị hợp lệ; cache từng batch để có thể tiếp tục khi dừng giữa chừng."""
        dim = self._s.embedding_dim
        bs = batch_size or self._s.embed_batch_size
        t0 = time.perf_counter()
        stats: dict[str, Any] = {"calls": 0, "tokens": 0, "retries": 0, "errors": []}
        vectors: list[Optional[list[float]]] = [None] * len(texts)

        cache = self.embed_cache if (use_cache and self._s.embed_cache) else None
        keys = [EmbeddingCache.key(self._s.embedding_model, dim, t) for t in texts]
        hits = cache.get_many(list(dict.fromkeys(keys)), dim) if cache else {}
        for i, k in enumerate(keys):
            if k in hits:
                vectors[i] = hits[k]
        missing = [i for i, v in enumerate(vectors) if v is None]

        for start in range(0, len(missing), bs):
            idxs = missing[start : start + bs]
            vecs = await self._embed_batch([texts[i] for i in idxs], stats)
            new = {}
            for i, v in zip(idxs, vecs):
                vectors[i] = v
                new[keys[i]] = v
            if cache:
                cache.put_many({k: v for k, v in new.items() if len(v) == dim})  # lưu ngay sau mỗi batch -> tiếp tục được

        out: list[list[float]] = []
        for idx, vec in enumerate(vectors):
            if vec is None or len(vec) != dim:
                raise EmbeddingError(
                    f"Vector #{idx} có {0 if vec is None else len(vec)} chiều nhưng EMBEDDING_DIM={dim}. Đặt EMBEDDING_DIM đúng với model "
                    f"'{self._s.embedding_model}' và dùng QDRANT_COLLECTION khớp số chiều.", kind="invalid_output",
                )
            if not all(math.isfinite(x) for x in vec) or not any(vec):
                raise EmbeddingError(f"Vector #{idx} không hợp lệ (toàn 0 hoặc chứa NaN): embedding trả dữ liệu hỏng.", kind="invalid_output")
            out.append(vec)
        self.last_embed = {
            "model": self._s.embedding_model, "dim": dim, "inputs": len(texts), "outputs": len(out), "batch_size": bs,
            "from_cache": len(texts) - len(missing), "api_texts": len(missing), "calls": stats["calls"], "retries": stats["retries"],
            "errors": stats["errors"][-5:], "chars": sum(len(t) for t in texts),
            "prompt_tokens": stats["tokens"] or None, "tokens_source": "provider" if stats["tokens"] else "unavailable",
            "ms": round((time.perf_counter() - t0) * 1000),
        }
        return out

    # ------------------------------------------------------------------ write
    def delete_by_filename(self, filename: str) -> None:
        self.ensure_collection()
        self.client.delete(
            collection_name=self._s.qdrant_collection,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=[_match("filename", filename)])),
            wait=True,
        )

    def replace_document(
        self, filename: str, payloads: list[dict[str, Any]], vectors: list[list[float]]
    ) -> int:
        """Upsert chunk mới (ID xác định => ghi đè, không trùng) RỒI mới xoá chunk thừa của lần nạp cũ (chunk_index >= số chunk mới)."""
        self.ensure_collection()
        points = [
            qm.PointStruct(id=_point_id(filename, p["chunk_index"]), vector=v, payload=p)
            for p, v in zip(payloads, vectors)
        ]
        for i in range(0, len(points), 100):
            self.client.upsert(collection_name=self._s.qdrant_collection, points=points[i : i + 100], wait=True)
        self.client.delete(
            collection_name=self._s.qdrant_collection,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                _match("filename", filename),
                qm.FieldCondition(key="chunk_index", range=qm.Range(gte=len(points))),
            ])),
            wait=True,
        )
        return len(points)

    def set_doc_payload(self, doc_number: str, payload: dict[str, Any]) -> None:
        """Cập nhật metadata (amended_by, status, ...) cho MỌI chunk của một văn bản."""
        self.ensure_collection()
        self.client.set_payload(
            collection_name=self._s.qdrant_collection,
            payload=payload,
            points=qm.Filter(must=[_match("doc_number", doc_number)]),
            wait=True,
        )

    # ------------------------------------------------------------------- read
    def search(
        self, vector: list[float], limit: int, doc_numbers: Optional[Sequence[str]] = None
    ) -> list[Hit]:
        self.ensure_collection()
        flt = qm.Filter(must=[_match_any("doc_number", doc_numbers)]) if doc_numbers else None
        result = self.client.query_points(
            collection_name=self._s.qdrant_collection,
            query=vector,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return [Hit(id=str(p.id), score=float(p.score), payload=dict(p.payload or {})) for p in result.points]

    def scroll_doc(self, doc_number: str) -> list[Hit]:
        """Lấy TOÀN BỘ chunk của một văn bản (chỉ payload, không vector), sắp theo chunk_index. Không tốn embedding/LLM."""
        self.ensure_collection()
        flt = qm.Filter(must=[_match("doc_number", doc_number)])
        out: list[Hit] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self._s.qdrant_collection, scroll_filter=flt, limit=256,
                offset=offset, with_payload=True, with_vectors=False,
            )
            out.extend(Hit(id=str(pt.id), score=0.0, payload=dict(pt.payload or {})) for pt in points)
            if offset is None:
                break
        out.sort(key=lambda h: h.payload.get("chunk_index", 0))
        return out

    def count_filename(self, filename: str) -> int:
        """Số điểm thực tế trong Qdrant của một file (để kiểm chứng đã lưu đủ chunk)."""
        self.ensure_collection()
        return self.client.count(
            collection_name=self._s.qdrant_collection,
            count_filter=qm.Filter(must=[_match("filename", filename)]), exact=True,
        ).count

    def scroll_all(self) -> list[Hit]:
        """Toàn bộ chunk (payload, không vector) — dùng dựng chỉ mục từ khoá BM25."""
        self.ensure_collection()
        out: list[Hit] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self._s.qdrant_collection, limit=512, offset=offset, with_payload=True, with_vectors=False,
            )
            out.extend(Hit(id=str(pt.id), score=0.0, payload=dict(pt.payload or {})) for pt in points)
            if offset is None:
                break
        return out

    def count(self) -> int:
        self.ensure_collection()
        return self.client.count(collection_name=self._s.qdrant_collection, exact=True).count
