"""Đồ thị quan hệ pháp lý: sửa đổi / thay thế / bãi bỏ, liên kết 2 chiều.

Nguồn sự thật: mỗi văn bản lưu danh sách quan hệ "đi ra" (outgoing). Các trường 2 chiều
(amended_by, replaced_by, repealed_by) và status được TÍNH từ đồ thị -> không phụ thuộc thứ tự nạp
file (nạp 349 trước hay sau 214 đều cho kết quả như nhau).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from difflib import SequenceMatcher
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.config import Settings, get_settings
from src.services.document_parser import Relation
from src.services.logger_service import get_logger
from src.services.qdrant_service import QdrantService

logger = get_logger("relations")

# (kiểu quan hệ đi ra, trường đi ra của nguồn, trường đi vào của đích)
_REL_FIELDS = {
    "amends": ("amends", "amended_by"),
    "replaces": ("replaces", "replaced_by"),
    "repeals": ("repeals", "repealed_by"),
}


def _split_number(number: str) -> Optional[tuple[str, str, str]]:
    parts = number.split("/")
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None


class RelationsService:
    def __init__(self, qdrant: QdrantService, settings: Optional[Settings] = None) -> None:
        self._qdrant = qdrant
        self._s = settings or get_settings()
        self._lock = threading.RLock()
        self._docs: dict[str, dict[str, Any]] = {}
        self._load()

    def _canon(self, target: str) -> str:
        """Quy số hiệu được nhắc tới về số hiệu của văn bản đã nạp, chịu lỗi OCR ở phần hậu tố.

        Ví dụ OCR đọc 'NĐ-CP' thành 'NB-CP': '214/2025/NB-CP' vẫn nối được với '214/2025/NĐ-CP'
        (cùng số + năm, hậu tố giống nhau >= 60%). Khác loại văn bản (vd. TT-BTC) thì không nối.
        """
        if target in self._docs:
            return target
        t = _split_number(target)
        if not t:
            return target
        best, best_ratio = None, 0.0
        for doc in self._docs:
            d = _split_number(doc)
            if d and d[0] == t[0] and d[1] == t[1]:
                ratio = SequenceMatcher(None, d[2], t[2]).ratio()
                if ratio > best_ratio:
                    best, best_ratio = doc, ratio
        return best if best and best_ratio >= 0.6 else target

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        path = self._s.relations_file
        if not path.exists():
            return
        try:
            self._docs = json.loads(path.read_text(encoding="utf-8")).get("documents", {})
            logger.info("Loaded %d documents from relations registry", len(self._docs))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Không đọc được %s: %s (khởi tạo registry rỗng)", path, exc)
            self._docs = {}

    def _save(self) -> None:
        path = self._s.relations_file
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"documents": self._docs}, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except OSError:
            logger.exception("Không ghi được registry quan hệ")
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # --------------------------------------------------------------- mutation
    def upsert_document(
        self,
        *,
        doc_number: str,
        filename: str,
        file_url: str,
        title: Optional[str],
        doc_type: Optional[str],
        chunk_count: int,
        relations: list[Relation],
        pages_read: Optional[int] = None,
        pages_total: Optional[int] = None,
        diagnostics: Optional[dict] = None,
    ) -> set[str]:
        """Ghi văn bản + quan hệ đi ra. Trả về tập doc_number bị ảnh hưởng (cần đồng bộ Qdrant)."""
        with self._lock:
            old = self._docs.get(doc_number, {})
            old_targets = {r["target"] for r in old.get("outgoing", [])}
            self._docs[doc_number] = {
                "doc_number": doc_number,
                "filename": filename,
                "file_url": file_url,
                "title": title,
                "doc_type": doc_type,
                "chunk_count": chunk_count,
                "pages_read": pages_read,
                "pages_total": pages_total,
                "diagnostics": diagnostics or {},
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "outgoing": [{"type": r.type, "target": r.target, "evidence": r.evidence, "scope": r.scope} for r in relations],
                "history": (old.get("history", []) + [{
                    "at": datetime.now(timezone.utc).isoformat(), "filename": filename, "chunks": chunk_count,
                    "pages": f"{pages_read}/{pages_total}" if pages_total else None,
                    "relations": [f"{r.type}:{r.scope}:{r.target}" for r in relations],
                }])[-20:],
            }
            self._save()
            new_targets = {r.target for r in relations}
            return {doc_number} | {self._canon(t) for t in old_targets | new_targets}

    def get_record(self, doc_number: str) -> Optional[dict[str, Any]]:
        """Bản sao bản ghi (để hoàn tác nếu việc lưu vào Qdrant thất bại)."""
        with self._lock:
            rec = self._docs.get(doc_number)
            return json.loads(json.dumps(rec)) if rec is not None else None

    def restore_record(self, doc_number: str, record: Optional[dict[str, Any]]) -> None:
        with self._lock:
            if record is None:
                self._docs.pop(doc_number, None)
            else:
                self._docs[doc_number] = record
            self._save()

    def remove_by_filename(self, filename: str) -> set[str]:
        with self._lock:
            affected: set[str] = set()
            for number, rec in list(self._docs.items()):
                if rec["filename"] == filename:
                    affected |= {number} | {self._canon(r["target"]) for r in rec.get("outgoing", [])}
                    del self._docs[number]
            self._save()
            return affected

    def sync_to_qdrant(self, doc_numbers: Iterable[str]) -> list[str]:
        """Đẩy metadata quan hệ mới nhất vào payload Qdrant của các văn bản đã nạp."""
        synced: list[str] = []
        for number in doc_numbers:
            if number not in self._docs:
                continue  # văn bản được nhắc tới nhưng chưa nạp -> sẽ nhận metadata khi nạp
            self._qdrant.set_doc_payload(number, self.payload_for(number))
            synced.append(number)
        if synced:
            logger.info("Synced relation metadata to Qdrant: %s", synced)
        return synced

    # ------------------------------------------------------------------ views
    def _incoming(self, doc_number: str, rel_type: str) -> list[str]:
        return sorted(
            src for src, rec in self._docs.items()
            if any(r["type"] == rel_type and self._canon(r["target"]) == doc_number for r in rec.get("outgoing", []))
        )

    def payload_for(self, doc_number: str) -> dict[str, Any]:
        """Metadata quan hệ ghi vào payload Qdrant (amended_by, status, ...)."""
        with self._lock:
            rec = self._docs.get(doc_number, {})
            outgoing = rec.get("outgoing", [])
            payload: dict[str, Any] = {}
            for rel_type, (out_field, in_field) in _REL_FIELDS.items():
                payload[out_field] = sorted({self._canon(r["target"]) for r in outgoing if r["type"] == rel_type})
                payload[in_field] = self._incoming(doc_number, rel_type)
            if payload["repealed_by"]:
                status = "repealed"
            elif payload["replaced_by"]:
                status = "replaced"
            elif payload["amended_by"]:
                status = "amended"
            else:
                status = "unverified"  # CHƯA xác minh: không có dữ liệu bãi bỏ ≠ còn hiệu lực
            payload["status"] = status
            return payload

    def base_metadata(self, doc_number: str) -> dict[str, Any]:
        rec = self._docs.get(doc_number, {})
        return {
            "doc_number": doc_number,
            "filename": rec.get("filename"),
            "file_url": rec.get("file_url"),
            "title": rec.get("title"),
            "doc_type": rec.get("doc_type"),
        }

    def resync_all(self) -> list[str]:
        """Đồng bộ lại metadata quan hệ/status vào payload Qdrant cho MỌI văn bản đã nạp (chỉ cập nhật payload, không đụng vector)."""
        return self.sync_to_qdrant(list(self._docs))

    def filename_of(self, doc_number: str) -> Optional[str]:
        return self._docs.get(doc_number, {}).get("filename")

    def file_url_of(self, doc_number: str) -> Optional[str]:
        return self._docs.get(doc_number, {}).get("file_url")

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            result = []
            for number, rec in sorted(self._docs.items()):
                item = {k: v for k, v in rec.items() if k != "outgoing"}
                item.update(self.payload_for(number))
                item["relations"] = [
                    {**r, "raw_target": r["target"], "target": self._canon(r["target"]),
                     "target_ingested": self._canon(r["target"]) in self._docs}
                    for r in rec.get("outgoing", [])
                ]
                total, read = rec.get("pages_total"), rec.get("pages_read")
                item["coverage"] = f"{read}/{total} trang" if total else None
                result.append(item)
            return result
