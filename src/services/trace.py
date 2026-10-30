"""Nhật ký đo lường dạng JSON Lines (mỗi dòng một sự kiện) để phân tích chất lượng và gửi cho người hỗ trợ.

- logs/trace.jsonl  : mỗi câu hỏi (chat / patch / diagnostics) với thời gian, token, điểm truy xuất, nguồn, kiểm tra đầu ra.
- logs/ingest.jsonl : mỗi lần nạp tài liệu với độ phủ trang, chất lượng OCR, embedding, số chunk kiểm chứng.
Xoay vòng 10MB x 3 file. Dùng scripts/analyze_trace.py để tóm tắt.
"""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.config import get_settings

_loggers: dict[str, logging.Logger] = {}

# Che khoá/token trước khi ghi log (phòng khi lỗi từ nhà cung cấp chứa khoá trong thông báo)
_REDACT = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "AIza***"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "sk-***"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer ***"),
    (re.compile(r"(?i)((?:api[_-]?key|token|secret|password)[\"'\s:=]+)[^\s\",}]{8,}"), r"\1***"),
]


def redact(text: str) -> str:
    for pattern, repl in _REDACT:
        text = pattern.sub(repl, text)
    return text


def _get(name: str, path: Path) -> logging.Logger:
    lg = _loggers.get(name)
    if lg is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lg = logging.getLogger(f"docqa.trace.{name}")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        for old in [h for h in lg.handlers if isinstance(h, RotatingFileHandler)]:  # bỏ handler file cũ (reload/khởi tạo lại)
            lg.removeHandler(old)
            old.close()
        # LUÔN gắn handler ghi file của mình, kể cả khi logger đã có handler khác (trước đây điều kiện `if not lg.handlers`
        # có thể làm trace âm thầm không được ghi)
        handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(handler)
        _loggers[name] = lg
    return lg


def _write(name: str, path: Path, kind: str, fields: dict[str, Any]) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "kind": kind, **fields}
    try:
        _get(name, path).info(redact(json.dumps(record, ensure_ascii=False, default=str)))
    except Exception:  # noqa: BLE001 - ghi trace không bao giờ được làm hỏng request
        logging.getLogger("docqa.trace").exception("Không ghi được trace")


def trace(kind: str, **fields: Any) -> None:
    _write("chat", get_settings().trace_file, kind, fields)


def trace_ingest(kind: str, **fields: Any) -> None:
    _write("ingest", get_settings().ingest_trace_file, kind, fields)


def est_tokens(chars: int) -> int:
    """Ước lượng thô số token từ số ký tự (tiếng Việt ~3 ký tự/token). Số thật lấy từ `usage` của nhà cung cấp."""
    return math.ceil(chars / 3.0)
