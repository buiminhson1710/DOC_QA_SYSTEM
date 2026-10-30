"""Logging xoay vòng: logs/app.log, tối đa 10MB/file, giữ 5 bản backup."""
from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Any

from src.config import get_settings

_ROOT_NAME = "docqa"
_configured = False


def setup_logging() -> None:
    """Cấu hình logger gốc 'docqa' (idempotent)."""
    global _configured
    if _configured:
        return
    settings = get_settings()
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(logging.INFO)
    root.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def preview(text: str, limit: int = 200) -> str:
    """Rút gọn văn bản người dùng trước khi ghi log (tránh log quá dài / lộ nội dung)."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Ghi log có cấu trúc: 'EVENT | {json}'."""
    logger.log(level, "%s | %s", event, json.dumps(fields, ensure_ascii=False, default=str))
