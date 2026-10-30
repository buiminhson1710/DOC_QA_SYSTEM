"""Định tuyến gọi LLM: chuỗi model dự phòng, retry cho lỗi tạm thời, và ghi nhớ model đã hết quota.

Vì sao cần: gói free của Gemini giới hạn theo NGÀY, theo từng model (ví dụ 20 lượt/ngày). Khi chạm hạn mức này,
thử lại là vô ích (phải chờ hàng giờ) và còn làm chậm. Router phát hiện lỗi quota theo ngày, đưa model vào "cooldown"
và chuyển ngay sang model kế tiếp; các request sau bỏ qua model đó cho tới khi hết cooldown.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import litellm

from src.services.logger_service import get_logger

logger = get_logger("llm")
litellm.drop_params = True

_lock = threading.Lock()
_cooldown: dict[str, tuple[float, str]] = {}  # model -> (hết cooldown lúc, lý do)

DAILY_QUOTA_COOLDOWN = 3600     # kiểm tra lại mỗi giờ (phòng khi bạn vừa bật billing / đổi gói)
OVERLOAD_COOLDOWN = 60          # model quá tải (503) tạm bỏ qua 1 phút
FATAL_COOLDOWN = 300            # sai key / model không tồn tại

_RETRY_DELAY_JSON = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
_RETRY_IN_TEXT = re.compile(r"retry in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", re.I)


class AllModelsUnavailable(Exception):
    """Mọi model trong chuỗi đều hết quota / quá tải / lỗi."""

    def __init__(self, message: str, reasons: dict[str, str]) -> None:
        super().__init__(message)
        self.reasons = reasons


@dataclass
class ErrorPlan:
    kind: str            # quota_daily | rate_limit | overloaded | fatal
    cooldown: float
    wait: float = 0.0


def retry_delay_seconds(message: str) -> Optional[float]:
    m = _RETRY_DELAY_JSON.search(message)
    if m:
        return float(m.group(1))
    m = _RETRY_IN_TEXT.search(message)
    if m and any(m.groups()):
        h, mi, sec = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + sec
    return None


def classify(exc: Exception) -> ErrorPlan:
    msg = str(exc)
    low = msg.lower()
    delay = retry_delay_seconds(msg)
    if isinstance(exc, litellm.RateLimitError) or "429" in msg or "resource_exhausted" in low or "quota" in low:
        daily = "perday" in low.replace("_", "").replace(" ", "") or "per day" in low or "daily" in low
        if daily or (delay is not None and delay > 90):
            return ErrorPlan("quota_daily", min(delay or DAILY_QUOTA_COOLDOWN, DAILY_QUOTA_COOLDOWN))
        wait = min(delay or 5.0, 20.0)
        return ErrorPlan("rate_limit", max(wait, 30.0), wait)
    if isinstance(exc, (litellm.ServiceUnavailableError, litellm.InternalServerError, litellm.Timeout,
                        litellm.APIConnectionError)) or any(c in msg for c in ("503", "500", "UNAVAILABLE", "overloaded")):
        return ErrorPlan("overloaded", OVERLOAD_COOLDOWN)
    return ErrorPlan("fatal", FATAL_COOLDOWN)


def _set_cooldown(model: str, seconds: float, reason: str) -> None:
    with _lock:
        _cooldown[model] = (time.time() + seconds, reason)
    logger.warning("Model %s tạm ngưng %.0fs (%s)", model, seconds, reason)


def available(models: list[str]) -> list[str]:
    now = time.time()
    with _lock:
        return [m for m in models if _cooldown.get(m, (0, ""))[0] <= now]


def status(models: Optional[list[str]] = None) -> dict[str, Any]:
    """Trạng thái cooldown (dùng cho /api/health)."""
    now = time.time()
    with _lock:
        items = {m: v for m, v in _cooldown.items() if v[0] > now and (models is None or m in models)}
    return {m: {"seconds_left": int(until - now), "reason": reason} for m, (until, reason) in items.items()}


def reset() -> None:
    with _lock:
        _cooldown.clear()


def _no_model_error(models: list[str], reasons: dict[str, str]) -> AllModelsUnavailable:
    detail = "; ".join(f"{m}: {r}" for m, r in reasons.items()) or "không có model nào được cấu hình"
    return AllModelsUnavailable(f"Mọi model đều không khả dụng ({detail})", reasons)


def _cooling_reasons(models: list[str]) -> dict[str, str]:
    now = time.time()
    with _lock:
        return {m: f"{_cooldown[m][1]} (còn {int(_cooldown[m][0] - now)}s)" for m in models if m in _cooldown and _cooldown[m][0] > now}


async def acomplete(models: list[str], info: Optional[dict[str, Any]] = None, **kwargs: Any) -> tuple[Any, str]:
    """Gọi chat completion qua chuỗi model. Trả về (response, model đã dùng). `info` (nếu có) được điền số lần thử và lỗi."""
    info = info if info is not None else {}
    info.update({"attempts": 0, "errors": []})
    candidates = available(models)
    reasons = _cooling_reasons(models)
    if not candidates:
        raise _no_model_error(models, reasons)
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("timeout", 35)

    for model in candidates:
        for attempt in (1, 2, 3):
            try:
                info["attempts"] += 1
                resp = await litellm.acompletion(model=model, num_retries=0, **call_kwargs)
                if model != models[0]:
                    logger.warning("Đã dùng model dự phòng: %s", model)
                return resp, model
            except Exception as exc:  # noqa: BLE001
                plan = classify(exc)
                msg = str(exc)
                reasons[model] = f"{plan.kind}: {msg[:120]}"
                info["errors"].append({"model": model, "kind": plan.kind, "msg": msg[:160]})
                is_timeout = isinstance(exc, (litellm.Timeout, asyncio.TimeoutError)) or "timeout" in msg.lower() or "timed out" in msg.lower()
                # Nếu timeout hoặc lỗi chết: không thử lại model này mà chuyển ngay sang model dự phòng kế tiếp
                if is_timeout:
                    _set_cooldown(model, plan.cooldown, "timeout")
                    break
                if plan.kind == "rate_limit" and attempt == 1:
                    await asyncio.sleep(plan.wait)
                    continue
                if plan.kind == "overloaded" and attempt < 2:
                    await asyncio.sleep(1.5)
                    continue
                _set_cooldown(model, plan.cooldown, plan.kind)
                break
    raise _no_model_error(models, reasons)


def complete(models: list[str], **kwargs: Any) -> tuple[Any, str]:
    """Bản đồng bộ (dùng trong luồng OCR chạy ở thread)."""
    candidates = available(models)
    reasons = _cooling_reasons(models)
    if not candidates:
        raise _no_model_error(models, reasons)
    for model in candidates:
        for attempt in (1, 2, 3):
            try:
                resp = litellm.completion(model=model, num_retries=0, **kwargs)
                return resp, model
            except Exception as exc:  # noqa: BLE001
                plan = classify(exc)
                reasons[model] = f"{plan.kind}: {str(exc)[:120]}"
                if plan.kind == "rate_limit" and attempt == 1:
                    time.sleep(plan.wait)
                    continue
                if plan.kind == "overloaded" and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                _set_cooldown(model, plan.cooldown, plan.kind)
                break
    raise _no_model_error(models, reasons)
