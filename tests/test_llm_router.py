"""Unit test llm_router: quota ngày không retry, cooldown, chuyển model dự phòng, ghi nhận số lần thử."""
from __future__ import annotations

import asyncio

import litellm
import pytest

from src.services import llm_router as LR
from test_embedding import DAILY, PER_MINUTE


def acall(models, **kw):
    return asyncio.run(LR.acomplete(models, **kw))


def test_daily_quota_one_call_then_cooldown_and_fallback(provider):
    provider.chat_behavior = lambda p, model, msgs: litellm.RateLimitError(DAILY, "gemini", model) if model == "gemini/a" else "ok"
    info: dict = {}
    resp, used = acall(["gemini/a", "gemini/b"], info=info, messages=[{"role": "user", "content": "x"}])
    assert used == "gemini/b" and info["attempts"] == 2 and info["errors"][0]["kind"] == "quota_daily"
    assert [c["model"] for c in provider.chat_calls] == ["gemini/a", "gemini/b"] and provider.sleeps == []
    provider.chat_calls.clear()
    acall(["gemini/a", "gemini/b"], messages=[])
    assert [c["model"] for c in provider.chat_calls] == ["gemini/b"]         # a bị bỏ qua nhờ cooldown
    assert "gemini/a" in LR.status() and LR.status()["gemini/a"]["reason"] == "quota_daily"


def test_overloaded_retried_three_times_then_next_model(provider):
    provider.chat_behavior = lambda p, model, msgs: litellm.ServiceUnavailableError("503 high demand", "gemini", model) if model == "gemini/a" else "ok"
    info: dict = {}
    _, used = acall(["gemini/a", "gemini/b"], info=info, messages=[])
    assert used == "gemini/b" and [c["model"] for c in provider.chat_calls].count("gemini/a") == 3 and len(provider.sleeps) == 2


def test_short_rate_limit_waits_once_then_moves_on(provider):
    provider.chat_behavior = lambda p, model, msgs: litellm.RateLimitError(PER_MINUTE, "gemini", model) if model == "gemini/a" else "ok"
    _, used = acall(["gemini/a", "gemini/b"], messages=[])
    assert used == "gemini/b" and [c["model"] for c in provider.chat_calls].count("gemini/a") == 2 and provider.sleeps == [3.0]


def test_all_models_unavailable_raises_with_reasons_and_does_not_loop(provider):
    provider.chat_behavior = lambda p, model, msgs: litellm.RateLimitError(DAILY, "gemini", model)
    with pytest.raises(LR.AllModelsUnavailable) as exc:
        acall(["gemini/a", "gemini/b"], messages=[])
    assert set(exc.value.reasons) == {"gemini/a", "gemini/b"} and len(provider.chat_calls) == 2
    with pytest.raises(LR.AllModelsUnavailable):                              # lần sau: không gọi API nào nữa
        acall(["gemini/a", "gemini/b"], messages=[])
    assert len(provider.chat_calls) == 2


def test_retry_delay_parsing():
    assert LR.retry_delay_seconds(DAILY) == 60451.0
    assert LR.retry_delay_seconds("Please retry in 16h47m31.8s.") == pytest.approx(16 * 3600 + 47 * 60 + 31.8)
    assert LR.retry_delay_seconds("no hint") is None
