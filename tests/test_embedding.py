"""Unit test embedding: số lượng/dimension, batch, phân loại lỗi (429 ngày/phút, auth, mạng, input), backoff+jitter, cache tiếp tục.
Mock litellm.aembedding => không gọi API thật, không tốn quota."""
from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import litellm
import pytest

from src.services import qdrant_service as QS
from src.services.qdrant_service import EmbeddingError, backoff_delay, classify_embedding_error

DAILY = ('litellm.RateLimitError: geminiException - {"error": {"code": 429, "message": "You exceeded your current quota. '
         'Quota exceeded for metric: generate_content_free_tier_requests, limit: 20", "details": [{"quotaId": '
         '"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}, {"retryDelay": "60451s"}]}}')
PER_MINUTE = 'litellm.RateLimitError: geminiException - {"error": {"code": 429, "message": "Rate limit PerMinute", "details": [{"retryDelay": "3s"}]}}'


def run(coro):
    return asyncio.run(coro)


def test_counts_dimension_and_batching(qdrant, provider):
    texts = [f"đoạn văn bản số {i}" for i in range(5)]
    out = run(qdrant.embed_texts(texts, batch_size=2))
    assert len(out) == 5 and all(len(v) == 64 for v in out)
    assert [len(c) for c in provider.embed_calls] == [2, 2, 1]
    st = qdrant.last_embed
    assert st["inputs"] == 5 and st["outputs"] == 5 and st["calls"] == 3 and st["batch_size"] == 2
    assert st["tokens_source"] == "provider" and st["prompt_tokens"] > 0


def test_dimension_mismatch_is_rejected(qdrant, provider):
    provider.embed_behavior = lambda p, inp: SimpleNamespace(data=[{"embedding": [0.1] * 32} for _ in inp], usage=None)
    with pytest.raises(EmbeddingError) as exc:
        run(qdrant.embed_texts(["a", "b"]))
    assert exc.value.kind == "invalid_output" and "EMBEDDING_DIM" in str(exc.value)


def test_zero_vector_and_nan_are_rejected(qdrant, provider):
    provider.embed_behavior = lambda p, inp: SimpleNamespace(data=[{"embedding": [0.0] * 64} for _ in inp], usage=None)
    with pytest.raises(EmbeddingError) as exc:
        run(qdrant.embed_texts(["a"]))
    assert exc.value.kind == "invalid_output"


def test_output_count_mismatch_is_rejected(qdrant, provider):
    provider.embed_behavior = lambda p, inp: SimpleNamespace(data=[{"embedding": [0.1] * 64}], usage=None)
    with pytest.raises(EmbeddingError) as exc:
        run(qdrant.embed_texts(["a", "b"]))
    assert exc.value.kind == "invalid_output" and "1 vector cho 2" in str(exc.value)


def test_short_429_is_retried_with_backoff_then_succeeds(qdrant, provider):
    state = {"n": 0}

    def behavior(p, inp):
        state["n"] += 1
        return litellm.RateLimitError(PER_MINUTE, "gemini", "m") if state["n"] <= 2 else None

    provider.embed_behavior = behavior
    out = run(qdrant.embed_texts(["xin chào"]))
    assert len(out) == 1 and len(provider.embed_calls) == 3
    assert qdrant.last_embed["retries"] == 2
    assert len(provider.sleeps) == 2 and all(3.0 <= s <= 30.0 for s in provider.sleeps)   # tôn trọng retryDelay 3s


def test_daily_quota_429_is_not_retried(qdrant, provider):
    provider.embed_behavior = lambda p, inp: litellm.RateLimitError(DAILY, "gemini", "m")
    with pytest.raises(EmbeddingError) as exc:
        run(qdrant.embed_texts(["a"]))
    assert exc.value.kind == "quota_daily" and not exc.value.retryable
    assert len(provider.embed_calls) == 1 and provider.sleeps == []     # retry không khắc phục được quota đã cạn
    assert "không retry" in str(exc.value)


def test_auth_error_is_not_retried_and_hints_env(qdrant, provider):
    provider.embed_behavior = lambda p, inp: litellm.AuthenticationError("Incorrect API key provided: sk-xxxxxxxx", "openai", "m")
    with pytest.raises(EmbeddingError) as exc:
        run(qdrant.embed_texts(["a"]))
    assert exc.value.kind == "auth" and len(provider.embed_calls) == 1 and ".env" in str(exc.value)


def test_network_error_retries_are_bounded_with_exponential_jittered_backoff(qdrant, provider, cfg):
    provider.embed_behavior = lambda p, inp: litellm.APIConnectionError("connection reset", "gemini", "m")
    with pytest.raises(EmbeddingError) as exc:
        run(qdrant.embed_texts(["a"]))
    assert exc.value.kind == "overloaded" and exc.value.retryable
    assert len(provider.embed_calls) == cfg.embed_max_retries + 1       # KHÔNG retry vô hạn
    assert len(provider.sleeps) == cfg.embed_max_retries
    for attempt, delay in enumerate(provider.sleeps, start=1):
        d = min(30.0, 2 ** (attempt - 1))
        assert d / 2 <= delay <= d                                       # equal jitter trong [d/2, d]


def test_input_error_isolates_bad_text(qdrant, provider):
    def behavior(p, inp):
        return litellm.BadRequestError("invalid argument: text too long", "gemini", "m") if any("BAD" in t for t in inp) else None

    provider.embed_behavior = behavior
    with pytest.raises(EmbeddingError) as exc:
        run(qdrant.embed_texts(["tốt", "BAD", "khác"], batch_size=3))
    assert exc.value.kind == "input"
    assert [len(c) for c in provider.embed_calls] == [3, 1, 1]            # gọi cả batch, rồi từng văn bản đến khi gặp văn bản lỗi


def test_resume_from_cache_after_midway_failure_makes_no_duplicate_calls(qdrant, provider):
    texts = [f"chunk {i}" for i in range(6)]
    state = {"n": 0}

    def fail_third(p, inp):
        state["n"] += 1
        return litellm.RateLimitError(DAILY, "gemini", "m") if state["n"] == 3 else None

    provider.embed_behavior = fail_third
    with pytest.raises(EmbeddingError) as exc:
        run(qdrant.embed_texts(texts, batch_size=2))
    assert exc.value.kind == "quota_daily"
    provider.embed_behavior = None
    provider.embed_calls.clear()
    out = run(qdrant.embed_texts(texts, batch_size=2))
    assert len(out) == 6
    assert provider.embed_calls == [["chunk 4", "chunk 5"]]                # chỉ phần còn thiếu
    assert qdrant.last_embed["from_cache"] == 4 and qdrant.last_embed["api_texts"] == 2


def test_use_cache_false_always_calls_api(qdrant, provider):
    run(qdrant.embed_texts(["x"], use_cache=False))
    run(qdrant.embed_texts(["x"], use_cache=False))
    assert len(provider.embed_calls) == 2


def test_backoff_delay_bounds_and_wait_floor():
    rng = random.Random(1)
    for attempt in range(1, 10):
        d = min(30.0, 2 ** (attempt - 1))
        assert d / 2 <= backoff_delay(attempt, rng=rng) <= d
    assert backoff_delay(1, wait=7.0, rng=rng) >= 7.0


@pytest.mark.parametrize("exc,kind", [
    (litellm.AuthenticationError("401 invalid_api_key", "openai", "m"), "auth"),
    (litellm.RateLimitError(DAILY, "gemini", "m"), "quota_daily"),
    (litellm.RateLimitError(PER_MINUTE, "gemini", "m"), "rate_limit"),
    (litellm.ServiceUnavailableError("503 high demand", "gemini", "m"), "overloaded"),
    (litellm.APIConnectionError("reset", "gemini", "m"), "overloaded"),
    (litellm.NotFoundError("model not found", "gemini", "m"), "config"),
    (litellm.BadRequestError("400 invalid argument", "gemini", "m"), "input"),
    (ValueError("???"), "error"),
])
def test_error_classification(exc, kind):
    assert classify_embedding_error(exc) == kind
