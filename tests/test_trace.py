"""Unit test trace: ghi JSONL, che khoá, không bao giờ làm hỏng request."""
from __future__ import annotations

import json

from conftest import read_jsonl
from src.services import trace as T


def test_trace_writes_jsonl_with_timestamp_and_kind(cfg):
    T.trace("chat", request_id="r1", mode="llm", usage={"prompt_tokens": 10})
    T.trace_ingest("ingest_done", ingest_id="i1", chunks=3)
    chat = read_jsonl(cfg.trace_file)
    ing = read_jsonl(cfg.ingest_trace_file)
    assert chat[0]["kind"] == "chat" and chat[0]["request_id"] == "r1" and chat[0]["ts"].endswith("+00:00")
    assert ing[0]["ingest_id"] == "i1" and ing[0]["chunks"] == 3


def test_secrets_are_redacted_everywhere(cfg):
    key_g = "AIza" + "S" * 35
    key_o = "sk-" + "a1B2" * 8
    T.trace("chat_error", error=f"401 invalid key {key_g} and {key_o}", note="Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
            cfg={"api_key": "supersecretvalue123", "password": "hunter2hunter2"})
    raw = cfg.trace_file.read_text(encoding="utf-8")
    for secret in (key_g, key_o, "abcdefghijklmnopqrstuvwxyz123456", "supersecretvalue123", "hunter2hunter2"):
        assert secret not in raw
    assert "AIza***" in raw and "sk-***" in raw


def test_trace_failure_never_raises(cfg, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(T, "_get", boom)
    T.trace("chat", x=1)                      # không ném ngoại lệ
    T.trace_ingest("ingest_done", x=1)


def test_non_json_values_are_stringified(cfg):
    from pathlib import Path

    T.trace("chat", path=Path("a/b"), s={1, 2})
    assert read_jsonl(cfg.trace_file)[0]["path"] == "a/b"


def test_est_tokens_is_labeled_estimate_math():
    assert T.est_tokens(300) == 100 and T.est_tokens(1) == 1


def test_question_text_can_be_hidden(cfg, monkeypatch):
    from src.services.rag_engine import _log_text

    assert "Hạn mức" in _log_text("Hạn mức là bao nhiêu?", 100)
    monkeypatch.setenv("TRACE_LOG_TEXT", "false")
    cfg.__class__  # noqa: B018
    from src import config

    config.get_settings.cache_clear()
    hidden = _log_text("Hạn mức là bao nhiêu?", 100)
    assert "Hạn mức" not in hidden and "sha1:" in hidden
    json.dumps(hidden)
