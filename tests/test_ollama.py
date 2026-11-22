import json
from unittest.mock import Mock

import pytest

from rag import ollama


def connection(monkeypatch, status=200, body=b"{}"):
    conn = Mock()
    conn.getresponse.return_value.status = status
    conn.getresponse.return_value.read.return_value = body
    monkeypatch.setattr(ollama, "HTTPConnection", Mock(return_value=conn))
    return conn


def test_local_chat_request_and_unmodified_answer(monkeypatch):
    raw = "Fact [MED-9].\n"
    body = json.dumps({"done": True, "done_reason": "stop", "message": {"content": raw}}).encode()
    conn = connection(monkeypatch, body=body)
    assert ollama.chat("system", "prompt") == raw
    ollama.HTTPConnection.assert_called_once_with("127.0.0.1", 11434, timeout=120)
    method, endpoint, payload, headers = conn.request.call_args.args
    assert (method, endpoint) == ("POST", "/api/chat")
    req = json.loads(payload)
    assert req["model"] == "qwen2.5:3b" and req["stream"] is False
    assert req["options"]["temperature"] == 0 and req["options"]["seed"] == 42
    assert req["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "prompt"}]
    conn.close.assert_called_once()


@pytest.mark.parametrize(("status", "body", "message"), [
    (404, b'{"error":"model not found"}', "ollama pull"),
    (500, b"generation failed", "HTTP 500"),
    (200, b"invalid json", "invalid JSON"),
    (200, b'{"error":"model runner stopped"}', "runner stopped"),
    (200, b'{"done":false}', "did not finish"),
    (200, b'{"done":true,"done_reason":"length","message":{"content":"Partial"}}', "did not finish"),
    (200, b'{"done":true,"message":{"content":" "}}', "empty answer"),
])
def test_generation_errors(monkeypatch, status, body, message):
    connection(monkeypatch, status, body)
    with pytest.raises(ollama.OllamaError, match=message):
        ollama.chat("system", "question")


@pytest.mark.parametrize(("cause", "message"), [(ConnectionRefusedError(), "Cannot reach"), (TimeoutError(), "timed out")])
def test_unavailable_service_and_timeout(monkeypatch, cause, message):
    conn = connection(monkeypatch)
    conn.request.side_effect = cause
    with pytest.raises(ollama.OllamaError, match=message):
        ollama.chat("system", "question")
    conn.close.assert_called_once()


def test_oversized_prompt_is_rejected_without_trimming(monkeypatch):
    conn = connection(monkeypatch)
    with pytest.raises(ollama.OllamaError, match="context budget"):
        ollama.chat("sys", "x" * 32768)
    conn.request.assert_not_called()
