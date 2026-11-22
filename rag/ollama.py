"""One non-streaming request to the local Ollama chat API."""

import json
from http.client import HTTPConnection, HTTPException


MODEL = "qwen2.5:3b"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 32768, "num_predict": 256}
TIMEOUT_SECONDS = 120


class OllamaError(RuntimeError):
    """The local service could not return a complete generated answer."""


def chat(system: str, user: str) -> str:
    # UTF-8 byte length is a conservative upper bound for Qwen's byte-level tokens.
    # Reserve room for the chat template and the requested output.
    if len((system + user).encode("utf-8")) + 256 + OPTIONS["num_predict"] > OPTIONS["num_ctx"]:
        raise OllamaError("Question and complete passages exceed the conservative local context budget.")
    payload = {
        "model": MODEL, "stream": False, "options": OPTIONS,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    conn = HTTPConnection("127.0.0.1", 11434, timeout=TIMEOUT_SECONDS)
    try:
        conn.request("POST", "/api/chat", json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read()
    except TimeoutError as exc:
        raise OllamaError(f"Ollama timed out after {TIMEOUT_SECONDS} seconds.") from exc
    except (OSError, HTTPException) as exc:
        raise OllamaError("Cannot reach local Ollama at 127.0.0.1:11434. Check that Ollama is running.") from exc
    finally:
        conn.close()
    if response.status == 404:
        raise OllamaError(f"Ollama could not find {MODEL}. Run 'ollama pull {MODEL}'.")
    if response.status != 200:
        raise OllamaError(f"Ollama returned HTTP {response.status}: {raw.decode('utf-8', errors='replace')[:300]}")
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise OllamaError("Ollama returned an invalid JSON response.") from exc
    if not isinstance(result, dict):
        raise OllamaError("Ollama returned an unexpected response format.")
    if result.get("error"):
        raise OllamaError(f"Ollama generation failed: {result['error']}")
    message = result.get("message")
    answer = message.get("content") if isinstance(message, dict) else None
    if result.get("done") is not True or result.get("done_reason") == "length":
        raise OllamaError("Ollama did not finish the answer within the generation budget.")
    if not isinstance(answer, str) or not answer.strip():
        raise OllamaError("Ollama returned an empty answer.")
    return answer
