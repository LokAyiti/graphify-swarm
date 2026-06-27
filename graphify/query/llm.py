"""
llm.py — Ollama LLM backend (local, no API key).

Uses Ollama's HTTP API directly (stdlib urllib only, zero extra deps).

Endpoints used
--------------
  GET  /api/tags           → list installed models
  POST /api/chat           → chat completion (streaming NDJSON)
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Generator


class OllamaLLM:
    """Thin wrapper around a local Ollama server."""

    def __init__(
        self,
        model: str,
        host:  str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.host  = host.rstrip("/")

    # ── Discovery ──────────────────────────────────────────────────────────

    def is_available(self, timeout: int = 3) -> bool:
        """Return True if Ollama is reachable."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return names of all locally installed models."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def model_exists(self) -> bool:
        """Return True if self.model is installed in Ollama."""
        available = self.list_models()
        return any(m == self.model or m.startswith(f"{self.model}:") for m in available)

    # ── Generation ─────────────────────────────────────────────────────────

    def ask_stream(
        self,
        messages: list[dict],
        timeout: int = 180,
    ) -> Generator[str, None, None]:
        """Yield text tokens from the model using the /api/chat streaming endpoint.

        *messages* should be in Ollama chat format:
            [{"role": "system", "content": "…"}, {"role": "user", "content": "…"}]
        """
        body = json.dumps({
            "model":    self.model,
            "messages": messages,
            "stream":   True,
            "options": {
                "temperature": 0.2,   # more deterministic for code Q&A
                "num_ctx":     8192,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token

                    if chunk.get("done"):
                        break

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.host}. "
                f"Start it with: ollama serve\n({exc})"
            ) from exc

    def ask(self, messages: list[dict], timeout: int = 180) -> str:
        """Return the full response as a single string (non-streaming)."""
        return "".join(self.ask_stream(messages, timeout=timeout))
