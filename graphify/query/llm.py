"""
llm.py — Multi-provider LLM backend for Graphify.

Provider auto-detection (priority order from .env / environment):
  1. DATABRICKS_TOKEN          → Databricks-hosted models (OpenAI-compatible)
  2. OPENAI_API_KEY            → OpenAI ChatGPT
  3. ANTHROPIC_API_KEY         → Anthropic Claude (direct API)
  4. GOOGLE_API_KEY            → Google Gemini
  5. (none configured)         → Ollama local fallback

CLI override examples:
  graphify ask "q" --provider databricks --llm databricks-claude-sonnet-4-6
  graphify ask "q" --provider openai     --llm gpt-4o
  graphify ask "q" --provider anthropic  --llm claude-3-5-sonnet-latest
  graphify ask "q" --provider google     --llm gemini-1.5-pro
  graphify ask "q" --provider ollama     --llm llama3
  graphify ask "q"  ← auto-detects from .env, falls back to ollama
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Generator, Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def ask_stream(
        self,
        messages: list[dict],
        timeout:  int = 180,
    ) -> Generator[str, None, None]:
        """Yield text tokens one at a time."""
        ...

    def ask(self, messages: list[dict], timeout: int = 180) -> str:
        """Non-streaming: return the full response as a string."""
        return "".join(self.ask_stream(messages, timeout=timeout))

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__.replace("LLM", "").lower()

    def is_available(self) -> bool:
        return True

    def model_exists(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Ollama (local, no API key)
# ---------------------------------------------------------------------------

class OllamaLLM(BaseLLM):
    """Thin wrapper around a local Ollama server."""

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host  = host.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    def is_available(self, timeout: int = 3) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def model_exists(self) -> bool:
        available = self.list_models()
        return any(m == self.model or m.startswith(f"{self.model}:") for m in available)

    def ask_stream(
        self,
        messages: list[dict],
        timeout:  int = 180,
    ) -> Generator[str, None, None]:
        body = json.dumps({
            "model":    self.model,
            "messages": messages,
            "stream":   True,
            "options":  {"temperature": 0.2, "num_ctx": 8192},
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


# ---------------------------------------------------------------------------
# OpenAI-compatible (OpenAI, Azure OpenAI, Groq, Together, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleLLM(BaseLLM):
    """
    Any provider that speaks the OpenAI Chat Completions API.

    Works with: OpenAI, Azure OpenAI, Groq, Together AI, Anyscale, etc.
    """

    def __init__(
        self,
        model:    str,
        api_key:  str,
        base_url: str,
        provider: str = "openai",
    ) -> None:
        self.model    = model
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return self._provider

    def ask_stream(
        self,
        messages: list[dict],
        timeout:  int = 180,
    ) -> Generator[str, None, None]:
        body = json.dumps({
            "model":       self.model,
            "messages":    messages,
            "stream":      True,
            "temperature": 0.2,
            "max_tokens":  4096,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        yield delta
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"{self._provider} API not reachable at {self.base_url}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Anthropic Claude (direct API — different format from OpenAI)
# ---------------------------------------------------------------------------

class AnthropicLLM(BaseLLM):
    """Direct Anthropic Messages API."""

    _BASE = "https://api.anthropic.com/v1"
    _VER  = "2023-06-01"

    def __init__(self, model: str, api_key: str) -> None:
        self.model   = model
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def ask_stream(
        self,
        messages: list[dict],
        timeout:  int = 180,
    ) -> Generator[str, None, None]:
        system_content = ""
        chat_messages  = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                chat_messages.append(m)

        payload: dict = {
            "model":      self.model,
            "messages":   chat_messages,
            "stream":     True,
            "max_tokens": 4096,
        }
        if system_content:
            payload["system"] = system_content

        body = json.dumps(payload).encode()
        req  = urllib.request.Request(
            f"{self._BASE}/messages",
            data=body,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         self.api_key,
                "anthropic-version": self._VER,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or line.startswith("event:"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "content_block_delta":
                        text = event.get("delta", {}).get("text", "")
                        if text:
                            yield text
                    elif event.get("type") == "message_stop":
                        break
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Anthropic API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GoogleLLM(BaseLLM):
    """Google Generative Language API (Gemini)."""

    _BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str, api_key: str) -> None:
        self.model   = model
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "google"

    def ask_stream(
        self,
        messages: list[dict],
        timeout:  int = 180,
    ) -> Generator[str, None, None]:
        contents = []
        for m in messages:
            if m["role"] == "system":
                contents.append({"role": "user",  "parts": [{"text": m["content"]}]})
                contents.append({"role": "model", "parts": [{"text": "Understood."}]})
            elif m["role"] == "user":
                contents.append({"role": "user",  "parts": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": m["content"]}]})

        body = json.dumps({
            "contents":         contents,
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
        }).encode()

        url = (
            f"{self._BASE}/models/{self.model}:streamGenerateContent"
            f"?key={self.api_key}&alt=sse"
        )
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for cand in event.get("candidates", []):
                        for part in cand.get("content", {}).get("parts", []):
                            text = part.get("text", "")
                            if text:
                                yield text
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Google API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Databricks (full invocations endpoint, not /chat/completions)
# ---------------------------------------------------------------------------

class DatabricksLLM(BaseLLM):
    """Databricks Model Serving endpoint."""

    def __init__(self, model: str, api_key: str, endpoint: str) -> None:
        self.model    = model
        self.api_key  = api_key
        self.endpoint = endpoint  # full URL ending in /invocations

    @property
    def provider_name(self) -> str:
        return "databricks"

    def ask_stream(
        self,
        messages: list[dict],
        timeout:  int = 180,
    ) -> Generator[str, None, None]:
        body = json.dumps({
            "messages":    messages,
            "temperature": 0.2,
            "max_tokens":  4096,
        }).encode()

        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data   = json.loads(resp.read())
                answer = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if answer:
                    yield answer
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Databricks API error at {self.endpoint}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Factory — auto-detect provider from environment
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: dict[str, str] = {
    "databricks": "databricks-claude-sonnet-4-6",
    "openai":     "gpt-4o-mini",
    "anthropic":  "claude-3-5-haiku-latest",
    "google":     "gemini-1.5-flash",
    "ollama":     "llama3",
}


def detect_provider() -> Optional[str]:
    """Return the first configured API provider name, or None (use Ollama)."""
    if os.environ.get("DATABRICKS_TOKEN") and (
        os.environ.get("DATABRICKS_SONNET_ENDPOINT")
        or os.environ.get("DATABRICKS_OPUS_ENDPOINT")
    ):
        return "databricks"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GOOGLE_API_KEY"):
        return "google"
    return None


def build_llm(
    provider:    Optional[str] = None,
    model:       Optional[str] = None,
    ollama_host: str            = "http://localhost:11434",
) -> BaseLLM:
    """
    Build the right LLM backend.

    Parameters
    ----------
    provider : str | None
        Force a specific provider. ``None`` = auto-detect from .env.
    model : str | None
        Model name. ``None`` = use the default for the resolved provider.
    ollama_host : str
        Ollama base URL (only used when provider resolves to "ollama").
    """
    resolved_provider = provider or detect_provider() or "ollama"
    resolved_model    = model or _DEFAULT_MODELS.get(resolved_provider, "llama3")

    if resolved_provider == "ollama":
        return OllamaLLM(model=resolved_model, host=ollama_host)

    if resolved_provider == "databricks":
        token      = os.environ.get("DATABRICKS_TOKEN", "")
        sonnet_ep  = os.environ.get("DATABRICKS_SONNET_ENDPOINT", "")
        opus_ep    = os.environ.get("DATABRICKS_OPUS_ENDPOINT", "")
        endpoint   = opus_ep if "opus" in resolved_model.lower() and opus_ep else sonnet_ep
        if token and endpoint:
            return DatabricksLLM(model=resolved_model, api_key=token, endpoint=endpoint)
        return OllamaLLM(model=_DEFAULT_MODELS["ollama"], host=ollama_host)

    if resolved_provider == "openai":
        return OpenAICompatibleLLM(
            model=resolved_model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url="https://api.openai.com/v1",
            provider="openai",
        )

    if resolved_provider == "anthropic":
        return AnthropicLLM(
            model=resolved_model,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    if resolved_provider == "google":
        return GoogleLLM(
            model=resolved_model,
            api_key=os.environ.get("GOOGLE_API_KEY", ""),
        )

    return OllamaLLM(model=resolved_model, host=ollama_host)


def build_llm_messages(
    system_prompt: str,
    user_query:    str,
    context:       str,
) -> list[dict]:
    """Return the messages list compatible with all provider backends."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"{context}\n\nQuestion: {user_query}"},
    ]

