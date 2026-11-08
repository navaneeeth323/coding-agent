"""Ollama's native chat API: schema-constrained output and a switch for thinking."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import List, Optional

from .base import ChatResult, LLMError, Message

DEFAULT_HOST = "http://localhost:11434"


class OllamaChat:
    def __init__(
        self,
        model: str = "qwen3.5:9b",
        host: str = DEFAULT_HOST,
        think: bool = False,
        num_ctx: int = 8192,
        timeout: float = 180.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.think = think
        self.num_ctx = num_ctx
        self.timeout = timeout

    def chat_json(
        self,
        messages: List[Message],
        schema: dict,
        *,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        options = {
            "temperature": temperature,
            "num_ctx": self.num_ctx,
            # Model defaults (e.g. qwen's presence_penalty 1.5) fight a JSON grammar.
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
        }
        if seed is not None:
            options["seed"] = seed
        if max_tokens is not None:
            # Without a cap a small model can loop inside a JSON string forever.
            options["num_predict"] = max_tokens
        body = {
            "model": self.model,
            "messages": messages,
            "format": schema,
            "think": self.think,
            "stream": False,
            "options": options,
        }
        started = time.time()
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            raise LLMError(f"ollama HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"ollama request to {self.host} failed: {exc}") from exc

        message = payload.get("message") or {}
        text = message.get("content", "")
        if payload.get("done_reason") == "length":
            raise LLMError(f"output hit the {max_tokens}-token cap: {text[-200:]!r}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned invalid JSON: {text[:300]!r}") from exc
        return ChatResult(
            data=data,
            text=text,
            model=self.model,
            prompt_tokens=payload.get("prompt_eval_count", 0),
            completion_tokens=payload.get("eval_count", 0),
            seconds=time.time() - started,
            thinking=message.get("thinking"),
        )
