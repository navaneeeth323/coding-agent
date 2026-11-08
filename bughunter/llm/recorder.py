"""Response cache plus full trace of every LLM call, in the state database.

Every call is recorded (request, response, tokens, latency), including cache
hits, so per-stage cost and behaviour can be analysed and prompts debugged by
reading traces. The cache key covers the model, its settings, the messages, the
schema, temperature and seed, so any change to the code in a prompt misses.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import List, Optional

from .base import ChatResult, JSONChat, LLMError, Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY,
    run_id            INTEGER,
    stage             TEXT NOT NULL,
    subject           TEXT,              -- what the call was about, e.g. "path::qualname"
    cache_key         TEXT NOT NULL,
    model             TEXT NOT NULL,
    request           TEXT NOT NULL,     -- JSON: messages, schema, settings
    response          TEXT,              -- raw model output; NULL on error
    thinking          TEXT,
    error             TEXT,
    cached            INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    seconds           REAL NOT NULL DEFAULT 0,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_calls_key ON llm_calls(cache_key);
CREATE INDEX IF NOT EXISTS llm_calls_run ON llm_calls(run_id, stage);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


class RecordingLLM:
    def __init__(
        self,
        inner: JSONChat,
        conn: sqlite3.Connection,
        stage: str,
        run_id: Optional[int] = None,
        use_cache: bool = True,
    ):
        self.inner = inner
        self.model = inner.model
        self.conn = conn
        self.stage = stage
        self.run_id = run_id
        self.use_cache = use_cache
        ensure_schema(conn)

    def _settings(self) -> dict:
        # Anything on the client that changes outputs (e.g. Ollama's `think`) must be in the key.
        return {
            k: v for k, v in vars(self.inner).items()
            if k not in ("host", "timeout") and isinstance(v, (str, int, float, bool, type(None)))
        }

    def chat_json(
        self,
        messages: List[Message],
        schema: dict,
        *,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        subject: Optional[str] = None,
    ) -> ChatResult:
        request = {
            "settings": self._settings(),
            "messages": messages,
            "schema": schema,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        # Stored as sent (schema property order matters to the model); keyed canonically.
        request_json = json.dumps(request)
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

        if self.use_cache:
            hit = self.conn.execute(
                "SELECT response, thinking FROM llm_calls WHERE cache_key = ? AND error IS NULL"
                " AND cached = 0 ORDER BY id DESC LIMIT 1",
                (key,),
            ).fetchone()
            if hit is not None:
                self._record(key, subject, request_json, hit["response"], hit["thinking"], None, cached=True)
                return ChatResult(
                    data=json.loads(hit["response"]), text=hit["response"], model=self.model,
                    thinking=hit["thinking"], cached=True,
                )

        try:
            result = self.inner.chat_json(
                messages, schema, temperature=temperature, seed=seed, max_tokens=max_tokens
            )
        except LLMError as exc:
            self._record(key, subject, request_json, None, None, str(exc))
            raise
        self._record(key, subject, request_json, result.text, result.thinking, None, result=result)
        return result

    def _record(self, key, subject, request_json, response, thinking, error, cached=False, result=None):
        with self.conn:
            self.conn.execute(
                "INSERT INTO llm_calls (run_id, stage, subject, cache_key, model, request, response,"
                " thinking, error, cached, prompt_tokens, completion_tokens, seconds, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.run_id, self.stage, subject, key, self.model, request_json, response,
                    thinking, error, int(cached),
                    result.prompt_tokens if result else 0,
                    result.completion_tokens if result else 0,
                    result.seconds if result else 0.0,
                    time.time(),
                ),
            )
