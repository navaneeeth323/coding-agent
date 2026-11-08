from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

Message = Dict[str, str]  # {"role": "system" | "user" | "assistant", "content": ...}


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    data: Any  # parsed JSON
    text: str  # raw model output
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    thinking: Optional[str] = None
    cached: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class JSONChat(Protocol):
    """The one capability the pipeline needs: a chat turn constrained to a JSON schema."""

    model: str

    def chat_json(
        self,
        messages: List[Message],
        schema: dict,
        *,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult: ...
