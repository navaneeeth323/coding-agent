"""A small time-based cache."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Hashable, Optional, Tuple


class TTLCache:
    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl_seconds
        self.clock = clock
        self._data: Dict[Hashable, Tuple[Any, float]] = {}

    def get(self, key: Hashable) -> Optional[Any]:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires = entry
        if self.clock() > expires:
            del self._data[key]
            return None
        return value

    def set(self, key: Hashable, value: Any) -> None:
        self._data[key] = (value, self.clock() + self.ttl)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
