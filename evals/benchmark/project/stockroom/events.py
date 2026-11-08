"""In-process event bus for stock changes."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

Listener = Callable[[str, str], None]


class EventBus:
    def __init__(self):
        self._listeners: List[Tuple[Optional[str], Listener]] = []
        self.history: List[Tuple[str, str]] = []

    def subscribe(self, fn: Listener, kind: Optional[str] = None) -> None:
        """Call `fn(kind, payload)` for events of `kind` (all events if None)."""
        self._listeners.append((kind, fn))

    def emit(self, kind: str, payload: str) -> None:
        self.history.append((kind, payload))
        for want, fn in self._listeners:
            if want is None or want == kind:
                try:
                    fn(kind, payload)
                except Exception:
                    # Listeners are best-effort (audit log, metrics); a broken one
                    # must never stop a stock change from completing.
                    pass

    def kinds(self) -> List[str]:
        return [k for k, _ in self.history]


def counters(bus: EventBus, kinds: Iterable[str]) -> Dict[str, int]:
    """Subscribe one counter per event kind; the returned dict updates as events arrive."""
    counts = {k: 0 for k in kinds}
    for kind in counts:
        bus.subscribe(lambda _k, _p: counts.__setitem__(kind, counts[kind] + 1), kind=kind)
    return counts
