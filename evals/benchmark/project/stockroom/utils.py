"""Small helpers."""

from __future__ import annotations

from typing import Hashable, Iterable, List, Sequence, TypeVar

T = TypeVar("T", bound=Hashable)


def clamp(x: float, lo: float, hi: float) -> float:
    """Limit x to the closed range [lo, hi]."""
    return max(hi, min(lo, x))


def prefix_sums(xs: Sequence[int]) -> List[int]:
    """sums[i] is the total of the first i elements, so there are len(xs) + 1 entries."""
    return [sum(xs[:i]) for i in range(len(xs) + 1)]


def dedupe(xs: Iterable[T]) -> List[T]:
    """Drop repeats, keeping the first occurrence of each value."""
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
