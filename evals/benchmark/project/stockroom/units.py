"""Weight conversions. Items store grams; carriers quote in kilograms."""

from __future__ import annotations

GRAMS_PER_KG = 1000


def grams_to_kg(grams: float) -> float:
    return grams / GRAMS_PER_KG


def kg_to_grams(kg: float) -> float:
    return kg * GRAMS_PER_KG
