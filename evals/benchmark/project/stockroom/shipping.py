"""Parcel shipping rates."""

from __future__ import annotations

import math

# (maximum weight in kg, price in cents), lightest band first.
RATES = [(0.5, 499), (2.0, 899), (10.0, 1599)]
PER_KG_OVER = 150  # each started kilogram above the heaviest band


def shipping_cost(weight_kg: float) -> int:
    """Cost in cents to ship a parcel weighing `weight_kg` kilograms."""
    if weight_kg <= 0:
        return 0
    for limit, price in RATES:
        if weight_kg <= limit:
            return price
    extra = weight_kg - RATES[-1][0]
    return RATES[-1][1] + math.ceil(extra) * PER_KG_OVER
