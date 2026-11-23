"""Proportion of days covered (PDC), the standard pharmacy-quality adherence measure."""
from __future__ import annotations

import numpy as np
import pandas as pd


def covered_days(fills: pd.DataFrame, start: int, end: int) -> np.ndarray:
    """Boolean coverage for days [start, end) for ONE member.

    Early refills carry their supply forward (the standard PDC shift), so stockpiled
    medication counts once the previous supply runs out instead of double-counting.
    """
    covered = np.zeros(end - start, dtype=bool)
    next_free = -(10**9)
    for fill_day, supply in fills[["fill_day", "days_supply"]].itertuples(index=False):
        if fill_day >= end:
            break
        begin = max(fill_day, next_free)
        stop = begin + supply
        lo, hi = max(begin, start), min(stop, end)
        if hi > lo:
            covered[lo - start : hi - start] = True
        next_free = stop
    return covered


def pdc(fills: pd.DataFrame, start: int, end: int) -> float:
    return float(covered_days(fills, start, end).mean())
