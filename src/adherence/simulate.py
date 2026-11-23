"""Generate synthetic pharmacy refill claims with known, documented behaviour.

No real patient or employer data is used. Each synthetic member has fixed traits (age,
plan, copay, mail order, number of chronic medications, drug class) and a latent
adherence propensity. Some members have a "disruption" partway through (a life event,
side effects, a cost change) after which refills become irregular. That is the signal a
model has to pick up early from recent refill behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DRUG_CLASSES = {"statin": 0.0, "metformin": -0.15, "ace_arb": 0.1, "ssri": -0.45}
PLAN_COPAY = {"commercial": (10, 25), "medicare": (5, 45), "medicaid": (0, 4)}
HORIZON_DAYS = 560


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def simulate(n_members: int = 20000, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (members, fills). fills has one row per pharmacy claim."""
    rng = np.random.default_rng(seed)
    plans = rng.choice(list(PLAN_COPAY), size=n_members, p=[0.5, 0.35, 0.15])
    copay = np.array([rng.uniform(*PLAN_COPAY[p]) for p in plans]).round(2)
    members = pd.DataFrame({
        "member_id": np.arange(1, n_members + 1),
        "age": rng.integers(30, 88, n_members),
        "sex": rng.choice(["F", "M"], n_members),
        "plan": plans,
        "copay": copay,
        "mail_order": rng.random(n_members) < 0.3,
        "n_chronic_meds": rng.integers(1, 6, n_members),
        "drug_class": rng.choice(list(DRUG_CLASSES), n_members, p=[0.4, 0.25, 0.25, 0.1]),
        "new_start": rng.random(n_members) < 0.35,
    })
    base = (
        2.1
        + 0.025 * (members["age"] - 60)
        - 0.03 * members["copay"]
        + 0.5 * members["mail_order"]
        - 0.7 * members["new_start"]
        + 0.12 * (members["n_chronic_meds"] - 3)
        + members["drug_class"].map(DRUG_CLASSES)
        + rng.normal(0, 0.9, n_members)
    ).to_numpy()
    disrupted = rng.random(n_members) < 0.3
    disruption_day = np.where(disrupted, rng.integers(60, HORIZON_DAYS - 60, n_members), 10**6)
    members["latent_propensity"] = base.round(3)  # kept for analysis, never used as a feature

    rows = []
    for i, m in enumerate(members.itertuples(index=False)):
        day = int(rng.integers(0, 45)) if m.new_start else 0
        supply = 90 if (m.mail_order and rng.random() < 0.85) else 30
        stopped = False
        while day < HORIZON_DAYS and not stopped:
            rows.append((m.member_id, day, supply))
            propensity = base[i] - (1.8 if day >= disruption_day[i] else 0.0)
            on_time = rng.random() < _sigmoid(propensity)
            gap = int(rng.integers(-4, 6)) if on_time else int(rng.exponential(20)) + 4
            # Low-propensity members sometimes stop entirely.
            stopped = rng.random() < 0.02 * _sigmoid(-propensity) * (supply / 30)
            day += supply + gap
    fills = pd.DataFrame(rows, columns=["member_id", "fill_day", "days_supply"])
    return members, fills
