import pandas as pd

from adherence.dataset import build_table
from adherence.pdc import covered_days, pdc
from adherence.simulate import simulate


def fills(rows):
    return pd.DataFrame(rows, columns=["fill_day", "days_supply"])


def test_pdc_fully_covered():
    assert pdc(fills([(0, 30), (30, 30), (60, 30)]), 0, 90) == 1.0


def test_pdc_counts_gap():
    # 30 days covered, 30-day gap, 30 days covered.
    assert pdc(fills([(0, 30), (60, 30)]), 0, 90) == 2 / 3


def test_early_refill_carries_supply_forward():
    # Refilled 10 days early: the extra supply shifts forward instead of double counting.
    cov = covered_days(fills([(0, 30), (20, 30)]), 0, 70)
    assert cov[:60].all() and not cov[60:].any()


def test_build_table_has_no_future_features():
    members, f = simulate(n_members=300, seed=1)
    table = build_table(members, f, [200])
    assert set(table["label_nonadherent"].unique()) <= {0, 1}
    # Lookback features must only use fills before the index date.
    assert (table["days_since_last_fill"] > 0).all()
    assert (table["first_fill_in_window"] >= 200 - 180).all()
    assert "latent_propensity" not in table.columns
