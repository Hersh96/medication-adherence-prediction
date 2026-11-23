"""Build the modelling table: lookback features at each index date, label from the next 180 days."""
from __future__ import annotations

import pathlib

import duckdb
import numpy as np
import pandas as pd

from adherence.pdc import pdc

SQL = pathlib.Path(__file__).resolve().parents[2] / "sql" / "refill_features.sql"
OUTCOME_DAYS = 180
PDC_THRESHOLD = 0.8  # PQA / CMS Star Ratings convention for "adherent"


def snapshots(fills: pd.DataFrame, index_day: int) -> pd.DataFrame:
    """Members on therapy at the index date: a fill in the 180-day lookback and >=60 days of history."""
    window = fills[(fills["fill_day"] < index_day) & (fills["fill_day"] >= index_day - 180)]
    first = fills.groupby("member_id")["fill_day"].min()
    eligible = window["member_id"].unique()
    eligible = [m for m in eligible if first[m] <= index_day - 60]
    return pd.DataFrame({"member_id": eligible, "index_day": index_day})


def build_table(members: pd.DataFrame, fills: pd.DataFrame, index_days: list[int]) -> pd.DataFrame:
    snaps = pd.concat([snapshots(fills, d) for d in index_days], ignore_index=True)
    con = duckdb.connect()
    con.register("fills", fills)
    con.register("snapshots", snaps)
    sql_feats = con.execute(SQL.read_text()).df()

    by_member = {m: g.sort_values("fill_day") for m, g in fills.groupby("member_id")}
    rows = []
    for m, d in snaps[["member_id", "index_day"]].itertuples(index=False):
        f = by_member[m]
        rows.append({
            "member_id": m,
            "index_day": d,
            "pdc_30": pdc(f, d - 30, d),
            "pdc_90": pdc(f, d - 90, d),
            "pdc_180": pdc(f, d - 180, d),
            "pdc_next_180": pdc(f, d, d + OUTCOME_DAYS),
        })
    pdc_df = pd.DataFrame(rows)
    table = (
        pdc_df.merge(sql_feats, on=["member_id", "index_day"])
        .merge(members.drop(columns=["latent_propensity"]), on="member_id")
    )
    table["pdc_trend"] = table["pdc_30"] - table["pdc_180"]
    table["days_supply_on_hand"] = table["days_until_runout"].clip(lower=0)
    table["label_nonadherent"] = (table["pdc_next_180"] < PDC_THRESHOLD).astype(int)
    return table


NUMERIC = [
    "pdc_30", "pdc_90", "pdc_180", "pdc_trend", "n_fills_180", "share_90_day_fills", "max_gap_days",
    "mean_gap_days", "n_late_fills", "n_late_fills_90", "days_since_last_fill", "days_supply_on_hand",
    "age", "copay", "n_chronic_meds",
]
CATEGORICAL = ["sex", "plan", "drug_class", "mail_order", "new_start"]
