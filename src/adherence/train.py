"""Simulate claims, build features, train and evaluate. Run: python -m adherence.train"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from adherence.dataset import CATEGORICAL, NUMERIC, build_table
from adherence.simulate import simulate

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS, ARTIFACTS = ROOT / "reports", ROOT / "artifacts"
SEED = 3
TRAIN_INDEX, TEST_INDEX = 200, 380  # out-of-time: test snapshots are 180 days after training ones
CAPACITY = 0.15  # outreach team can call 15% of members


def recall_at(y: np.ndarray, s: np.ndarray, cap: float) -> float:
    k = int(round(len(s) * cap))
    return float(y[np.argsort(-s)[:k]].sum() / y.sum())


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        f"recall_at_{int(CAPACITY * 100)}pct_capacity": round(recall_at(y, p, CAPACITY), 4),
        "base_rate": round(float(y.mean()), 4),
        "n": int(len(y)),
    }


def models(pos_weight: float) -> dict[str, Pipeline]:
    def prep(scale: bool) -> ColumnTransformer:
        return ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ], verbose_feature_names_out=False)

    return {
        "logistic_regression": Pipeline([("prep", prep(True)), ("model", LogisticRegression(class_weight="balanced", max_iter=3000))]),
        "xgboost": Pipeline([("prep", prep(False)), ("model", XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=4, min_child_weight=5, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, random_state=SEED, n_jobs=4))]),
    }


def main() -> None:
    for d in (REPORTS / "figures", ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)
    members, fills = simulate()
    rng = np.random.default_rng(SEED)
    ids = rng.permutation(members["member_id"].to_numpy())
    n = len(ids)
    groups = {"train": set(ids[: int(0.7 * n)]), "valid": set(ids[int(0.7 * n): int(0.85 * n)]), "test": set(ids[int(0.85 * n):])}

    early = build_table(members, fills, [TRAIN_INDEX])
    late = build_table(members, fills, [TEST_INDEX])
    data = {
        "train": early[early["member_id"].isin(groups["train"])],
        "valid": early[early["member_id"].isin(groups["valid"])],
        "test": late[late["member_id"].isin(groups["test"])],  # new members AND a later time
    }
    X = {k: v[NUMERIC + CATEGORICAL] for k, v in data.items()}
    y = {k: v["label_nonadherent"].to_numpy() for k, v in data.items()}

    # Baseline a pharmacist might use today: rank by lowest recent PDC.
    baseline = metrics(y["test"], 1 - data["test"]["pdc_90"].to_numpy())

    pos_weight = float((y["train"] == 0).sum() / (y["train"] == 1).sum())
    results, fitted = {"rule_lowest_pdc_90": baseline}, {}
    for name, pipe in models(pos_weight).items():
        pipe.fit(X["train"], y["train"])
        cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(X["valid"], y["valid"])
        results[name] = metrics(y["test"], cal.predict_proba(X["test"])[:, 1])
        fitted[name] = (pipe, cal)
        print(name, results[name])
    print("baseline", baseline)

    # Stability across drug classes (cohort check).
    best = max(fitted, key=lambda k: results[k]["pr_auc"])
    p_test = fitted[best][1].predict_proba(X["test"])[:, 1]
    by_class = {}
    for cls, idx in data["test"].groupby("drug_class").groups.items():
        mask = data["test"].index.isin(idx)
        by_class[cls] = metrics(y["test"][mask], p_test[mask])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (_, cal) in fitted.items():
        frac, mean = calibration_curve(y["test"], cal.predict_proba(X["test"])[:, 1], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name.replace("_", " "))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="perfect")
    ax.set(xlabel="Predicted probability", ylabel="Observed non-adherence rate", title="Calibration, out-of-time test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "calibration.png", dpi=150)

    pipe = fitted["xgboost"][0]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    sample = X["test"].sample(min(2000, len(X["test"])), random_state=SEED)
    tx = pd.DataFrame(prep.transform(sample), columns=prep.get_feature_names_out())
    sv = shap.TreeExplainer(model).shap_values(tx)
    plt.figure()
    shap.summary_plot(sv, tx, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS / "figures" / "shap_summary.png", dpi=150, bbox_inches="tight")
    importance = pd.Series(np.abs(sv).mean(axis=0), index=tx.columns).sort_values(ascending=False)

    joblib.dump({"model": fitted[best][1], "features": NUMERIC + CATEGORICAL, "name": best}, ARTIFACTS / "model.joblib")
    report = {
        "data": "synthetic pharmacy claims generated by adherence.simulate (seeded)",
        "members": int(len(members)), "fills": int(len(fills)),
        "splits": {k: {"rows": int(len(v)), "nonadherent_rate": round(float(v["label_nonadherent"].mean()), 4)} for k, v in data.items()},
        "test_metrics": results, "selected_model": best, "by_drug_class": by_class,
        "top_shap_features": importance.head(10).round(4).to_dict(),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
