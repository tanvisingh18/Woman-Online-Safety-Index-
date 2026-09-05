"""
Joint Monte Carlo weight sensitivity (Faculty Step 7).

Perturbs all hand-set weights simultaneously; reports rank stability.

Output: outputs/results/monte_carlo_sensitivity.json

Run: PYTHONPATH=src python src/monte_carlo_sensitivity.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def _perturb_weight(base: float, rng: np.random.Generator, delta: float = 0.25) -> float:
    lo = base * (1 - delta)
    hi = base * (1 + delta)
    return float(rng.uniform(lo, hi))


def _normalize_triple(a: float, b: float, c: float) -> tuple[float, float, float]:
    s = a + b + c
    if s <= 0:
        return 0.45, 0.30, 0.25
    return a / s, b / s, c / s


def monte_carlo_whshi(
    features: pd.DataFrame,
    n_runs: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    platforms = features["platform"].tolist()
    rank_counts = {p: {p2: 0 for p2 in platforms} for p in platforms}

    for _ in range(n_runs):
        w_wtshi = _perturb_weight(0.70, rng)
        w_fuzzy = 1.0 - w_wtshi
        scores = {}
        for _, row in features.iterrows():
            wtshi = float(row.get("wtshi_construct", 0) or 0)
            fuzzy = float(row.get("fuzzy_component", 25) or 25)
            scores[row["platform"]] = w_wtshi * wtshi + w_fuzzy * fuzzy
        order = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
        for i, p in enumerate(order):
            for j, p2 in enumerate(order):
                if j <= i:
                    rank_counts[p][p2] += 1

    # Rank stability: fraction of runs where platform X is rank 1
    rank1_stability = {}
    for p in platforms:
        rank1_stability[p] = round(rank_counts[p][p] / n_runs, 3)

    live = [p for p in platforms if p in ("YouTube", "Reddit", "Telegram")]
    live_orders = []
    for _ in range(n_runs):
        w_wtshi = _perturb_weight(0.70, rng)
        w_fuzzy = 1.0 - w_wtshi
        scores = {}
        for _, row in features.iterrows():
            if row["platform"] not in live:
                continue
            wtshi = float(row.get("wtshi_construct", 0) or 0)
            fuzzy = float(row.get("fuzzy_component", 25) or 25)
            scores[row["platform"]] = w_wtshi * wtshi + w_fuzzy * fuzzy
        live_orders.append(tuple(sorted(scores.keys(), key=lambda p: scores[p], reverse=True)))

    unique_orders = len(set(live_orders))
    most_common_order = max(set(live_orders), key=live_orders.count)
    stability_pct = live_orders.count(most_common_order) / n_runs

    return {
        "metric": "WHSI_raw_weight_perturbation",
        "n_runs": n_runs,
        "perturbation": "+/-25% on 0.70/0.30 WTSHI/fuzzy",
        "live_platforms": live,
        "most_common_rank_order": list(most_common_order),
        "rank_order_stable_pct": round(stability_pct, 3),
        "rank_order_stable_above_90pct": stability_pct >= 0.90,
        "rank1_by_platform": rank1_stability,
    }


def monte_carlo_mri(n_runs: int = 1000, seed: int = 42) -> dict:
    from mri_engine import speed_score
    from structural_platform_evidence import adjust_consistency_score, adjust_removal_rate

    rng = np.random.default_rng(seed)
    trans = pd.read_csv("data/transparency/moderation_reports.csv")
    live = trans[trans["platform"].isin(["YouTube", "Reddit", "Telegram"])]

    orders = []
    for _ in range(n_runs):
        wr = _perturb_weight(0.45, rng)
        ws = _perturb_weight(0.30, rng)
        wc = _perturb_weight(0.25, rng)
        wr, ws, wc = _normalize_triple(wr, ws, wc)
        max_h = _perturb_weight(720, rng, delta=0.5)

        scores = {}
        for _, row in live.iterrows():
            plat = row["platform"]
            rem_adj, _ = adjust_removal_rate(float(row["removal_rate"]), plat)
            con_adj, _ = adjust_consistency_score(float(row["consistency_score"]), plat)
            R = rem_adj * 100
            S = speed_score(row["avg_response_hours"], max_hours=max_h)
            C = con_adj * 100
            scores[plat] = wr * R + ws * S + wc * C
        orders.append(tuple(sorted(scores.keys(), key=lambda p: scores[p], reverse=True)))

    most_common = max(set(orders), key=orders.count)
    stability = orders.count(most_common) / n_runs

    return {
        "metric": "MRI_joint_weight_perturbation",
        "n_runs": n_runs,
        "perturbation": "+/-25% on removal/speed/consistency weights + horizon",
        "most_common_rank_order": list(most_common),
        "rank_order_stable_pct": round(stability, 3),
        "rank_order_stable_above_90pct": stability >= 0.90,
    }


def run_monte_carlo() -> dict:
    features_path = "data/processed/whsi_scores_integrated.csv"
    if not os.path.exists(features_path):
        return {"error": "Run integrated_scoring.py first"}

    features = pd.read_csv(features_path)
    live_feats = features[features["platform"].isin(["YouTube", "Reddit", "Telegram"])]

    result = {
        "whsi_raw": monte_carlo_whshi(live_feats),
        "mri": monte_carlo_mri(),
        "weight_rationale_doc": "docs/WEIGHT_RATIONALE_APPENDIX.md",
    }

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/monte_carlo_sensitivity.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"WHSI rank stability: {result['whsi_raw']['rank_order_stable_pct']:.1%}")
    print(f"MRI rank stability: {result['mri']['rank_order_stable_pct']:.1%}")
    return result


if __name__ == "__main__":
    run_monte_carlo()
