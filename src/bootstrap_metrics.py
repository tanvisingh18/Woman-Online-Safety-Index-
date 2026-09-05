"""
Bootstrap confidence intervals for platform harm rates.

Output: outputs/results/harm_rate_confidence_intervals.csv

Run: python src/bootstrap_metrics.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from corpus_config import platform_analysis_mask
from feature_extraction import _harm_column


def bootstrap_rate(
    flags: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    flags = flags.astype(int)
    n = len(flags)
    if n == 0:
        return {"rate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(flags, size=n, replace=True)
        means.append(sample.mean())

    means = np.array(means)
    return {
        "rate": round(float(flags.mean()), 4),
        "ci_low": round(float(np.percentile(means, 2.5)), 4),
        "ci_high": round(float(np.percentile(means, 97.5)), 4),
        "n": int(n),
    }


def compute_platform_cis(
    master_path: str = "data/processed/master_women_relevant.csv",
    live_only: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(master_path, low_memory=False)
    if live_only:
        df = df[platform_analysis_mask(df)].copy()

    rows = []
    for plat, g in df.groupby("platform"):
        harm_col = _harm_column(g)
        gendered = g[harm_col].astype(int).values if harm_col in g.columns else np.zeros(len(g))
        perp = (g.get("harm_role", "") == "perpetrator_attack").astype(int).values
        targets = g.get("targets_women", pd.Series(0, index=g.index)).astype(int).values

        perp_ci = bootstrap_rate(perp)
        perp_women = perp & targets
        perp_women_ci = bootstrap_rate(perp_women)
        gendered_ci = bootstrap_rate(gendered)

        rows.append(
            {
                "platform": plat,
                "n_comments": len(g),
                "perpetrator_harm_rate": perp_ci["rate"],
                "perpetrator_ci_low": perp_ci["ci_low"],
                "perpetrator_ci_high": perp_ci["ci_high"],
                "women_targeted_perpetrator_rate": perp_women_ci["rate"],
                "women_targeted_ci_low": perp_women_ci["ci_low"],
                "women_targeted_ci_high": perp_women_ci["ci_high"],
                "gendered_harm_rate": gendered_ci["rate"],
                "gendered_ci_low": gendered_ci["ci_low"],
                "gendered_ci_high": gendered_ci["ci_high"],
                "is_proxy_data": int(
                    bool(
                        (g["is_proxy"].fillna(0).max() if "is_proxy" in g.columns else 0)
                        or g.get("dataset_source", "")
                        .astype(str)
                        .str.contains("gab_proxy", case=False, na=False)
                        .any()
                    )
                ),
                "is_historical_data": int(
                    bool(
                        (g["is_historical"].fillna(0).max() if "is_historical" in g.columns else 0)
                        or g.get("dataset_split", pd.Series([""]))
                        .astype(str)
                        .eq("historical_twitter")
                        .any()
                    )
                ),
            }
        )

    out = pd.DataFrame(rows).sort_values("perpetrator_harm_rate", ascending=False)
    from platform_labels import add_display_columns

    out = add_display_columns(out, proxy_col="is_proxy_data")
    os.makedirs("outputs/results", exist_ok=True)
    out.to_csv("outputs/results/harm_rate_confidence_intervals.csv", index=False)
    print("\nHARM RATE 95% BOOTSTRAP CI (perpetrator attacks):")
    disp = "platform_display" if "platform_display" in out.columns else "platform"
    print(
        out[
            [
                disp,
                "n_comments",
                "perpetrator_harm_rate",
                "perpetrator_ci_low",
                "perpetrator_ci_high",
                "is_proxy_data",
            ]
        ].to_string(index=False)
    )
    return out


if __name__ == "__main__":
    compute_platform_cis()
