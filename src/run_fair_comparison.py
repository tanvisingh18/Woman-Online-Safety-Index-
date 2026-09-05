"""
Fair cross-platform WHSI — live scrapes only, equal sample size per platform.

Outputs:
  data/processed/platform_features_fair.csv
  data/processed/whsi_scores_fair.csv
  outputs/results/fair_comparison.csv

Run: python src/run_fair_comparison.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from corpus_config import platform_analysis_mask


def fair_sample(master: pd.DataFrame, n_per_platform: int = 2000) -> pd.DataFrame:
    live = master[platform_analysis_mask(master)].copy()
    if "dataset_split" in live.columns:
        live = live[live["dataset_split"].astype(str) == "live_scrape"]
    if "is_historical" in live.columns:
        live = live[live["is_historical"].fillna(0).astype(int) != 1]

    parts = []
    for plat, grp in live.groupby("platform"):
        n = min(n_per_platform, len(grp))
        parts.append(grp.sample(n, random_state=42))
        print(f"  {plat}: sampled {n} / {len(grp)} live rows")

    return pd.concat(parts, ignore_index=True)


def main(n_per_platform: int = 2000):
    from integrated_scoring import build_integrated_features, run_integrated_whsi
    from hotspot_detection import compute_community_whsi

    master_path = "data/processed/master_women_relevant.csv"
    if not os.path.exists(master_path):
        master_path = "data/processed/master_dataset.csv"
    master = pd.read_csv(master_path, low_memory=False)

    print(f"\nFair comparison — {n_per_platform} live rows per platform (integrated features)")
    fair = fair_sample(master, n_per_platform=n_per_platform)
    community = compute_community_whsi(fair, min_comments=10)
    features = build_integrated_features(fair, community)
    whsi = run_integrated_whsi(features, export_main_outputs=False)
    whsi.to_csv("data/processed/whsi_scores_fair.csv", index=False)

    mri = pd.read_csv("data/processed/mri_scores.csv")
    lit_cols = [
        c
        for c in [
            "WHSI_literature_adjusted",
            "WHSI_lit_sensitivity_low",
            "WHSI_lit_sensitivity_high",
            "ecosystem_harm_exposure_rate",
            "wtshi_literature",
        ]
        if c in whsi.columns
    ]
    merged = whsi.merge(mri[["platform", "MRI_score", "MRI_label"]], on="platform", how="left")
    merged["spread_rank"] = merged["WHSI_literature_adjusted"].rank(ascending=False, na_option="bottom")
    merged["spread_rank_raw"] = merged["WHSI_raw"].rank(ascending=False)

    from platform_labels import add_display_columns

    merged = add_display_columns(
        merged,
        proxy_col="is_proxy_data" if "is_proxy_data" in merged.columns else "is_proxy",
    )
    merged.to_csv("outputs/results/fair_comparison.csv", index=False)

    print("\nFAIR WHSI COMPARISON (live only, equal N):")
    disp = "platform_display" if "platform_display" in merged.columns else "platform"
    show = [disp, "n_comments", "harm_rate", "WHSI_raw", "WHSI_literature_adjusted", "MRI_score"]
    show = [c for c in show if c in merged.columns]
    print(merged[show].to_string(index=False))
    return merged


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()
    main(n_per_platform=args.n)
