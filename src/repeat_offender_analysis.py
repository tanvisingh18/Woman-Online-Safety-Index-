"""
Repeat-offender concentration analysis (YouTube + Reddit; Telegram lacks author IDs).

Outputs:
  outputs/results/repeat_offender_analysis.csv
  outputs/results/repeat_offender_analysis.json
  outputs/plots/repeat_offender_distribution.png

Run: PYTHONPATH=src python src/repeat_offender_analysis.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _hhi(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman Index on share vector (sums to 1)."""
    if len(shares) == 0:
        return float("nan")
    return float(np.sum(np.square(shares)))


def analyze_platform(g: pd.DataFrame, platform: str) -> dict:
    perp = g[g["harm_role"] == "perpetrator_attack"].copy()
    # drop missing / deleted authors
    perp = perp[
        perp["author"].notna()
        & (~perp["author"].astype(str).str.strip().isin(["", "nan", "None", "[deleted]", "AutoModerator"]))
    ]
    total_perp = len(perp)
    if total_perp == 0:
        return {
            "platform": platform,
            "total_perpetrator_comments": 0,
            "unique_perpetrator_users": 0,
            "users_with_2plus_comments_in_corpus": 0,
            "top_10pct_user_share": None,
            "HHI": None,
            "interpretation": "no_perpetrator_rows_with_author",
            "status": "skipped",
        }

    counts = perp.groupby("author").size().sort_values(ascending=False)
    n_users = len(counts)
    shares = (counts / total_perp).values
    hhi = _hhi(shares)

    # top 10% of perpetrator users by attack count
    k = max(1, int(np.ceil(0.10 * n_users)))
    top_share = float(counts.iloc[:k].sum() / total_perp)

    # users with ≥2 comments in full platform live scrape (any role)
    all_auth = g[
        g["author"].notna()
        & (~g["author"].astype(str).str.strip().isin(["", "nan", "None", "[deleted]"]))
    ]
    user_n = all_auth.groupby("author").size()
    multi = int((user_n >= 2).sum())

    if hhi < 0.01:
        band = "highly_diffuse"
    elif hhi <= 0.15:
        band = "moderately_concentrated"
    else:
        band = "highly_concentrated"

    return {
        "platform": platform,
        "total_perpetrator_comments": int(total_perp),
        "unique_perpetrator_users": int(n_users),
        "users_with_2plus_comments_in_corpus": multi,
        "top_10pct_user_count": int(k),
        "top_10pct_user_share": round(top_share, 4),
        "HHI": round(hhi, 6),
        "concentration_band": band,
        "mean_attacks_per_perp_user": round(float(counts.mean()), 3),
        "max_attacks_by_single_user": int(counts.iloc[0]),
        "status": "ok",
        "counts_for_plot": counts.value_counts().sort_index().to_dict(),
    }


def run(
    master_path: str = "data/processed/master_women_relevant.csv",
) -> pd.DataFrame:
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    df = pd.read_csv(master_path, low_memory=False)
    live = df[df["dataset_split"] == "live_scrape"].copy()
    if "author" not in live.columns:
        raise SystemExit("No author column in master — cannot run repeat-offender analysis")

    results = []
    plot_data = {}
    for plat, g in live.groupby("platform"):
        # Telegram typically has no author IDs in this scrape
        if g["author"].notna().sum() < 50:
            results.append(
                {
                    "platform": plat,
                    "total_perpetrator_comments": int((g["harm_role"] == "perpetrator_attack").sum()),
                    "unique_perpetrator_users": None,
                    "users_with_2plus_comments_in_corpus": None,
                    "top_10pct_user_share": None,
                    "HHI": None,
                    "concentration_band": None,
                    "status": "no_author_metadata",
                    "note": "Author IDs missing for this platform scrape — analysis not run",
                }
            )
            continue
        row = analyze_platform(g, plat)
        plot_data[plat] = row.pop("counts_for_plot", {})
        results.append(row)

    out = pd.DataFrame(results)
    # drop plot helper if present
    export_cols = [
        c
        for c in out.columns
        if c not in ("counts_for_plot",)
    ]
    out[export_cols].to_csv("outputs/results/repeat_offender_analysis.csv", index=False)

    # Plot: attacks-per-user distribution
    plats = [p for p in plot_data if plot_data[p]]
    if plats:
        fig, axes = plt.subplots(1, len(plats), figsize=(5 * len(plats), 4), squeeze=False)
        for ax, plat in zip(axes[0], plats):
            dist = plot_data[plat]
            xs = sorted(int(k) for k in dist.keys())
            ys = [dist[x] if x in dist else dist[str(x)] for x in xs]
            ax.bar(xs, ys, color="#2E5A88", width=0.8)
            ax.set_xlabel("Perpetrator attacks per user")
            ax.set_ylabel("Number of users")
            ax.set_title(f"{plat}")
            ax.set_xlim(0.5, max(xs) + 0.5)
        fig.suptitle("Repeat-offender distribution (live scrape, author-linked)", fontsize=12)
        fig.tight_layout()
        fig.savefig("outputs/plots/repeat_offender_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    payload = {
        "method": (
            "Among live_scrape rows with non-null author, restrict to harm_role=perpetrator_attack. "
            "HHI = sum of squared user shares of perpetrator comments. "
            "top_10pct_user_share = share of perpetrator comments from the top ceil(10%×n_users) users by count."
        ),
        "hhi_bands": {
            "<0.01": "highly_diffuse",
            "0.01–0.15": "moderately_concentrated",
            ">0.15": "highly_concentrated",
        },
        "rows": out[export_cols].to_dict(orient="records"),
    }
    with open("outputs/results/repeat_offender_analysis.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(out[export_cols].to_string(index=False))
    print("\nSaved → outputs/results/repeat_offender_analysis.csv")
    print("Saved → outputs/plots/repeat_offender_distribution.png")
    return out


if __name__ == "__main__":
    run()
