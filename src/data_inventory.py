"""
Data inventory — documents every row source for thesis transparency.

Output: outputs/results/data_inventory.json + data_inventory.csv

Run: python src/data_inventory.py
"""

from __future__ import annotations

import json
import os

import pandas as pd


def build_inventory(master_path: str = "data/processed/master_women_relevant.csv") -> dict:
    df = pd.read_csv(master_path, low_memory=False)
    inv = {
        "total_rows": len(df),
        "platforms": df["platform"].value_counts().to_dict(),
        "by_source": df["dataset_source"].value_counts().to_dict(),
        "by_split": df.get("dataset_split", pd.Series(["unknown"] * len(df))).value_counts().to_dict(),
        "live_scrape_rows": int((df.get("dataset_split", "") == "live_scrape").sum()),
        "labelled_anchor_rows": int(df.get("dataset_source", "").astype(str).str.contains("EDOS", na=False).sum()),
        "proxy_rows": int(
            df.get("dataset_source", "").astype(str).str.contains("twitter_gab_proxy|gab_1M", na=False).sum()
        ),
        "twitter_rows": int((df["platform"] == "Twitter").sum()),
        "twitter_live_scrape_rows": int(
            ((df["platform"] == "Twitter") & (df.get("dataset_split", "") == "live_scrape")).sum()
        ),
        "gendered_harm_rate": round(float(df["is_gendered_harm"].mean()), 4) if "is_gendered_harm" in df.columns else None,
        "severe_language_rate": round(float(df["has_severe_language"].mean()), 4) if "has_severe_language" in df.columns else None,
        "youtube_replies": int(((df["platform"] == "YouTube") & (df.get("thread_depth", 0) > 0)).sum()) if "thread_depth" in df.columns else 0,
        "notes": {
            "whsi_platform_scores": "Use live_scrape rows only for fair cross-platform comparison",
            "edos": "Training/validation anchor only — not presented as live Reddit scrape",
            "gab_proxy": "Twitter stand-in until live X scrape available",
            "community_column": "YouTube channels / Reddit subreddits / Telegram groups — NOT mixed in one column",
        },
    }

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/data_inventory.json", "w") as f:
        json.dump(inv, f, indent=2)

    rows = []
    for src, n in inv["by_source"].items():
        sub = df[df["dataset_source"] == src]
        rows.append({
            "dataset_source": src,
            "n_rows": n,
            "platforms": ", ".join(sub["platform"].unique()),
            "gendered_harm_rate": round(sub["is_gendered_harm"].mean(), 4) if "is_gendered_harm" in sub.columns else None,
            "split": sub["dataset_split"].mode()[0] if "dataset_split" in sub.columns else "",
        })
    pd.DataFrame(rows).to_csv("outputs/results/data_inventory.csv", index=False)
    print(json.dumps(inv, indent=2))
    return inv


if __name__ == "__main__":
    build_inventory()
