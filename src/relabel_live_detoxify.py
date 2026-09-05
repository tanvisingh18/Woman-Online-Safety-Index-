"""
Apply hybrid (EDOS gendered harm + Detoxify) to live rows in an existing master CSV.

Faster than a full dataset rebuild when only classifier scores need refreshing.

Run:
  TORCH_HOME=models/torch_cache python src/relabel_live_detoxify.py
  TORCH_HOME=models/torch_cache python src/relabel_live_detoxify.py --sample 1000
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

MASTER = "data/processed/master_women_relevant.csv"


def relabel_live(
    master_path: str = MASTER,
    sample: int = 0,
) -> pd.DataFrame:
    from dataset_builder import tag_content_severity
    from hybrid_classifier import label_dataframe

    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Missing {master_path} — run dataset_builder first.")

    master = pd.read_csv(master_path, low_memory=False)
    live_mask = master.get("dataset_split", "").astype(str) == "live_scrape"
    if not live_mask.any():
        raise RuntimeError("No live_scrape rows found in master.")

    live_df = master.loc[live_mask].copy()
    if sample:
        live_df = live_df.sample(min(sample, len(live_df)), random_state=42)
        relabel_mask = live_df.index
        print(f"Relabelling sample of {len(live_df)} live rows…")
    else:
        relabel_mask = live_mask
        print(f"Relabelling all {live_mask.sum()} live rows with Detoxify hybrid…")

    labelled = label_dataframe(live_df)
    for col in labelled.columns:
        if col not in master.columns:
            master[col] = pd.NA
        master.loc[relabel_mask, col] = labelled[col].values

    master["whsi_harm_flag"] = (
        master["is_gendered_harm"].fillna(master.get("is_harmful")).fillna(0).astype(int)
    )
    master = tag_content_severity(master)

    tmp = master_path + ".tmp"
    master.to_csv(tmp, index=False)
    os.replace(tmp, master_path)
    master.to_csv("data/processed/master_dataset.csv", index=False)

    live = master[master.get("dataset_split", "") == "live_scrape"]
    print(f"\nDone. Live gendered harm rate: {live['is_gendered_harm'].mean():.2%}")
    if "detoxify_toxicity" in live.columns:
        print(f"Detoxify toxicity mean: {live['detoxify_toxicity'].mean():.3f}")
    print(f"Saved → {master_path}")
    return master


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=MASTER)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    relabel_live(master_path=args.input, sample=args.sample)
