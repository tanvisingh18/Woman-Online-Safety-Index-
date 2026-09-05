"""
Path C — published historical Twitter corpus (no X API).

Primary source: Davidson et al. (2017) hate-speech/offensive Twitter dataset
  https://huggingface.co/datasets/tdavidson/hate_speech_offensive

Output:
  data/scraped/twitter_historical.parquet
  data/scraped/twitter_historical_provenance.json

Run: python src/twitter_historical.py
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

WOMEN_KW = re.compile(
    r"\b(?:woman|women|female|girl|girls|she|her|bitch|slut|feminist|metoo|"
    r"harass|rape|sexist|misogyn)\b",
    re.IGNORECASE,
)

CITATION = {
    "dataset": "Davidson et al. (2017) — Automated Hate Speech Detection and the Problem of Offensive Language",
    "venue": "ICWSM 2017",
    "huggingface": "tdavidson/hate_speech_offensive",
    "url": "https://huggingface.co/datasets/tdavidson/hate_speech_offensive",
    "platform": "Twitter",
    "collection_period": "~2017 (historical; not contemporaneous with live scrapes)",
    "license_note": "Academic research use; cite original paper and HF dataset card",
}


def _load_davidson_raw(cache_path: str = "data/raw/davidson.parquet") -> pd.DataFrame:
    os.makedirs("data/raw", exist_ok=True)
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    from datasets import load_dataset

    print("Downloading Davidson Twitter corpus from HuggingFace…")
    ds = load_dataset("tdavidson/hate_speech_offensive", split="train")
    df = ds.to_pandas()
    df.to_parquet(cache_path, index=False)
    return df


def build_historical_twitter_corpus(
    sample_n: int = 2500,
    women_filter: bool = True,
    seed: int = 42,
    output_path: str = "data/scraped/twitter_historical.parquet",
) -> pd.DataFrame:
    raw = _load_davidson_raw()
    text_col = "tweet" if "tweet" in raw.columns else "text"
    class_col = "class" if "class" in raw.columns else None

    df = pd.DataFrame(
        {
            "comment_text": raw[text_col].astype(str),
            "tweet_id": raw.index.astype(str),
            "davidson_class": raw[class_col] if class_col else np.nan,
        }
    )
    df = df[df["comment_text"].str.len() > 10].copy()

    if women_filter:
        df = df[df["comment_text"].str.contains(WOMEN_KW, na=False)]
        print(f"[Historical Twitter] Women-relevant Davidson tweets: {len(df)}")

    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed).copy()

    df["platform"] = "Twitter"
    df["community"] = df["comment_text"].str.extract(r"#(\w+)", expand=False).fillna("twitter_historical")
    df["community_type"] = "historical_hashtag"
    df["subreddit"] = df["community"]
    df["hashtags"] = df["comment_text"].str.findall(r"#(\w+)").apply(
        lambda tags: ",".join(tags[:5]) if isinstance(tags, list) else ""
    )
    df["dataset_source"] = "davidson_2017_twitter"
    df["dataset_split"] = "historical_twitter"
    df["is_proxy"] = 0
    df["is_historical"] = 1
    df["target_gender"] = "female"
    df["is_harmful"] = df["davidson_class"].isin([0, 1]).astype(int) if "davidson_class" in df.columns else np.nan
    df["toxicity_score"] = df["davidson_class"].map({0: 0.90, 1: 0.60, 2: 0.05}).fillna(0.30)
    df["threat_score"] = df["davidson_class"].map({0: 0.70, 1: 0.30, 2: 0.02}).fillna(0.15)
    df["scraped_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)

    provenance = {
        **CITATION,
        "built_at": df["scraped_at"].iloc[0],
        "n_rows": int(len(df)),
        "women_keyword_filter": women_filter,
        "sample_n": sample_n,
        "class_distribution": df["davidson_class"].value_counts().to_dict() if "davidson_class" in df.columns else {},
        "output_path": output_path,
        "thesis_note": (
            "Historical published Twitter corpus (Path C). "
            "Not live X API data; label as supplementary / non-contemporaneous in platform rankings."
        ),
    }
    prov_path = output_path.replace(".parquet", "_provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"[Historical Twitter] Saved {len(df)} rows → {output_path}")
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build historical Twitter corpus (Davidson 2017)")
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--all-tweets", action="store_true", help="Disable women-keyword filter")
    args = p.parse_args()
    build_historical_twitter_corpus(sample_n=args.n, women_filter=not args.all_tweets)
