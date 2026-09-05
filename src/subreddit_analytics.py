"""
SECTION 12 — SUBREDDIT-LEVEL DEEP ANALYTICS
Women Safety Index | src/subreddit_analytics.py

Extends hotspot detection with:
  - Cross-subreddit WHSI profiles
  - Moderator gap (mods per 1K subscribers)
  - Danger index = WHSI × (1 + 1 / (mods_per_1k + 0.1))

Output: outputs/results/subreddit_profiles.csv

Run: python src/subreddit_analytics.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.makedirs("outputs/results", exist_ok=True)

import sys

sys.path.insert(0, os.path.dirname(__file__))

from feature_extraction import (
    compute_frequency,
    compute_normalization,
    compute_threat,
    compute_toxicity,
)
from fuzzy_engine import compute_whsi


def fetch_subreddit_metadata(subreddit_list: list, reddit=None) -> pd.DataFrame:
    """
    Fetch subscribers, mod count via PRAW.
    Returns empty DataFrame if credentials missing (offline mode).
    """
    try:
        from scraper.reddit_scraper import get_reddit_client
    except ImportError:
        try:
            from reddit_scraper import get_reddit_client
        except ImportError:
            print("[Subreddit Analytics] PRAW scraper not importable.")
            return pd.DataFrame()

    if reddit is None:
        try:
            reddit = get_reddit_client()
        except Exception as e:
            print(f"[Subreddit Analytics] Reddit API unavailable: {e}")
            return pd.DataFrame()

    records = []
    for sub_name in subreddit_list:
        try:
            sub = reddit.subreddit(sub_name)
            mod_count = sum(1 for _ in sub.moderator())
            records.append(
                {
                    "subreddit": sub_name,
                    "subscribers": sub.subscribers,
                    "mod_count": mod_count,
                    "created_utc": sub.created_utc,
                    "over18": sub.over18,
                }
            )
        except Exception as e:
            print(f"  [skip] r/{sub_name}: {e}")

    return pd.DataFrame(records)


def compute_subreddit_profile(
    master_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
    community_col: str = "subreddit",
    min_comments: int = 20,
) -> pd.DataFrame:
    """WHSI + engagement per community; merge metadata for danger_index."""
    if community_col not in master_df.columns:
        print(f"[Subreddit Analytics] Missing column '{community_col}'")
        return pd.DataFrame()

    reddit_df = master_df[master_df["platform"].str.lower() == "reddit"] if "platform" in master_df.columns else master_df
    if reddit_df.empty:
        reddit_df = master_df

    results = []
    for sub, group in reddit_df[reddit_df[community_col].notna()].groupby(community_col):
        if len(group) < min_comments:
            continue

        T = compute_toxicity(group)
        Th = compute_threat(group)
        F = compute_frequency(group)
        N = compute_normalization(group)
        score, cat = compute_whsi(T, Th, F, N)

        avg_score = group["score"].mean() if "score" in group.columns else 0.0
        avg_contro = group["controversiality"].mean() if "controversiality" in group.columns else 0.0
        harm_n = int(group["is_harmful"].sum()) if "is_harmful" in group.columns else 0

        results.append(
            {
                "subreddit": sub,
                "WHSI": score,
                "category": cat,
                "n_comments": len(group),
                "n_harmful": harm_n,
                "harm_rate": round(group["is_harmful"].mean(), 4) if "is_harmful" in group.columns else 0.0,
                "avg_comment_score": round(float(avg_score), 2),
                "avg_controversiality": round(float(avg_contro), 3),
                "toxicity": T,
                "threat": Th,
                "frequency": F,
                "normalization": N,
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        return df

    if metadata_df is not None and not metadata_df.empty:
        df = df.merge(metadata_df, on="subreddit", how="left")
        df["mods_per_1k"] = (df["mod_count"] / (df["subscribers"] / 1000 + 1e-9)).round(3)
        df["moderator_burnout_proxy"] = (df["n_harmful"] / (df["mod_count"] + 1)).round(2)
        df["danger_index"] = (df["WHSI"] * (1 + 1 / (df["mods_per_1k"] + 0.1))).round(2)
    else:
        df["mods_per_1k"] = np.nan
        df["danger_index"] = df["WHSI"]

    df = df.sort_values("WHSI", ascending=False)
    out = "outputs/results/subreddit_profiles.csv"
    df.to_csv(out, index=False)
    print(f"[Subreddit Analytics] {len(df)} profiles → {out}")
    return df


def compute_youtube_channel_profiles(
    master_df: pd.DataFrame, min_comments: int = 15
) -> pd.DataFrame:
    """Section 12 analogue for YouTube — channel-level WHSI."""
    yt = master_df[master_df["platform"].str.lower() == "youtube"] if "platform" in master_df.columns else pd.DataFrame()
    if yt.empty:
        return pd.DataFrame()

    return compute_subreddit_profile(yt, metadata_df=None, community_col="subreddit", min_comments=min_comments)


if __name__ == "__main__":
    path = "data/processed/master_with_influence.parquet"
    if os.path.exists(path):
        master = pd.read_parquet(path)
    else:
        master = pd.read_csv("data/processed/master_dataset.csv")

    subs = master[master["platform"] == "Reddit"]["subreddit"].dropna().unique().tolist()[:30]
    meta = fetch_subreddit_metadata(subs) if subs else pd.DataFrame()

    profile = compute_subreddit_profile(master, meta)
    if not profile.empty:
        cols = ["subreddit", "WHSI", "category", "harm_rate", "mods_per_1k", "danger_index"]
        cols = [c for c in cols if c in profile.columns]
        print(profile[cols].head(15).to_string(index=False))

    yt_prof = compute_youtube_channel_profiles(master)
    if not yt_prof.empty:
        yt_prof.to_csv("outputs/results/youtube_channel_profiles.csv", index=False)
        print(f"\nYouTube channels profiled: {len(yt_prof)}")
