#!/usr/bin/env python3
"""
Women Safety Index — end-to-end pipeline orchestrator

Runs all document sections in order:
  4  preprocessing        → master_dataset.csv
  10 sentiment/influence  → master_with_influence.parquet (optional)
  5  feature extraction   → platform_features.csv
  6  fuzzy WHSI           → whsi_scores.csv
  7  MRI                  → mri_scores.csv
  8  safety matrix        → plots + CSV
  9  hotspot detection    → community_whsi.csv
  11 hashtag analysis     → hashtag_danger.csv

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-influence
  python run_pipeline.py --merge-scrape data/scraped/reddit_live.parquet
"""

from __future__ import annotations

import argparse
import os
import sys

# Project root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)


def step_build_women_relevant(label: bool = True):
    from dataset_builder import build_women_relevant

    print("\n>>> DATASET: Women-relevant master (no Reddit 1M bulk)")
    return build_women_relevant(label=label)


def step_preprocessing(use_legacy: bool = False):
    from preprocessing import build_master_dataset

    print("\n>>> SECTION 4: Preprocessing (legacy public datasets)")
    return build_master_dataset()


def step_merge_scrape(master_path: str, scrape_path: str):
    import pandas as pd

    print(f"\n>>> Merging live scrape: {scrape_path}")
    master = pd.read_csv(master_path)
    if scrape_path.endswith(".parquet"):
        scrape = pd.read_parquet(scrape_path)
    else:
        scrape = pd.read_csv(scrape_path)

    if scrape["is_harmful"].isna().any():
        from classifier import classify_dataframe

        scrape = classify_dataframe(scrape)

    # Align columns
    core = [
        "comment_text",
        "platform",
        "subreddit",
        "hashtags",
        "is_harmful",
        "toxicity_score",
        "threat_score",
        "dataset_source",
        "target_gender",
    ]
    for c in core:
        if c not in scrape.columns:
            scrape[c] = None

    merged = pd.concat([master, scrape[core]], ignore_index=True)
    merged.drop_duplicates(subset=["comment_text", "platform"], inplace=True)
    merged.to_csv(master_path, index=False)
    print(f"Merged master: {len(merged)} rows → {master_path}")
    return merged


def step_influence(master_path: str, vader_only: bool = True):
    import pandas as pd
    from sentiment_influence import analyze_dataframe, plot_influence_distribution

    print("\n>>> SECTION 10: Sentiment & Influence (folded into integrated WHSI)")
    master = pd.read_csv(master_path)
    live = master[master.get("dataset_split", "") == "live_scrape"]
    if len(live) < len(master):
        print(f"  Scoring {len(live)} live rows (VADER layer)")
        live_scored = analyze_dataframe(live.copy(), vader_only=vader_only)
        for col in live_scored.columns:
            if col not in master.columns:
                master[col] = pd.NA
            master.loc[live.index, col] = live_scored[col].values
    else:
        master = analyze_dataframe(master, vader_only=vader_only)
    out = "data/processed/master_with_influence.parquet"
    master.to_parquet(out, index=False)
    master.to_csv("data/processed/master_dataset.csv", index=False)
    if "influence_type" in master.columns:
        plot_influence_distribution(master)
    return out


def _master_path() -> str:
    """Prefer canonical women-relevant master (avoids stale influence parquet)."""
    for path in (
        "data/processed/master_women_relevant.csv",
        "data/processed/master_dataset.csv",
        "data/processed/master_with_influence.parquet",
    ):
        if os.path.exists(path):
            return path
    return "data/processed/master_dataset.csv"


def step_integrated_scoring():
    from integrated_scoring import run_integrated_pipeline

    print("\n>>> INTEGRATED SCORING: Hotspots + threads + sentiment → WHSI")
    path = _master_path()
    print(f"  Using master: {path}")
    return run_integrated_pipeline(path)


def step_features(use_influence: bool = False, live_only: bool = True):
    import pandas as pd
    from feature_extraction import extract_platform_features
    from sentiment_influence import apply_influence_weighting

    print("\n>>> SECTION 5: Feature Extraction")
    if use_influence and os.path.exists("data/processed/master_with_influence.parquet"):
        master = pd.read_parquet("data/processed/master_with_influence.parquet")
    else:
        master = pd.read_csv("data/processed/master_dataset.csv")

    features = extract_platform_features(master, live_only=live_only)

    if use_influence and "influence_score" in master.columns:
        features = apply_influence_weighting(features, master)
        # Use influence-weighted values for fuzzy engine when available
        for plat in features["platform"]:
            row = features[features["platform"] == plat].iloc[0]
            if pd.notna(row.get("toxicity_influence_weighted")):
                features.loc[features["platform"] == plat, "toxicity"] = row[
                    "toxicity_influence_weighted"
                ]
            if pd.notna(row.get("threat_incitement_weighted")):
                features.loc[features["platform"] == plat, "threat"] = row[
                    "threat_incitement_weighted"
                ]
        features.to_csv("data/processed/platform_features.csv", index=False)

    return features


def step_whsi(integrated: bool = True):
    import pandas as pd
    from fuzzy_engine import run_whsi_for_all_platforms, plot_membership_functions

    if integrated and os.path.exists("data/processed/whsi_scores_integrated.csv"):
        print("\n>>> SECTION 6: WHSI — using integrated scores (hotspot/thread/sentiment)")
        whsi = pd.read_csv("data/processed/whsi_scores_integrated.csv")
        plot_membership_functions()
        return whsi

    print("\n>>> SECTION 6: WHSI (Mamdani Fuzzy)")
    features = pd.read_csv("data/processed/platform_features.csv")
    whsi = run_whsi_for_all_platforms(features)
    plot_membership_functions()
    return whsi


def step_mri(run_two_wave: bool = True):
    import json
    import pandas as pd
    from mri_engine import compute_mri_from_transparency, plot_mri_breakdown, run_empirical_mri_and_blend, mri_label

    print("\n>>> SECTION 7 + 13: MRI (transparency + empirical two-wave)")

    if run_two_wave:
        try:
            from empirical_mri_waves import run_full as run_empirical_mri

            print("  Running empirical MRI (Reddit removal flags)…")
            import os as _os
            _os.environ["SKIP_YOUTUBE_EMPIRICAL_MRI"] = "1"
            emp = run_empirical_mri(hours_between=48.0, youtube_videos=15, resrape_reddit=False)
            if emp.get("Reddit", {}).get("MRI_blended"):
                print(f"  Reddit empirical MRI: {emp['Reddit']['MRI_blended']}")
        except Exception as exc:
            print(f"  Empirical MRI skipped: {exc}")

    mri = compute_mri_from_transparency()
    blended = run_empirical_mri_and_blend()

    empirical_plats = set()
    for tw_path in ("outputs/results/empirical_mri_all.json", "outputs/results/empirical_mri_waves.json"):
        if not os.path.exists(tw_path):
            continue
        with open(tw_path) as f:
            tw_data = json.load(f)
        entries = tw_data if isinstance(tw_data, dict) and "platform" in tw_data else tw_data
        if isinstance(entries, dict) and "platform" in entries:
            entries = {"Reddit": entries}
        for plat, tw in entries.items():
            if not isinstance(tw, dict) or "MRI_blended" not in tw:
                continue
            mask = mri["platform"] == plat
            if not mask.any():
                continue
            mri.loc[mask, "MRI_score"] = tw["MRI_blended"]
            mri.loc[mask, "MRI_label"] = tw.get("MRI_label", mri_label(tw["MRI_blended"]))
            method = tw.get("method", "two_wave")
            mri.loc[mask, "source"] = f"{plat} Transparency + {method} empirical blend"
            empirical_plats.add(plat)

    if blended is not None and not blended.empty:
        for _, brow in blended.iterrows():
            plat = brow["platform"]
            if plat in empirical_plats:
                continue
            if pd.notna(brow.get("empirical_removal_rate")):
                mask = mri["platform"] == plat
                mri.loc[mask, "MRI_score"] = brow["MRI_score"]
                mri.loc[mask, "MRI_label"] = brow["MRI_label"]
                mri.loc[mask, "source"] = (
                    mri.loc[mask, "source"].astype(str)
                    + " + empirical blend ("
                    + str(brow.get("empirical_method", "scrape"))
                    + ")"
                )

    mri.to_csv("data/processed/mri_scores.csv", index=False)
    mri.to_csv("data/processed/mri_scores_integrated.csv", index=False)
    plot_mri_breakdown(mri)
    return mri


def step_safety_matrix():
    from safety_matrix import build_safety_matrix, plot_safety_matrix, print_safety_summary

    print("\n>>> SECTION 8: Safety Matrix")
    matrix = build_safety_matrix()
    plot_safety_matrix(matrix)
    print_safety_summary(matrix)
    return matrix


def step_hotspots():
    import pandas as pd
    from hotspot_detection import (
        build_hotspot_leaderboard,
        compute_community_whsi,
        detect_emerging_hotspots,
        plot_hotspot_heatmap,
        plot_whsi_timeseries,
        compute_time_window_whsi,
        _community_column,
    )

    print("\n>>> SECTION 9 EXT: Hotspot Detection")
    path = "data/processed/master_with_influence.parquet"
    if os.path.exists(path):
        master = pd.read_parquet(path)
    else:
        master = pd.read_csv("data/processed/master_dataset.csv")

    community_df = compute_community_whsi(master)
    if not community_df.empty:
        build_hotspot_leaderboard(community_df, top_n=10)
        plot_hotspot_heatmap(community_df, top_n=15)
        if "created_utc" in master.columns:
            top_comm = community_df.iloc[0]["community"]
            ts = compute_time_window_whsi(master, top_comm, community_col=_community_column(master))
            if not ts.empty:
                plot_whsi_timeseries(ts, top_comm)
        detect_emerging_hotspots(master, community_col=_community_column(master))

    # Hashtag communities (Twitter extension)
    if master["hashtags"].notna().any():
        hashtag_df = compute_community_whsi(
            master[master["hashtags"].astype(str).str.len() > 1],
            community_col="hashtags",
            min_comments=15,
        )
        if not hashtag_df.empty:
            hashtag_df.to_csv("outputs/results/hashtag_community_whsi.csv", index=False)

    return community_df


def step_subreddit_analytics():
    import pandas as pd
    from subreddit_analytics import (
        compute_subreddit_profile,
        compute_youtube_channel_profiles,
        fetch_subreddit_metadata,
    )

    print("\n>>> SECTION 12: Subreddit Deep Analytics")
    path = "data/processed/master_with_influence.parquet"
    master = pd.read_parquet(path) if os.path.exists(path) else pd.read_csv(
        "data/processed/master_dataset.csv"
    )
    subs = master[master["platform"] == "Reddit"]["subreddit"].dropna().unique().tolist()[:30]
    meta = fetch_subreddit_metadata(subs) if subs else None
    compute_subreddit_profile(master, meta)
    compute_youtube_channel_profiles(master)


def step_ingest_scraped(
    youtube_path: str | None = None,
    reddit_million: str | None = None,
    telegram_path: str | None = None,
    reddit_sample: int = 50_000,
):
    from load_scraped import load_all_scraped, merge_into_master

    print("\n>>> INGEST: User scraped datasets")
    frames = load_all_scraped(
        youtube_path=youtube_path,
        reddit_million_path=reddit_million,
        telegram_path=telegram_path,
        reddit_sample_n=reddit_sample,
    )
    merge_into_master(frames, rebuild_from_public=True)


def step_hashtags():
    import pandas as pd
    from hashtag_analysis import (
        compute_hashtag_danger,
        build_hashtag_cooccurrence_graph,
        plot_hashtag_danger_bar,
    )

    print("\n>>> SECTION 11: Hashtag Analysis")
    path = "data/processed/master_with_influence.parquet"
    master = pd.read_parquet(path) if os.path.exists(path) else pd.read_csv(
        "data/processed/master_dataset.csv"
    )
    danger = compute_hashtag_danger(master)
    if not danger.empty:
        plot_hashtag_danger_bar(danger)
        build_hashtag_cooccurrence_graph(master)
    return danger


def main():
    parser = argparse.ArgumentParser(description="Women Safety Index — full pipeline")
    parser.add_argument(
        "--skip-influence",
        action="store_true",
        help="Skip Section 10 (faster; no SBERT)",
    )
    parser.add_argument(
        "--merge-scrape",
        default=None,
        help="Path to classified scrape parquet/csv to merge into master",
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated: build,ingest,preprocess,influence,features,whsi,mri,matrix,hotspots,subreddit,hashtags",
    )
    parser.add_argument(
        "--youtube",
        default=None,
        help="Path to youtube_master_comments.csv (your scraped YouTube data)",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only merge scraped files into master, then exit",
    )
    parser.add_argument(
        "--legacy-preprocess",
        action="store_true",
        help="Use old public-dataset preprocessing instead of women-relevant builder",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Skip rebuild+label step (use when master is already labelled)",
    )
    parser.add_argument("--reddit-million", default="../reddit_1M_unlabelled.csv")
    parser.add_argument("--telegram", default=None, help="Telegram export CSV path")
    parser.add_argument("--reddit-sample", type=int, default=50000)
    args = parser.parse_args()

    steps = (
        [
            "build",
            "influence",
            "features",
            "hotspots",
            "integrated",
            "whsi",
            "mri",
            "matrix",
            "subreddit",
            "hashtags",
        ]
        if args.steps == "all"
        else [s.strip() for s in args.steps.split(",")]
    )

    master_path = "data/processed/master_dataset.csv"

    if "ingest" in steps or args.youtube or args.telegram:
        step_ingest_scraped(
            youtube_path=args.youtube,
            reddit_million=args.reddit_million,
            telegram_path=args.telegram,
            reddit_sample=args.reddit_sample,
        )

    if args.ingest_only:
        return

    if "build" in steps or ("preprocess" in steps and not args.legacy_preprocess):
        step_build_women_relevant(label=not args.no_label)
    elif "preprocess" in steps:
        step_preprocessing(use_legacy=True)

    if args.merge_scrape and os.path.exists(args.merge_scrape):
        step_merge_scrape(master_path, args.merge_scrape)

    if "influence" in steps and not args.skip_influence:
        step_influence(master_path, vader_only=True)
    elif "influence" in steps:
        print("\n>>> SECTION 10: Skipped (--skip-influence)")

    use_inf = "influence" in steps and not args.skip_influence

    if "features" in steps:
        step_features(use_influence=use_inf)
    if "hotspots" in steps:
        step_hotspots()
    if "integrated" in steps:
        step_integrated_scoring()
    if "whsi" in steps:
        step_whsi(integrated="integrated" in steps)
    if "mri" in steps:
        step_mri(run_two_wave=True)
    if "matrix" in steps:
        step_safety_matrix()
    if "subreddit" in steps:
        step_subreddit_analytics()
    if "hashtags" in steps:
        step_hashtags()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("Key outputs:")
    print("  data/processed/whsi_scores.csv")
    print("  data/processed/mri_scores.csv")
    print("  outputs/plots/safety_matrix.png")
    print("  outputs/results/community_whsi.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
