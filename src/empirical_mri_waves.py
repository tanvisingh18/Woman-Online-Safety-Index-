"""
Two-wave empirical MRI for Reddit — compare scrape snapshots by comment_id.

Wave 1: saved snapshot (existing pullpush CSV)
Wave 2: fresh Pullpush pull on same subreddits (immediate re-scrape proxy)

Output:
  data/scraped/reddit_wave1.parquet
  data/scraped/reddit_wave2.parquet
  outputs/results/empirical_mri_waves.json

Run: python src/empirical_mri_waves.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scraper"))


WAVE1_SRC = "data/scraped/reddit_women_pullpush.csv"
WAVE1_OUT = "data/scraped/reddit_wave1.parquet"
WAVE2_OUT = "data/scraped/reddit_wave2.parquet"
YT_WAVE1_SRC = "data/scraped/youtube_replies_ytdlp.csv"
YT_WAVE1_OUT = "data/scraped/youtube_wave1.parquet"
YT_WAVE2_OUT = "data/scraped/youtube_wave2.parquet"
EMPIRICAL_ALL_OUT = "outputs/results/empirical_mri_all.json"
HOURS_BETWEEN = 48.0


def _label_wave(df: pd.DataFrame) -> pd.DataFrame:
    if "is_gendered_harm" not in df.columns or df["is_gendered_harm"].isna().all():
        from gendered_harm_model import label_dataframe

        df = label_dataframe(df, text_col="comment_text", batch_size=256)
    df["is_harmful"] = df.get("is_gendered_harm", df.get("is_harmful", 0)).fillna(0).astype(int)
    return df


def save_wave1(source: str = WAVE1_SRC, out: str = WAVE1_OUT) -> pd.DataFrame:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = pd.read_csv(source, low_memory=False)
    df = _label_wave(df)
    df["wave"] = 1
    df["wave_saved_at"] = datetime.utcnow().isoformat()
    if "comment_id" not in df.columns and "post_id" in df.columns:
        df["comment_id"] = df["post_id"].astype(str) + "||" + df.get("author", "").astype(str)
    df.to_parquet(out, index=False)
    print(f"Wave 1 saved: {len(df)} rows → {out}")
    return df


def scrape_wave2(per_sub: int = 200) -> pd.DataFrame:
    from reddit_pullpush import scrape_all

    print("\n=== WAVE 2: fresh Pullpush scrape ===")
    df = scrape_all(per_sub=per_sub)
    if df.empty:
        return df
    df = _label_wave(df)
    df["wave"] = 2
    df["wave_saved_at"] = datetime.utcnow().isoformat()
    if "comment_id" not in df.columns and "post_id" in df.columns:
        df["comment_id"] = df["post_id"].astype(str) + "||" + df.get("author", "").astype(str)
    df.to_parquet(WAVE2_OUT, index=False)
    print(f"Wave 2 saved: {len(df)} rows → {WAVE2_OUT}")
    return df


def measure_two_wave_mri(
    wave1_path: str = WAVE1_OUT,
    wave2_path: str = WAVE2_OUT,
    hours_between: float = HOURS_BETWEEN,
) -> dict:
    from mri_engine import compute_mri, compute_enhanced_mri, mri_label

    if not os.path.exists(wave1_path) or not os.path.exists(wave2_path):
        return {"error": "missing wave files"}

    w1 = pd.read_parquet(wave1_path)
    w2 = pd.read_parquet(wave2_path)

    id_col = "comment_id" if "comment_id" in w1.columns else None
    if not id_col:
        w1["comment_id"] = w1["post_id"].astype(str) + "||" + w1.get("author", "").astype(str)
        w2["comment_id"] = w2["post_id"].astype(str) + "||" + w2.get("author", "").astype(str)
        id_col = "comment_id"

    w1_ids = set(w1[id_col].astype(str))
    w2_ids = set(w2[id_col].astype(str))
    removed = w1_ids - w2_ids

    harm_col = "is_gendered_harm" if "is_gendered_harm" in w1.columns else "is_harmful"
    harmful = w1[w1[harm_col].astype(int) == 1]
    harmful_ids = set(harmful[id_col].astype(str))
    n_harmful = len(harmful_ids)
    n_removed = len(removed & harmful_ids)

    removal_rate = n_removed / (n_harmful + 1e-9)
    avg_hours = hours_between / 2.0

    empirical = {
        "platform": "Reddit",
        "empirical_removal_rate": round(removal_rate, 4),
        "avg_response_hours_estimate": avg_hours,
        "survivor_rate": round(1 - removal_rate, 4),
        "n_harmful_wave1": int(n_harmful),
        "n_removed_in_window": int(n_removed),
        "hours_between_waves": hours_between,
        "method": "two_wave_pullpush_comment_id",
        "wave1_path": wave1_path,
        "wave2_path": wave2_path,
    }

    trans = pd.read_csv("data/transparency/moderation_reports.csv")
    reddit_row = trans[trans["platform"] == "Reddit"].iloc[0]
    blended = compute_enhanced_mri(reddit_row, empirical, empirical_weight=0.45)

    empirical["MRI_transparency_only"] = compute_mri(
        reddit_row["removal_rate"], reddit_row["avg_response_hours"], reddit_row["consistency_score"]
    )
    empirical["MRI_blended"] = blended
    empirical["MRI_label"] = mri_label(blended)

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/empirical_mri_waves.json", "w") as f:
        json.dump(empirical, f, indent=2)

    print(f"\n[TWO-WAVE EMPIRICAL MRI — Reddit]")
    print(f"  Harmful in wave 1     : {n_harmful}")
    print(f"  Missing in wave 2     : {n_removed} ({removal_rate:.2%})")
    print(f"  Blended Reddit MRI    : {blended:.2f} ({empirical['MRI_label']})")
    return empirical


def save_youtube_wave1(source: str = YT_WAVE1_SRC, out: str = YT_WAVE1_OUT) -> pd.DataFrame:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.exists(source):
        raise FileNotFoundError(f"Missing {source}")
    df = pd.read_csv(source, low_memory=False)
    df = _label_wave(df)
    df["wave"] = 1
    df["wave_saved_at"] = datetime.utcnow().isoformat()
    if "comment_id" not in df.columns:
        raise ValueError("YouTube scrape needs comment_id column")
    df["uid"] = df["video_id"].astype(str) + "||" + df["comment_id"].astype(str)
    df.to_parquet(out, index=False)
    print(f"YouTube wave 1 saved: {len(df)} rows → {out}")
    return df


def scrape_youtube_wave2(max_videos: int = 25, wave1_path: str = YT_WAVE1_OUT) -> pd.DataFrame:
    """Re-fetch comment threads from video manifest (not limited to wave1 single-video bug)."""
    scraper_dir = os.path.join(os.path.dirname(__file__), "scraper")
    if scraper_dir not in sys.path:
        sys.path.insert(0, scraper_dir)
    from youtube_replies_ytdlp import scrape_video_threads, _ytdlp_bin, load_video_manifest

    ytdlp = _ytdlp_bin()
    manifest = load_video_manifest()
    if os.path.exists(wave1_path):
        w1 = pd.read_parquet(wave1_path)
        if "video_id" in w1.columns and w1["video_id"].nunique() > 1:
            top_vids = w1["video_id"].value_counts().head(max_videos).index.tolist()
            manifest = manifest[manifest["video_id"].isin(top_vids)]
    if len(manifest) > max_videos:
        manifest = manifest.head(max_videos)

    print(f"\n=== YOUTUBE WAVE 2: re-fetch {len(manifest)} videos ===")
    frames = []
    for _, row in manifest.iterrows():
        vid = str(row["video_id"])
        ch = str(row.get("channel_title", "unknown"))
        try:
            df = scrape_video_threads(vid, ch, ytdlp)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"  skip {vid}: {exc}")

    if not frames:
        return pd.DataFrame()

    out_df = pd.concat(frames, ignore_index=True)
    out_df = _label_wave(out_df)
    out_df["wave"] = 2
    out_df["wave_saved_at"] = datetime.utcnow().isoformat()
    out_df["uid"] = out_df["video_id"].astype(str) + "||" + out_df["comment_id"].astype(str)
    out_df.to_parquet(YT_WAVE2_OUT, index=False)
    print(f"YouTube wave 2 saved: {len(out_df)} rows → {YT_WAVE2_OUT}")
    return out_df


def measure_youtube_two_wave(
    wave1_path: str = YT_WAVE1_OUT,
    wave2_path: str = YT_WAVE2_OUT,
    hours_between: float = HOURS_BETWEEN,
) -> dict:
    from mri_engine import compute_mri, compute_enhanced_mri, mri_label

    if not os.path.exists(wave1_path) or not os.path.exists(wave2_path):
        return {"error": "missing youtube wave files", "platform": "YouTube"}

    w1 = pd.read_parquet(wave1_path)
    w2 = pd.read_parquet(wave2_path)

    id_col = "uid" if "uid" in w1.columns else "comment_id"
    w1_ids = set(w1[id_col].astype(str))
    w2_ids = set(w2[id_col].astype(str))
    removed = w1_ids - w2_ids

    harm_col = "is_gendered_harm" if "is_gendered_harm" in w1.columns else "is_harmful"
    harmful = w1[w1[harm_col].astype(int) == 1]
    harmful_ids = set(harmful[id_col].astype(str))
    n_harmful = len(harmful_ids)
    n_removed = len(removed & harmful_ids)
    removal_rate = n_removed / (n_harmful + 1e-9)

    empirical = {
        "platform": "YouTube",
        "empirical_removal_rate": round(removal_rate, 4),
        "avg_response_hours_estimate": hours_between / 2.0,
        "survivor_rate": round(1 - removal_rate, 4),
        "n_harmful_wave1": int(n_harmful),
        "n_removed_in_window": int(n_removed),
        "n_videos_rescraped": int(w2["video_id"].nunique()) if "video_id" in w2.columns else 0,
        "hours_between_waves": hours_between,
        "method": "two_wave_ytdlp_comment_id",
        "wave1_path": wave1_path,
        "wave2_path": wave2_path,
        "caveat": "Comment disappearance may reflect creator moderation, not only platform removal",
    }

    trans = pd.read_csv("data/transparency/moderation_reports.csv")
    yt_row = trans[trans["platform"] == "YouTube"].iloc[0]
    empirical["MRI_transparency_only"] = compute_mri(
        yt_row["removal_rate"], yt_row["avg_response_hours"], yt_row["consistency_score"]
    )
    empirical["MRI_blended"] = compute_enhanced_mri(yt_row, empirical, empirical_weight=0.40)
    empirical["MRI_label"] = mri_label(empirical["MRI_blended"])

    print(f"\n[TWO-WAVE EMPIRICAL MRI — YouTube]")
    print(f"  Harmful in wave 1     : {n_harmful}")
    print(f"  Missing in wave 2     : {n_removed} ({removal_rate:.2%})")
    print(f"  Blended YouTube MRI   : {empirical['MRI_blended']:.2f} ({empirical['MRI_label']})")
    return empirical


def measure_reddit_removal_flags(
    path: str = "data/scraped/reddit_harassment_dataset.csv",
) -> dict:
    """Empirical removal from is_removed/is_deleted flags in labelled Reddit scrape."""
    from mri_engine import compute_mri, compute_enhanced_mri, mri_label

    if not os.path.exists(path):
        return {"error": "missing harassment dataset", "platform": "Reddit"}

    df = pd.read_csv(path, low_memory=False)
    df = _label_wave(df)
    harm_col = "is_gendered_harm" if "is_gendered_harm" in df.columns else "is_harmful"
    harmful = df[df[harm_col].astype(int) == 1]
    n_harmful = len(harmful)
    if n_harmful == 0:
        return {"error": "no harmful rows", "platform": "Reddit"}

    removed = harmful[
        (harmful.get("is_removed", 0).astype(int) == 1)
        | (harmful.get("is_deleted", 0).astype(int) == 1)
    ]
    removal_rate = len(removed) / (n_harmful + 1e-9)

    empirical = {
        "platform": "Reddit",
        "empirical_removal_rate": round(removal_rate, 4),
        "avg_response_hours_estimate": 72.0,
        "survivor_rate": round(1 - removal_rate, 4),
        "n_harmful_wave1": int(n_harmful),
        "n_removed_in_window": int(len(removed)),
        "method": "removal_flags_harassment_scrape",
        "source_path": path,
    }

    trans = pd.read_csv("data/transparency/moderation_reports.csv")
    reddit_row = trans[trans["platform"] == "Reddit"].iloc[0]
    empirical["MRI_transparency_only"] = compute_mri(
        reddit_row["removal_rate"], reddit_row["avg_response_hours"], reddit_row["consistency_score"]
    )
    empirical["MRI_blended"] = compute_enhanced_mri(reddit_row, empirical, empirical_weight=0.35)
    empirical["MRI_label"] = mri_label(empirical["MRI_blended"])
    print(f"\n[EMPIRICAL MRI — Reddit removal flags] {len(removed)}/{n_harmful} = {removal_rate:.2%}")
    return empirical


def _write_empirical_all(reddit: dict, youtube: dict | None = None) -> dict:
    os.makedirs("outputs/results", exist_ok=True)
    payload = {"Reddit": reddit}
    if youtube and "error" not in youtube:
        payload["YouTube"] = youtube
    with open(EMPIRICAL_ALL_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    with open("outputs/results/empirical_mri_waves.json", "w") as f:
        json.dump(reddit, f, indent=2)
    return payload


def run_full(hours_between: float = HOURS_BETWEEN, youtube_videos: int = 25, resrape_reddit: bool = False) -> dict:
    reddit = measure_reddit_removal_flags()

    if resrape_reddit:
        save_wave1()
        scrape_wave2(per_sub=180)
        tw = measure_two_wave_mri(hours_between=hours_between)
        if tw.get("empirical_removal_rate", 0) > reddit.get("empirical_removal_rate", 0):
            reddit = tw

    youtube = {}
    if os.environ.get("SKIP_YOUTUBE_EMPIRICAL_MRI", "").lower() in ("1", "true", "yes"):
        print("  YouTube empirical MRI skipped (SKIP_YOUTUBE_EMPIRICAL_MRI=1)")
    else:
        try:
            save_youtube_wave1()
            scrape_youtube_wave2(max_videos=youtube_videos)
            youtube = measure_youtube_two_wave(hours_between=hours_between)
        except Exception as exc:
            print(f"YouTube two-wave skipped: {exc}")
            youtube = {"error": str(exc), "platform": "YouTube"}

    _write_empirical_all(reddit, youtube)
    return {"Reddit": reddit, "YouTube": youtube}


def run_youtube_only(max_videos: int = 30) -> dict:
    save_youtube_wave1()
    scrape_youtube_wave2(max_videos=max_videos)
    yt = measure_youtube_two_wave()
    all_data = {}
    if os.path.exists(EMPIRICAL_ALL_OUT):
        with open(EMPIRICAL_ALL_OUT) as f:
            all_data = json.load(f)
    all_data["YouTube"] = yt
    with open(EMPIRICAL_ALL_OUT, "w") as f:
        json.dump(all_data, f, indent=2)
    return yt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--youtube-only", action="store_true")
    parser.add_argument("--max-youtube-videos", type=int, default=30)
    args = parser.parse_args()
    if args.youtube_only:
        run_youtube_only(max_videos=args.max_youtube_videos)
    else:
        run_full(youtube_videos=args.max_youtube_videos)
