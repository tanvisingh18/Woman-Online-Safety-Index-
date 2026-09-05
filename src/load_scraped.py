"""
Load and normalize user-uploaded scraped data into master_dataset schema.
Supports: YouTube master CSV, Reddit/Twitter parquet, generic CSV.

Run:
  python src/load_scraped.py --all
  python src/load_scraped.py --youtube ../dataset/data/youtube_master_comments.csv
  python src/load_scraped.py --file data/scraped/reddit_live.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

MASTER_COLS = [
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

TEXT_ALIASES = ["comment_text", "text", "body", "content", "tweet", "comment"]
SCORE_ALIASES = ["score", "like_count", "likes", "upvotes"]
TIME_ALIASES = ["created_utc", "published_at", "timestamp", "created_at"]

_PLATFORM_CANON = {
    "reddit": "Reddit",
    "youtube": "YouTube",
    "twitter": "Twitter",
    "x": "Twitter",
    "telegram": "Telegram",
    "gab": "Gab",
}


def _canon_platform(value: str) -> str:
    if value is None:
        return "Unknown"
    v = str(value).strip()
    key = v.lower()
    return _PLATFORM_CANON.get(key, v)


def _first_col(df: pd.DataFrame, aliases: list) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _tier_to_toxicity(tier: str) -> float:
    m = {"none": 0.05, "low": 0.35, "medium": 0.65, "high": 0.90}
    return m.get(str(tier).lower().strip(), 0.40)


def load_youtube_all(
    master_path: str = "../dataset/data/youtube_master_comments.csv",
    ytdlp_path: str = "../dataset/data/youtube_comments_ytdlp.csv",
) -> pd.DataFrame:
    """
    Merge all YouTube tables: comments + subcomments from every export.
    Deduplicates on comment_id (or text+video_id fallback).
    """
    frames = []
    for p in [master_path, ytdlp_path]:
        if os.path.exists(p):
            frames.append(load_youtube_master(p))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "comment_id" in combined.columns:
        combined.drop_duplicates(subset=["comment_id"], inplace=True)
    else:
        combined.drop_duplicates(subset=["comment_text", "video_id"], inplace=True)
    print(f"[YouTube] Combined {len(combined)} unique comments/subcomments")
    return combined


def load_reddit_raw_text(
    path: str,
    sample_n: int | None = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load reddit_1M_unlabelled.csv (single 'text' column).
    Full 1M rows is too heavy for interactive runs — default sample 50k stratified by length.
    Set sample_n=None to load entire file (slow; run classifier overnight).
    """
    print(f"[Reddit] Reading {path} …")
    df = pd.read_csv(path, usecols=["text"] if "text" in pd.read_csv(path, nrows=0).columns else None)
    text_col = "text" if "text" in df.columns else df.columns[0]
    df = df.rename(columns={text_col: "comment_text"})
    df["comment_text"] = df["comment_text"].astype(str)
    df = df[df["comment_text"].str.len() > 5]

    if sample_n and len(df) > sample_n:
        df["_len"] = df["comment_text"].str.len()
        df = df.sort_values("_len").reset_index(drop=True)
        # Evenly spaced sample across length distribution
        idx = np.linspace(0, len(df) - 1, sample_n, dtype=int)
        df = df.iloc[idx].drop(columns=["_len"])
        print(f"[Reddit] Sampled {sample_n:,} from {path} (set sample_n=None for full file)")

    out = pd.DataFrame(
        {
            "comment_text": df["comment_text"],
            "platform": "Reddit",
            "subreddit": "reddit_scraped_pool",
            "hashtags": df["comment_text"].apply(
                lambda t: ",".join(re.findall(r"#(\w+)", str(t).lower()))
            ),
            "is_harmful": np.nan,
            "toxicity_score": np.nan,
            "threat_score": np.nan,
            "dataset_source": "reddit_1M_unlabelled",
            "target_gender": "mixed",
        }
    )
    print(f"[Reddit] Prepared {len(out)} rows")
    return out


def load_telegram_export(path: str) -> pd.DataFrame:
    """
    Normalize Telegram scrape exports.
    Expected columns (any alias): text/message, chat_title/group/channel, date, from_user
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    text_col = _first_col(df, ["text", "message", "comment_text", "body", "content"])
    if not text_col:
        raise ValueError(f"No message text column in {path}")

    chat_col = _first_col(df, ["chat_title", "group", "channel", "community", "chat_name", "subreddit"])
    user_col = _first_col(df, ["from_user", "author", "sender", "username"])
    time_col = _first_col(df, TIME_ALIASES)

    out = pd.DataFrame(
        {
            "comment_text": df[text_col].astype(str),
            "platform": "Telegram",
            "subreddit": df[chat_col].astype(str) if chat_col else "telegram_unknown",
            "hashtags": df[text_col].astype(str).apply(
                lambda t: ",".join(re.findall(r"#(\w+)", t.lower()))
            ),
            "is_harmful": np.nan,
            "toxicity_score": np.nan,
            "threat_score": np.nan,
            "dataset_source": os.path.basename(path),
            "target_gender": "mixed",
        }
    )
    if user_col:
        out["author"] = df[user_col].astype(str)
    if time_col:
        out["created_utc"] = df[time_col]

    out = out[out["comment_text"].str.len() > 5]
    print(f"[Telegram] Loaded {len(out)} messages from {path}")
    return out


def load_youtube_master(path: str) -> pd.DataFrame:
    """Normalize youtube_master_comments.csv from your collector."""
    df = pd.read_csv(path, low_memory=False)
    text_col = _first_col(df, ["text", "comment_text"]) or "text"

    records = []
    for _, row in df.iterrows():
        text = str(row.get(text_col, ""))
        if len(text.strip()) < 5:
            continue

        tier = row.get("toxicity_tier", "none")
        kw_hit = bool(row.get("harassment_keyword_hit", False))
        tox = _tier_to_toxicity(tier)
        if kw_hit:
            tox = max(tox, 0.55)

        flag = row.get("harassment_research_flag", False)
        is_harm = int(
            flag is True
            or str(flag).lower() == "true"
            or tier in ("medium", "high")
            or kw_hit
        )

        tags = row.get("hashtags_in_text", "")
        if pd.isna(tags) or tags == "[]":
            tags = row.get("video_hashtags", "")
        if isinstance(tags, str) and tags.startswith("["):
            tags = ",".join(re.findall(r"[\w#]+", tags))

        channel = row.get("channel_title", row.get("channel_id", "unknown_channel"))
        created = row.get("published_at", row.get("created_utc", np.nan))
        cat = row.get("content_category", "comment")
        depth = row.get("thread_depth", 0)

        records.append(
            {
                "comment_text": text[:8000],
                "platform": "YouTube",
                "subreddit": str(channel)[:120],
                "content_category": cat,
                "thread_depth": depth,
                "parent_comment_id": row.get("parent_comment_id", ""),
                "hashtags": str(tags) if pd.notna(tags) else "",
                "is_harmful": is_harm,
                "toxicity_score": float(np.clip(tox, 0, 1)),
                "threat_score": float(0.7 if is_harm and kw_hit else 0.2 * is_harm),
                "dataset_source": str(row.get("data_source", "youtube_scraped")),
                "target_gender": "female",
                "score": pd.to_numeric(row.get("like_count", 0), errors="coerce"),
                "author": row.get("author_display_name", row.get("author_channel_id", "")),
                "created_utc": created,
                "video_id": row.get("video_id", ""),
                "thread_id": row.get("thread_id", row.get("video_id", "")),
            }
        )

    out = pd.DataFrame(records)
    print(f"[YouTube] Loaded {len(out)} comments from {path}")
    return out


def load_generic_scrape(path: str, platform: str | None = None) -> pd.DataFrame:
    """Best-effort normalization for arbitrary scrape files."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    text_col = _first_col(df, TEXT_ALIASES)
    if not text_col:
        raise ValueError(f"No text column found in {path}")

    plat = platform
    if not plat and "platform" in df.columns:
        plat = str(df["platform"].mode().iloc[0]) if len(df) else "Unknown"
    plat = plat or "Scraped"

    out = pd.DataFrame()
    out["comment_text"] = df[text_col].astype(str)

    if "platform" in df.columns:
        out["platform"] = df["platform"].apply(_canon_platform)
    else:
        out["platform"] = _canon_platform(plat)

    sub_col = _first_col(df, ["subreddit", "channel_title", "community", "group"])
    out["subreddit"] = df[sub_col] if sub_col else None

    if "hashtags" in df.columns:
        out["hashtags"] = df["hashtags"].fillna("")
    else:
        out["hashtags"] = out["comment_text"].apply(
            lambda t: ",".join(re.findall(r"#(\w+)", str(t).lower()))
        )

    if "is_harmful" in df.columns:
        out["is_harmful"] = pd.to_numeric(df["is_harmful"], errors="coerce").fillna(0).astype(int)
    else:
        out["is_harmful"] = np.nan

    out["toxicity_score"] = (
        pd.to_numeric(df["toxicity_score"], errors="coerce")
        if "toxicity_score" in df.columns
        else np.nan
    )
    out["threat_score"] = (
        pd.to_numeric(df["threat_score"], errors="coerce")
        if "threat_score" in df.columns
        else np.nan
    )

    out["dataset_source"] = df.get("dataset_source", os.path.basename(path))
    out["target_gender"] = df.get("target_gender", "mixed")

    sc = _first_col(df, SCORE_ALIASES)
    if sc:
        out["score"] = pd.to_numeric(df[sc], errors="coerce")

    if "author" in df.columns:
        out["author"] = df["author"]

    tc = _first_col(df, TIME_ALIASES)
    if tc:
        out["created_utc"] = df[tc]

    for c in ["post_id", "comment_id", "controversiality", "video_id", "thread_id"]:
        if c in df.columns:
            out[c] = df[c]

    out = out[out["comment_text"].str.len() > 5]
    print(f"[Generic] Loaded {len(out)} rows from {path}")
    return out


def classify_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if df["is_harmful"].notna().all() and df["toxicity_score"].notna().all():
        df["is_harmful"] = df["is_harmful"].fillna(0).astype(int)
        df["toxicity_score"] = df["toxicity_score"].fillna(0).clip(0, 1)
        df["threat_score"] = df["threat_score"].fillna(0).clip(0, 1)
        return df

    from classifier import classify_dataframe

    print("[Load] Running classifier on unlabelled rows …")
    return classify_dataframe(df)


def merge_into_master(
    scrape_frames: list[pd.DataFrame],
    master_path: str = "data/processed/master_dataset.csv",
    rebuild_from_public: bool = True,
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(master_path), exist_ok=True)

    if rebuild_from_public and os.path.exists(os.path.join(os.path.dirname(__file__), "preprocessing.py")):
        from preprocessing import build_master_dataset

        master = build_master_dataset()
    elif os.path.exists(master_path):
        master = pd.read_csv(master_path)
    else:
        master = pd.DataFrame(columns=MASTER_COLS)

    extras = []
    for df in scrape_frames:
        if df is None or df.empty:
            continue
        df = classify_if_needed(df)
        core = [c for c in MASTER_COLS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in core]
        extras.append(df[core + extra_cols])

    if not extras:
        print("[Merge] No scraped frames to merge.")
        return master

    combined = pd.concat([master] + extras, ignore_index=True)
    if "platform" in combined.columns:
        combined["platform"] = combined["platform"].apply(_canon_platform)
        # Defensive cleanup: older runs may have produced a placeholder platform label.
        combined = combined[combined["platform"] != "Scraped"]
    before = len(combined)
    combined.drop_duplicates(subset=["comment_text", "platform"], inplace=True)
    combined.to_csv(master_path, index=False)
    print(f"[Merge] master_dataset: {before} → {len(combined)} rows (deduped {before - len(combined)})")
    print(f"  Platforms: {combined['platform'].value_counts().to_dict()}")
    return combined


def discover_scraped_files(scraped_dir: str = "data/scraped") -> list[str]:
    patterns = ["*.csv", "*.parquet"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(scraped_dir, p)))
    files = sorted(set(files))
    # Skip metadata-only tables that are not comment/message corpora
    skip = {
        "youtube_hotspots.csv",
        "youtube_master_channels.csv",
        "youtube_master_videos.csv",
        "youtube_videos_ytdlp.csv",
    }
    return [f for f in files if os.path.basename(f) not in skip]


def load_all_scraped(
    scraped_dir: str = "data/scraped",
    youtube_path: str | None = None,
    reddit_million_path: str | None = None,
    reddit_sample_n: int = 50_000,
    telegram_path: str | None = None,
) -> list[pd.DataFrame]:
    frames = []

    # YouTube — merge all exports (comments + subcomments when present)
    if youtube_path:
        frames.append(load_youtube_master(youtube_path))
    else:
        yt_combined = load_youtube_all(
            master_path=os.path.join(scraped_dir, "youtube_master_comments.csv"),
            ytdlp_path=os.path.join(scraped_dir, "youtube_comments_ytdlp.csv"),
        )
        if yt_combined.empty:
            yt_combined = load_youtube_all(
                master_path="../dataset/data/youtube_master_comments.csv",
                ytdlp_path="../dataset/data/youtube_comments_ytdlp.csv",
            )
        if not yt_combined.empty:
            frames.append(yt_combined)

    # Reddit 1M raw
    rp = reddit_million_path or os.path.join(scraped_dir, "reddit_1M_unlabelled.csv")
    if not os.path.exists(rp):
        rp = "../reddit_1M_unlabelled.csv"
    if os.path.exists(rp):
        frames.append(load_reddit_raw_text(rp, sample_n=reddit_sample_n))

    # Telegram
    tp = telegram_path
    if not tp:
        for name in ["telegram_live.csv", "telegram_messages.csv", "telegram_export.csv"]:
            cand = os.path.join(scraped_dir, name)
            if os.path.exists(cand):
                tp = cand
                break
    if tp and os.path.exists(tp):
        frames.append(load_telegram_export(tp))

    for path in discover_scraped_files(scraped_dir):
        base = os.path.basename(path).lower()
        if "youtube_master" in base:
            continue
        if "telegram_messages" in base or "telegram" in base:
            # Prefer load_telegram_export() above to preserve chat/group field names
            continue
        if "wave" in base:
            continue
        try:
            if "youtube" in base:
                frames.append(load_youtube_master(path))
            else:
                plat = "Reddit" if "reddit" in base else ("Twitter" if "twitter" in base else None)
                frames.append(load_generic_scrape(path, platform=plat))
        except Exception as e:
            print(f"[Skip] {path}: {e}")

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Load everything in data/scraped/")
    parser.add_argument("--file", default=None, help="Single scrape file")
    parser.add_argument("--youtube", default=None, help="YouTube master CSV path")
    parser.add_argument("--no-rebuild", action="store_true", help="Do not re-download public datasets")
    parser.add_argument("--reddit-million", default=None, help="Path to reddit_1M_unlabelled.csv")
    parser.add_argument("--reddit-sample", type=int, default=50000, help="Sample size from 1M Reddit")
    parser.add_argument("--telegram", default=None, help="Path to Telegram export CSV")
    args = parser.parse_args()

    frames = []
    if args.youtube:
        frames.append(load_youtube_master(args.youtube))
    elif args.file:
        frames.append(load_generic_scrape(args.file))
    elif args.all:
        frames = load_all_scraped(
            youtube_path=args.youtube,
            reddit_million_path=args.reddit_million,
            reddit_sample_n=args.reddit_sample,
            telegram_path=args.telegram,
        )
    else:
        parser.print_help()
        return

    merge_into_master(frames, rebuild_from_public=not args.no_rebuild)


if __name__ == "__main__":
    main()
