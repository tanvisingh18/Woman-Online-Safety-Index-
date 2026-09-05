"""
Fetch YouTube comment THREADS (replies included) via yt-dlp — no API key needed.

Reads video IDs from data/scraped/youtube_master_videos.csv and writes:
  data/scraped/youtube_replies_ytdlp.csv

Run:
  python src/scraper/youtube_replies_ytdlp.py
  python src/scraper/youtube_replies_ytdlp.py --max-videos 10
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VIDEO_LIST = "data/scraped/youtube_master_videos.csv"
OUT_PATH = "data/scraped/youtube_replies_ytdlp.csv"


def _ytdlp_bin() -> str:
    for candidate in ("yt-dlp", "ytdlp"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("yt-dlp not found. Install: pip install yt-dlp")


def fetch_comments_json(video_id: str, ytdlp: str) -> list[dict]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        ytdlp,
        "--skip-download",
        "--write-comments",
        "--dump-single-json",
        "--no-warnings",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    return payload.get("comments") or []


def _flatten_comment(
    node: dict,
    video_id: str,
    channel_title: str,
    thread_depth: int,
    parent_id: str | None,
    rows: list[dict],
) -> None:
    cid = node.get("id", "")
    text = (node.get("text") or "").strip()
    if len(text) < 2:
        return

    rows.append(
        {
            "comment_text": text[:8000],
            "platform": "YouTube",
            "community": channel_title,
            "community_type": "youtube_channel",
            "subreddit": channel_title,
            "video_id": video_id,
            "comment_id": cid,
            "parent_comment_id": parent_id or "",
            "thread_depth": thread_depth,
            "author": node.get("author", ""),
            "score": node.get("like_count", 0),
            "reply_count": node.get("_total_reply_count", len(node.get("replies") or [])),
            "created_utc": node.get("timestamp"),
            "dataset_source": "yt-dlp_replies",
            "dataset_split": "women_relevant",
            "content_category": "reply" if thread_depth > 0 else "comment",
        }
    )

    for reply in node.get("replies") or []:
        _flatten_comment(reply, video_id, channel_title, thread_depth + 1, cid, rows)


def scrape_video_threads(video_id: str, channel_title: str, ytdlp: str) -> pd.DataFrame:
    comments = fetch_comments_json(video_id, ytdlp)
    rows: list[dict] = []
    for top in comments:
        _flatten_comment(top, video_id, channel_title, 0, None, rows)
    return pd.DataFrame(rows)


def load_video_manifest(path: str = VIDEO_LIST) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    if "video_id" not in df.columns:
        raise ValueError("youtube_master_videos.csv must have video_id column")
    channel_col = "channel_title" if "channel_title" in df.columns else "channel_id"
    return df[["video_id", channel_col]].drop_duplicates("video_id").rename(
        columns={channel_col: "channel_title"}
    )


def scrape_all(max_videos: int | None = None, output_path: str = OUT_PATH) -> pd.DataFrame:
    os.makedirs("data/scraped", exist_ok=True)
    ytdlp = _ytdlp_bin()
    manifest = load_video_manifest()
    if max_videos:
        manifest = manifest.head(max_videos)

    frames = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="YouTube threads"):
        vid = row["video_id"]
        ch = str(row.get("channel_title", "unknown"))
        try:
            df = scrape_video_threads(vid, ch, ytdlp)
            if not df.empty:
                frames.append(df)
                # Incremental save so partial runs are usable
                partial = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["comment_id"])
                partial["collected_at"] = datetime.utcnow().isoformat()
                partial.to_csv(output_path, index=False)
        except subprocess.TimeoutExpired:
            print(f"  timeout: {vid}")
        except Exception as exc:
            print(f"  error {vid}: {exc}")

    if not frames:
        print("No reply data collected.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.drop_duplicates(subset=["comment_id"], inplace=True)

    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        out = pd.concat([prev, out], ignore_index=True).drop_duplicates(subset=["comment_id"])

    out["collected_at"] = datetime.utcnow().isoformat()
    out.to_csv(output_path, index=False)

    depth_counts = out["thread_depth"].value_counts().sort_index()
    print(f"\nYouTube replies saved → {output_path}")
    print(f"  Total rows: {len(out)} | replies (depth>0): {(out['thread_depth']>0).sum()}")
    print(f"  Depth distribution:\n{depth_counts.to_string()}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--output", default=OUT_PATH)
    args = parser.parse_args()
    scrape_all(max_videos=args.max_videos, output_path=args.output)
