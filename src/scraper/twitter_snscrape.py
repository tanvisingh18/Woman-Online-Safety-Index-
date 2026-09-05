"""
Twitter/X collector via snscrape (no API key).

Falls back gracefully if X blocks scraping. Output:
  data/scraped/twitter_snscrape.csv

Run:
  python src/scraper/twitter_snscrape.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
from tqdm import tqdm

HASHTAGS = [
    "MeToo",
    "WomenSafety",
    "OnlineAbuse",
    "CyberHarassment",
    "DigitalViolence",
    "WomenInTech",
]

QUERIES = [
    "women harassed online lang:en",
    "sexist women lang:en",
]

OUT_PATH = "data/scraped/twitter_snscrape.csv"


def _snscrape_bin() -> str:
    for candidate in ("snscrape", sys.executable + " -m snscrape"):
        try:
            cmd = candidate.split() + ["--version"]
            subprocess.run(cmd, capture_output=True, check=True)
            return candidate.split()[0] if " " not in candidate else sys.executable
        except Exception:
            continue
    return "snscrape"


def scrape_query(query: str, max_tweets: int = 150) -> pd.DataFrame:
    """Run snscrape CLI and parse JSONL output."""
    import json
    import tempfile

    ytdlp_style = f"twitter-search:{query}"
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()

    cmd = [sys.executable, "-m", "snscrape", "--jsonl", "-n", str(max_tweets), "twitter-search", query]
    if __import__("shutil").which("snscrape"):
        cmd = ["snscrape", "--jsonl", "-n", str(max_tweets), "twitter-search", query]

    try:
        with open(tmp.name, "w") as out_f:
            proc = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True, timeout=120)
        if proc.returncode != 0:
            print(f"  snscrape failed for '{query}': {proc.stderr[:200]}")
            return pd.DataFrame()
    except Exception as exc:
        print(f"  snscrape error '{query}': {exc}")
        return pd.DataFrame()
    finally:
        pass

    records = []
    try:
        with open(tmp.name) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    tw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = tw.get("rawContent") or tw.get("content") or ""
                if len(text.strip()) < 5:
                    continue
                tags = ",".join(
                    h.get("text", h) if isinstance(h, dict) else str(h)
                    for h in (tw.get("hashtags") or [])
                )
                records.append(
                    {
                        "comment_text": text[:8000],
                        "platform": "Twitter",
                        "community": query[:80],
                        "community_type": "hashtag_search",
                        "subreddit": query[:80],
                        "hashtags": tags or query.replace(" ", ","),
                        "tweet_id": tw.get("id") or tw.get("id_str"),
                        "author": tw.get("user", {}).get("username", ""),
                        "score": tw.get("likeCount", 0),
                        "created_utc": tw.get("date"),
                        "dataset_source": "snscrape",
                        "dataset_split": "live_scrape",
                        "scraped_at": datetime.utcnow().isoformat(),
                    }
                )
    finally:
        os.unlink(tmp.name)

    return pd.DataFrame(records)


def scrape_all(max_per_query: int = 120, output_path: str = OUT_PATH) -> pd.DataFrame:
    os.makedirs("data/scraped", exist_ok=True)
    frames = []

    targets = [f"#{h}" for h in HASHTAGS] + QUERIES
    for q in tqdm(targets, desc="Twitter snscrape"):
        df = scrape_query(q, max_tweets=max_per_query)
        if not df.empty:
            frames.append(df)
            print(f"  {q}: {len(df)} tweets")

    if not frames:
        print("No Twitter data via snscrape (X may be blocking). Gab fallback used in dataset_builder.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if "tweet_id" in out.columns:
        out.drop_duplicates(subset=["tweet_id"], inplace=True)
    else:
        out.drop_duplicates(subset=["comment_text", "hashtags"], inplace=True)

    out.to_csv(output_path, index=False)
    print(f"\nSaved {len(out)} tweets → {output_path}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-query", type=int, default=120)
    parser.add_argument("--output", default=OUT_PATH)
    args = parser.parse_args()
    scrape_all(max_per_query=args.max_per_query, output_path=args.output)
