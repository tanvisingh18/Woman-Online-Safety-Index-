"""
Reddit comment collector via Pullpush.io (public archive API) — no PRAW credentials.

Targets women-relevant subreddits and saves:
  data/scraped/reddit_women_pullpush.csv

Run:
  python src/scraper/reddit_pullpush.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

WOMEN_SUBREDDITS = [
    "TwoXChromosomes",
    "AskWomen",
    "feminism",
    "WomenInTech",
    "GirlGamers",
    "offmychest",
    "relationship_advice",
    "AskReddit",
]

API = "https://api.pullpush.io/reddit/search/comment/"
OUT_PATH = "data/scraped/reddit_women_pullpush.csv"


def fetch_subreddit_comments(subreddit: str, size: int = 500) -> pd.DataFrame:
    """Paginate Pullpush comment search for a subreddit."""
    records = []
    before = None
    pages = max(1, size // 100)

    for _ in range(pages):
        params = {"subreddit": subreddit, "size": min(100, size - len(records))}
        if before:
            params["before"] = before
        try:
            resp = requests.get(API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception as exc:
            print(f"  r/{subreddit}: API error — {exc}")
            break

        if not data:
            break

        for c in data:
            body = (c.get("body") or "").strip()
            if body in ("[deleted]", "[removed]", "") or len(body) < 5:
                continue
            records.append(
                {
                    "comment_text": body[:8000],
                    "platform": "Reddit",
                    "community": subreddit,
                    "community_type": "subreddit",
                    "subreddit": subreddit,
                    "comment_id": c.get("id"),
                    "post_id": c.get("link_id", "").replace("t3_", ""),
                    "author": c.get("author", "[deleted]"),
                    "score": c.get("score", 0),
                    "created_utc": c.get("created_utc"),
                    "is_deleted": int(body == "[deleted]"),
                    "is_removed": int(body == "[removed]"),
                    "permalink": c.get("permalink", ""),
                    "dataset_source": "pullpush",
                    "dataset_split": "women_relevant",
                    "women_priority_community": 1,
                    "scraped_at": datetime.utcnow().isoformat(),
                }
            )

        before = data[-1].get("created_utc")
        if len(data) < 100:
            break
        time.sleep(0.5)

    return pd.DataFrame(records)


def scrape_all(
    subreddits: list[str] | None = None,
    per_sub: int = 400,
    output_path: str = OUT_PATH,
) -> pd.DataFrame:
    os.makedirs("data/scraped", exist_ok=True)
    subs = subreddits or WOMEN_SUBREDDITS
    frames = []

    for sub in tqdm(subs, desc="Pullpush subreddits"):
        df = fetch_subreddit_comments(sub, size=per_sub)
        if not df.empty:
            frames.append(df)
            print(f"  r/{sub}: {len(df)} comments")
        time.sleep(1)

    if not frames:
        print("No Pullpush data collected.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.drop_duplicates(subset=["comment_id"], inplace=True)
    out.to_csv(output_path, index=False)
    print(f"\nSaved {len(out)} Reddit comments → {output_path}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sub", type=int, default=400)
    parser.add_argument("--output", default=OUT_PATH)
    args = parser.parse_args()
    scrape_all(per_sub=args.per_sub, output_path=args.output)
