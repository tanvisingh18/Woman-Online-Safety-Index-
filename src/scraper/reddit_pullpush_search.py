"""
Reddit LIVE archive search via Pullpush — real comments matching harassment queries.
No EDOS, no proxies. Appends to data/scraped/reddit_live_search.csv

Run: python src/scraper/reddit_pullpush_search.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

API = "https://api.pullpush.io/reddit/search/comment/"
OUT_PATH = "data/scraped/reddit_live_search.csv"

# Women-safety search queries on real Reddit
SEARCH_QUERIES = [
    "sexist",
    "harassed me",
    "misogyny",
    "rape",
    "stupid bitch",
    "women are",
    "feminazi",
    "sexual harassment",
    "catcalled",
    "assaulted",
    "whore",
    "slut",
    "send nudes",
    "kill yourself woman",
    "dumb bitch",
]


def fetch_query(query: str, size: int = 200) -> pd.DataFrame:
    records = []
    before = None
    pages = max(1, size // 100)

    for _ in range(pages):
        params = {"q": query, "size": min(100, size - len(records))}
        if before:
            params["before"] = before
        try:
            resp = requests.get(API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception as exc:
            print(f"  '{query}': {exc}")
            break
        if not data:
            break

        for c in data:
            body = (c.get("body") or "").strip()
            if body in ("[deleted]", "[removed]", "") or len(body) < 8:
                continue
            sub = c.get("subreddit", "unknown")
            records.append(
                {
                    "comment_text": body[:8000],
                    "platform": "Reddit",
                    "community": sub,
                    "community_type": "subreddit",
                    "subreddit": sub,
                    "comment_id": c.get("id"),
                    "author": c.get("author", "[deleted]"),
                    "score": c.get("score", 0),
                    "created_utc": c.get("created_utc"),
                    "dataset_source": "pullpush_live_search",
                    "dataset_split": "live_scrape",
                    "search_query": query,
                    "scraped_at": datetime.utcnow().isoformat(),
                }
            )
        before = data[-1].get("created_utc")
        if len(data) < 100:
            break
        time.sleep(0.4)

    return pd.DataFrame(records)


def scrape_all(per_query: int = 150, output_path: str = OUT_PATH) -> pd.DataFrame:
    os.makedirs("data/scraped", exist_ok=True)
    frames = []
    for q in tqdm(SEARCH_QUERIES, desc="Reddit live search"):
        df = fetch_query(q, size=per_query)
        if not df.empty:
            frames.append(df)
            print(f"  '{q}': {len(df)}")
        time.sleep(1)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["comment_id"])
    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        out = pd.concat([prev, out], ignore_index=True).drop_duplicates(subset=["comment_id"])

    out.to_csv(output_path, index=False)
    print(f"\nSaved {len(out)} live Reddit search comments → {output_path}")
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--per-query", type=int, default=150)
    args = p.parse_args()
    scrape_all(per_query=args.per_query)
