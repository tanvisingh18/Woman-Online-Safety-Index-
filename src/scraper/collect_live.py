#!/usr/bin/env python3
"""
Collect REAL platform scrapes only — no EDOS, no Gab, no filler datasets.

  1. Reddit subreddits + keyword search (Pullpush — live Reddit archive)
  2. Twitter hashtags (snscrape, if available)
  3. Rebuild master_live_scrapes.csv from your YouTube/Telegram/Reddit files

Usage:
  python src/scraper/collect_live.py
  python src/scraper/collect_live.py --skip-twitter
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "scraper"))
os.chdir(ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-reddit", action="store_true")
    parser.add_argument("--skip-twitter", action="store_true")
    parser.add_argument("--per-sub", type=int, default=400)
    parser.add_argument("--per-query", type=int, default=150)
    args = parser.parse_args()

    if not args.skip_reddit:
        from reddit_pullpush import scrape_all as scrape_subs
        from reddit_pullpush_search import scrape_all as scrape_search

        print("\n=== REDDIT LIVE: subreddit scrape ===")
        scrape_subs(per_sub=args.per_sub)
        print("\n=== REDDIT LIVE: harassment keyword search ===")
        scrape_search(per_query=args.per_query)

    if not args.skip_twitter:
        from twitter_snscrape import scrape_all as scrape_twitter

        print("\n=== TWITTER LIVE: snscrape ===")
        scrape_twitter(max_per_query=100)

    from dataset_builder import build_women_relevant

    print("\n=== BUILD: mixed master (live scrapes + EDOS) ===")
    build_women_relevant(label=True, live_only=False)
    print("\nDone. Run: python run_pipeline.py --skip-influence --no-label")


if __name__ == "__main__":
    main()
