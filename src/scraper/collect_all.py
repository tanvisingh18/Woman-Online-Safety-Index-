#!/usr/bin/env python3
"""
One-command data collection — no user credentials required for most sources.

Runs (in order):
  1. YouTube reply threads (yt-dlp)
  2. Reddit women subreddits (Pullpush API)
  3. Twitter hashtags (snscrape) — Gab fallback in dataset_builder if blocked
  4. Rebuild master + validation report

Usage:
  python src/scraper/collect_all.py
  python src/scraper/collect_all.py --skip-youtube   # faster
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
    parser.add_argument("--skip-youtube", action="store_true")
    parser.add_argument("--skip-reddit", action="store_true")
    parser.add_argument("--skip-twitter", action="store_true")
    parser.add_argument("--max-youtube-videos", type=int, default=None)
    parser.add_argument("--rebuild-only", action="store_true")
    args = parser.parse_args()

    if not args.rebuild_only:
        if not args.skip_youtube:
            from youtube_replies_ytdlp import scrape_all

            scrape_all(max_videos=args.max_youtube_videos)

        if not args.skip_reddit:
            from reddit_pullpush import scrape_all as scrape_reddit

            scrape_reddit(per_sub=350)

        if not args.skip_twitter:
            from twitter_snscrape import scrape_all as scrape_twitter

            scrape_twitter(max_per_query=100)

    from dataset_builder import build_women_relevant
    from validate_classifier import run_validation
    from mri_engine import run_empirical_mri_and_blend

    build_women_relevant(label=True)
    run_validation()
    run_empirical_mri_and_blend()

    print("\n✓ collect_all complete. Run: python run_pipeline.py --skip-influence --no-label")


if __name__ == "__main__":
    main()
