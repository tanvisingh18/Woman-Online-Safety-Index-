"""
Scrape PUBLIC Telegram channels via t.me/s/ web preview — no API key required.

Limitations (document in thesis):
  - Public channels only (private/invite groups need Telethon + my.telegram.org API)
  - Web preview may omit some messages vs native client
  - Do NOT scrape CSAM/abuse channels without ethics approval

Usage:
  PYTHONPATH=src python src/scraper/telegram_public_scraper.py
  PYTHONPATH=src python src/scraper/telegram_public_scraper.py --channels data/scraped/telegram_public_channels.txt

Output: data/scraped/telegram_public_scrape.csv (append or merge into telegram_messages.csv)
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

DEFAULT_CHANNELS_FILE = "data/scraped/telegram_public_channels.txt"
OUT_PATH = "data/scraped/telegram_public_scrape.csv"
USER_AGENT = "WomenSafetyIndexResearch/1.0 (academic; public channel preview only)"


def load_channel_list(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    channels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ch = line.lstrip("@").split("/")[-1]
            channels.append(ch)
    return channels


def fetch_public_channel(username: str, max_pages: int = 3, pause_sec: float = 1.5) -> list[dict]:
    """Fetch messages from https://t.me/s/{username} paginated preview."""
    base = f"https://t.me/s/{username}"
    headers = {"User-Agent": USER_AGENT}
    rows: list[dict] = []
    url = base

    for _ in range(max_pages):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [{username}] fetch failed: {exc}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        widgets = soup.select(".tgme_widget_message")
        if not widgets:
            break

        for w in widgets:
            text_el = w.select_one(".tgme_widget_message_text")
            text = text_el.get_text("\n", strip=True) if text_el else ""
            if len(text) < 5:
                continue
            mid = w.get("data-post", "")
            date_el = w.select_one("time")
            date_str = date_el.get("datetime", "") if date_el else ""
            views_el = w.select_one(".tgme_widget_message_views")
            views = views_el.get_text(strip=True) if views_el else ""
            rows.append(
                {
                    "channel": username,
                    "message_id": mid.split("/")[-1] if mid else "",
                    "date": date_str,
                    "text": text,
                    "views": views,
                    "forwards": "",
                    "replies": "",
                    "length": len(text),
                    "source": "t.me/s/public_preview",
                }
            )

        # Older messages: find "load more" link with before= param
        older = soup.select_one('a[href*="?before="]')
        if not older or not older.get("href"):
            break
        url = urljoin("https://t.me", older["href"])
        time.sleep(pause_sec)

    return rows


def scrape_channels(channels: list[str], max_pages: int = 3) -> pd.DataFrame:
    all_rows: list[dict] = []
    for ch in channels:
        print(f"Scraping public channel @{ch} ...")
        rows = fetch_public_channel(ch, max_pages=max_pages)
        print(f"  → {len(rows)} messages")
        all_rows.extend(rows)
        time.sleep(1.0)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["channel", "message_id"], keep="first")
    return df


def merge_into_master_export(df: pd.DataFrame, master_path: str = "data/scraped/telegram_messages.csv") -> None:
    if df.empty:
        return
    if os.path.exists(master_path):
        old = pd.read_csv(master_path, low_memory=False)
        combined = pd.concat([old, df], ignore_index=True)
        key = ["channel", "message_id"] if "message_id" in combined.columns else ["channel", "text"]
        combined = combined.drop_duplicates(subset=key, keep="first")
    else:
        combined = df
    combined.to_csv(master_path, index=False)
    print(f"Merged {len(df)} new rows → {master_path} (total {len(combined)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", default=DEFAULT_CHANNELS_FILE)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--merge", action="store_true", help="Append into telegram_messages.csv")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    channels = load_channel_list(args.channels)
    if not channels:
        print(f"No channels in {args.channels}. Add one username per line (see data/scraped/TELEGRAM_SAMPLING.md).")
        return

    df = scrape_channels(channels, max_pages=args.max_pages)
    if df.empty:
        print("No messages scraped.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows → {args.out}")

    if args.merge:
        merge_into_master_export(df)


if __name__ == "__main__":
    main()
