"""
Dataset quality audit — checks if scraped data is sufficient for WHSI/MRI.

Run:
  python src/dataset_audit.py --path data/scraped/youtube_master_comments.csv
  python src/dataset_audit.py --all-scraped
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from load_scraped import discover_scraped_files, load_generic_scrape, load_youtube_master


def audit_dataframe(df: pd.DataFrame, name: str = "dataset") -> dict:
    """Return audit report dict with pass/fail and recommendations."""
    n = len(df)
    report = {
        "name": name,
        "n_rows": n,
        "checks": {},
        "score": 0,
        "max_score": 0,
        "verdict": "",
        "recommendations": [],
    }

    def check(key: str, passed: bool, weight: int, detail: str):
        report["checks"][key] = {"passed": bool(passed), "detail": str(detail)}
        report["max_score"] += weight
        if passed:
            report["score"] += weight

    # Volume
    check("min_volume_500", n >= 500, 15, f"{n} rows (need ≥500 for stable platform scores)")
    check("min_volume_2000", n >= 2000, 10, f"{n} rows (≥2000 recommended for thesis-grade stats)")

    # Text
    text_ok = "comment_text" in df.columns and df["comment_text"].astype(str).str.len().gt(5).mean() > 0.95
    check("valid_text", text_ok, 15, "comment_text present and >5 chars for 95%+ rows")

    # Labels
    has_harm = "is_harmful" in df.columns and df["is_harmful"].notna().mean() > 0.8
    harm_rate = float(df["is_harmful"].mean()) if has_harm else None
    check("harm_labels", has_harm, 20, f"is_harmful labelled: {has_harm}" + (f", rate={harm_rate:.1%}" if harm_rate is not None else ""))

    if harm_rate is not None and (harm_rate < 0.02 or harm_rate > 0.85):
        report["recommendations"].append(
            f"Harm rate {harm_rate:.1%} is extreme — verify labels/classifier; ideal range ~5–40% for discourse data."
        )

    # Platform
    has_plat = "platform" in df.columns and df["platform"].nunique() >= 1
    check("platform_column", has_plat, 10, f"platforms: {df['platform'].value_counts().to_dict() if has_plat else 'missing'}")

    # Hotspot community
    comm_col = next((c for c in ["subreddit", "channel_title", "community"] if c in df.columns), None)
    if comm_col:
        comm_counts = df[comm_col].value_counts()
        n_comm_20 = int((comm_counts >= 20).sum())
        check(
            "hotspot_communities",
            n_comm_20 >= 3,
            15,
            f"{n_comm_20} communities with ≥20 comments (column={comm_col})",
        )
        if n_comm_20 < 3:
            report["recommendations"].append(
                "Scrape more subreddits/channels — need ≥3 communities with 20+ comments each for hotspot leaderboard."
            )
    else:
        check("hotspot_communities", False, 15, "No subreddit/channel column — hotspot detection limited")

    # Engagement (Normalization)
    eng_col = next((c for c in ["score", "like_count", "likes"] if c in df.columns), None)
    check("engagement", eng_col is not None, 10, f"engagement column: {eng_col or 'MISSING'}")

    # Time series
    time_col = next((c for c in ["created_utc", "published_at"] if c in df.columns), None)
    check("timestamps", time_col is not None, 10, f"time column: {time_col or 'MISSING — no spike detection'}")

    # Authors (Frequency HHI)
    check("authors", "author" in df.columns or "author_display_name" in df.columns, 5, "author column for repeat-offender frequency")

    pct = 100 * report["score"] / max(report["max_score"], 1)
    if pct >= 80:
        report["verdict"] = "GOOD — suitable for full WHSI + hotspot + matrix pipeline"
    elif pct >= 55:
        report["verdict"] = "ACCEPTABLE — pipeline runs; add labels/engagement/communities for accuracy"
    else:
        report["verdict"] = "WEAK — augment with more fields or run classifier + more scrape volume"

    return report


def print_report(report: dict):
    print(f"\n{'='*60}")
    print(f"AUDIT: {report['name']}")
    print(f"Rows: {report['n_rows']}  |  Score: {report['score']}/{report['max_score']}")
    print(f"VERDICT: {report['verdict']}")
    print(f"{'='*60}")
    for k, v in report["checks"].items():
        mark = "✓" if v.get("passed", v.get("pass")) else "✗"
        print(f"  {mark} {k}: {v['detail']}")
    if report["recommendations"]:
        print("\nRecommendations:")
        for r in report["recommendations"]:
            print(f"  → {r}")


def load_for_audit(path: str) -> pd.DataFrame:
    if "youtube" in path.lower():
        return load_youtube_master(path)
    return load_generic_scrape(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None)
    parser.add_argument("--all-scraped", action="store_true")
    parser.add_argument("--json-out", default="outputs/results/dataset_audit.json")
    args = parser.parse_args()

    paths = []
    if args.all_scraped:
        paths = discover_scraped_files()
        yp = "../dataset/data/youtube_master_comments.csv"
        if os.path.exists(yp):
            paths.append(yp)
    elif args.path:
        paths = [args.path]
    else:
        parser.print_help()
        return

    reports = []
    for p in paths:
        try:
            df = load_for_audit(p)
            r = audit_dataframe(df, name=os.path.basename(p))
            print_report(r)
            reports.append(r)
        except Exception as e:
            print(f"[Error] {p}: {e}")

    if reports and args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"\nSaved audit JSON → {args.json_out}")


if __name__ == "__main__":
    main()
