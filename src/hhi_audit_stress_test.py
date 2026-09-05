"""
Stress-test repeat-offender HHI using hotspot hand-audit labels.

Joins audit rows back to the live scrape (author IDs), then computes HHI on:
  (a) all audited classifier-flagged rows (n=120)
  (b) audited-genuine perpetrator rows only (final_perpetrator=1)

Also reports platform-level classifier HHI (existing) vs this audit check.

Run: PYTHONPATH=src python src/hhi_audit_stress_test.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

AUDIT = "data/labelled/hotspot_audit/hotspot_audit_labelled.csv"
MASTER = "data/processed/master_women_relevant.csv"
OUT_JSON = "outputs/results/hhi_audit_stress_test.json"
OUT_MD = "outputs/results/hhi_audit_stress_test.md"
OUT_CSV = "outputs/results/hhi_audit_stress_test_rows.csv"


def _hhi(counts: pd.Series) -> float:
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    shares = (counts / total).to_numpy()
    return float(np.sum(np.square(shares)))


def _band(hhi: float) -> str:
    if not np.isfinite(hhi):
        return "undefined"
    if hhi < 0.01:
        return "highly_diffuse"
    if hhi <= 0.15:
        return "moderately_concentrated"
    return "highly_concentrated"


def main() -> None:
    audit = pd.read_csv(AUDIT)
    master = pd.read_csv(MASTER, low_memory=False)
    live = master[
        (master["dataset_split"] == "live_scrape") & (master["platform"] == "YouTube")
    ].copy()

    # Join on exact comment_text within YouTube live (audit channels are YouTube)
    live["comment_text"] = live["comment_text"].astype(str)
    audit["comment_text"] = audit["comment_text"].astype(str)
    merged = audit.merge(
        live[["comment_text", "author", "community", "harm_role"]].drop_duplicates(
            subset=["comment_text"]
        ),
        on="comment_text",
        how="left",
        suffixes=("", "_live"),
    )
    merged["final_perpetrator"] = pd.to_numeric(
        merged["final_perpetrator"], errors="coerce"
    ).fillna(0).astype(int)

    matched = int(merged["author"].notna().sum())
    merged.to_csv(OUT_CSV, index=False)

    def subset_stats(df: pd.DataFrame, label: str) -> dict:
        sub = df[
            df["author"].notna()
            & (~df["author"].astype(str).str.strip().isin(["", "nan", "None", "[deleted]"]))
        ].copy()
        n = len(sub)
        if n == 0:
            return {
                "label": label,
                "n_comments": 0,
                "n_unique_authors": 0,
                "HHI": None,
                "band": "undefined",
                "top_10pct_user_share": None,
                "note": "no rows with author",
            }
        counts = sub.groupby("author").size().sort_values(ascending=False)
        n_users = len(counts)
        k = max(1, int(np.ceil(0.10 * n_users)))
        top_share = float(counts.iloc[:k].sum() / n)
        hhi = _hhi(counts)
        return {
            "label": label,
            "n_comments": int(n),
            "n_unique_authors": int(n_users),
            "HHI": round(hhi, 6),
            "band": _band(hhi),
            "top_10pct_user_share": round(top_share, 4),
            "max_by_single_user": int(counts.iloc[0]),
            "mean_per_user": round(float(counts.mean()), 3),
        }

    flagged_all = subset_stats(merged, "audit_classifier_flagged_all_120")
    genuine = subset_stats(
        merged[merged["final_perpetrator"] == 1], "audit_genuine_perpetrator_only"
    )

    # Full YouTube classifier HHI (baseline from existing method)
    yt = live[
        (live["harm_role"] == "perpetrator_attack")
        & live["author"].notna()
        & (~live["author"].astype(str).str.strip().isin(["", "nan", "None", "[deleted]"]))
    ]
    yt_counts = yt.groupby("author").size()
    yt_hhi = _hhi(yt_counts)
    baseline = {
        "label": "youtube_live_classifier_perpetrator_all",
        "n_comments": int(len(yt)),
        "n_unique_authors": int(len(yt_counts)),
        "HHI": round(yt_hhi, 6),
        "band": _band(yt_hhi),
    }

    # Interpretation
    n_gen = genuine["n_comments"]
    if n_gen < 10:
        interpretation = (
            f"Audited-genuine n={n_gen} is too small for a stable HHI estimate. "
            "Cannot confirm or refute corpus-level diffusion on clean labels alone. "
            "Classifier-flagged audit HHI and corpus classifier HHI both look diffuse, "
            "but because ~57% of held-out FPs are structural (topic/lexical), "
            "diffusion on flags remains a provisional finding pending Path A gold-scale "
            "recompute — do not treat 'noise cancels' as proven."
        )
        status = "inconclusive_small_n_genuine"
    else:
        interpretation = (
            "Compare genuine vs flagged HHI in this audit panel; if both diffuse, "
            "supports robustness; if genuine concentrates, flag-HHI was artifactual."
        )
        status = "compared"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matched_authors_of_120": matched,
        "status": status,
        "baseline_youtube_classifier": baseline,
        "audit_flagged": flagged_all,
        "audit_genuine": genuine,
        "interpretation": interpretation,
        "policy_caveat": (
            "Policy claim 'harm is diffuse across users' was computed on classifier "
            "flags. Hotspot audit shows most flags in high-flag channels are not "
            "genuine attacks. Until HHI is recomputed on a larger gold-genuine set, "
            "state diffusion as provisional / classifier-flag distributional, not as "
            "proven true-perpetrator diffusion."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    md = [
        "# HHI / diffusion stress-test on hotspot audit labels",
        "",
        f"Matched authors for audit rows: **{matched}/120**.",
        "",
        "| Set | n comments | unique authors | HHI | Band |",
        "|-----|------------|----------------|-----|------|",
        f"| YouTube live classifier (baseline) | {baseline['n_comments']} | {baseline['n_unique_authors']} | {baseline['HHI']} | {baseline['band']} |",
        f"| Audit flagged (all 120) | {flagged_all['n_comments']} | {flagged_all['n_unique_authors']} | {flagged_all.get('HHI')} | {flagged_all.get('band')} |",
        f"| Audit genuine only | {genuine['n_comments']} | {genuine['n_unique_authors']} | {genuine.get('HHI')} | {genuine.get('band')} |",
        "",
        f"**Status:** `{status}`",
        "",
        interpretation,
        "",
        payload["policy_caveat"],
        "",
        f"Rows: `{OUT_CSV}`",
        "",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))

    print(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
