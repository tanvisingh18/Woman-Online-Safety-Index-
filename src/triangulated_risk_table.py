"""
Triangulated Platform Risk Table — WHSI observed, WHSI_lit (Assumption A1), MRI, structural risk.

WHSI_corrected REMOVED per faculty Step 5 — use WHSI_literature_adjusted only.

Output: outputs/results/triangulated_platform_risk.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from structural_platform_evidence import (
    AI_FORENSICS_TELEGRAM_2026,
    literature_alignment_note,
    visibility_inversion_score,
    MODERATION_BYPASS_RATE,
    structural_risk_label,
)


def build_triangulated_table(
    whsi_path: str = "data/processed/whsi_scores.csv",
    mri_path: str = "data/processed/mri_scores.csv",
) -> pd.DataFrame:
    whsi = pd.read_csv(whsi_path)
    mri = pd.read_csv(mri_path)[["platform", "MRI_score", "MRI_label"]]

    # Live platforms only — no Twitter in comparison table (Faculty Step 4)
    live = whsi[(whsi.get("is_historical", 0) != 1) & (whsi.get("is_proxy", 0) != 1)].copy()
    merged = live.merge(mri, on="platform", how="left")

    rows = []
    for _, r in merged.iterrows():
        plat = r["platform"]
        w_raw = float(r["WHSI_raw"])
        w_lit = float(r.get("WHSI_literature_adjusted", np.nan))
        w_lit_lo = float(r.get("WHSI_lit_sensitivity_low", w_lit))
        w_lit_hi = float(r.get("WHSI_lit_sensitivity_high", w_lit))
        rows.append(
            {
                "platform": plat,
                "platform_display": r.get("platform_display", plat),
                "WHSI_raw_observed": w_raw,
                "WHSI_raw_display": r.get("WHSI_score_display", f"{w_raw:.1f}"),
                "WHSI_literature_adjusted": w_lit,
                "WHSI_lit_sensitivity_band": f"[{w_lit_lo:.1f}–{w_lit_hi:.1f}]",
                "WHSI_literature_display": r.get("WHSI_literature_display", f"{w_lit:.1f}" if pd.notna(w_lit) else ""),
                "ecosystem_harm_exposure_rate": r.get("ecosystem_harm_exposure_rate"),
                "assumption_A1": "Harm in invisible channels >= visible channel density (see docs/ASSUMPTION_A1_VISIBILITY.md)",
                "MRI_score": r["MRI_score"],
                "MRI_label": r["MRI_label"],
                "VIS": visibility_inversion_score(plat),
                "MBR_pct": round(MODERATION_BYPASS_RATE.get(plat, 0.5) * 100, 1),
                "structural_risk_signal": structural_risk_label(plat),
                "signal_independence_note": (
                    "VIS/MBR/EHER for Telegram derive largely from one visibility estimate (AI Forensics 2026); "
                    "NOT four independent converging signals — literature-consistent, not literature-confirmed"
                ),
                "literature_alignment": literature_alignment_note(plat, w_raw, float(r["MRI_score"])),
                "row_type": "live_scrape",
            }
        )

    out = pd.DataFrame(rows)
    os.makedirs("outputs/results", exist_ok=True)
    out.to_csv("outputs/results/triangulated_platform_risk.csv", index=False)
    return out


def print_triangulated_summary(df: pd.DataFrame):
    print(f"\n{'='*90}")
    print("TRIANGULATED PLATFORM RISK (live platforms — 3-way comparison)")
    print(f"{'='*90}")
    cols = [
        "platform_display",
        "WHSI_raw_observed",
        "WHSI_literature_adjusted",
        "WHSI_lit_sensitivity_band",
        "MRI_score",
        "structural_risk_signal",
    ]
    print(df[cols].to_string(index=False))
    print(f"{'='*90}\n")


if __name__ == "__main__":
    df = build_triangulated_table()
    print_triangulated_summary(df)
