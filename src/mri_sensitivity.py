"""
MRI sensitivity analysis — weight perturbation + response-time horizon.

Uses PDR/RAS-adjusted inputs and actual blended MRI scores from mri_scores.csv
(Faculty Step 9 — point estimate must lie inside its band).

Outputs: outputs/results/mri_sensitivity.csv

Run: PYTHONPATH=src python src/mri_sensitivity.py
"""

from __future__ import annotations

import os

import pandas as pd

from mri_engine import compute_mri, mri_label, speed_score
from structural_platform_evidence import adjust_consistency_score, adjust_removal_rate


def _mri_row_inputs(row: pd.Series) -> tuple[float, float, float]:
    plat = row["platform"]
    removal_adj, _ = adjust_removal_rate(float(row["removal_rate"]), plat)
    consistency_adj, _ = adjust_consistency_score(float(row["consistency_score"]), plat)
    return removal_adj, float(row["avg_response_hours"]), consistency_adj


def run_sensitivity(
    report_path: str = "data/transparency/moderation_reports.csv",
    mri_scores_path: str = "data/processed/mri_scores.csv",
) -> pd.DataFrame:
    df = pd.read_csv(report_path)
    actual_mri = pd.read_csv(mri_scores_path).set_index("platform")["MRI_score"].to_dict() if os.path.exists(mri_scores_path) else {}

    rows = []
    weight_sets = [
        (0.45, 0.30, 0.25),
        (0.50, 0.25, 0.25),
        (0.40, 0.35, 0.25),
        (0.45, 0.30, 0.35),
        (0.33, 0.33, 0.34),
    ]

    for w_r, w_s, w_c in weight_sets:
        for _, row in df.iterrows():
            rem_adj, hours, con_adj = _mri_row_inputs(row)
            R = rem_adj * 100
            S = speed_score(hours)
            C = con_adj * 100
            score = round(w_r * R + w_s * S + w_c * C, 2)
            plat = row["platform"]
            rows.append({
                "platform": plat,
                "analysis": "weight_sensitivity_pdr_ras",
                "w_removal": w_r,
                "w_speed": w_s,
                "w_consistency": w_c,
                "MRI_score": score,
                "MRI_actual_file": actual_mri.get(plat),
                "MRI_label": mri_label(score),
            })

    for max_h in [168, 336, 720, 1440]:
        for _, row in df.iterrows():
            rem_adj, hours, con_adj = _mri_row_inputs(row)
            R = rem_adj * 100
            S = speed_score(hours, max_hours=float(max_h))
            C = con_adj * 100
            alt = round(0.45 * R + 0.30 * S + 0.25 * C, 2)
            rows.append({
                "platform": row["platform"],
                "analysis": "horizon_sensitivity_pdr_ras",
                "max_hours": max_h,
                "MRI_score": alt,
                "speed_score": round(S, 2),
            })

    out = pd.DataFrame(rows)
    os.makedirs("outputs/results", exist_ok=True)
    out.to_csv("outputs/results/mri_sensitivity.csv", index=False)

    # Audit: actual MRI inside band?
    audit = []
    for plat in df["platform"].unique():
        plat_rows = out[out["platform"] == plat]
        lo, hi = plat_rows["MRI_score"].min(), plat_rows["MRI_score"].max()
        actual = actual_mri.get(plat)
        audit.append({
            "platform": plat,
            "MRI_actual": actual,
            "sensitivity_band": f"[{lo:.1f}–{hi:.1f}]",
            "point_inside_band": actual is None or (lo <= actual <= hi),
        })
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv("outputs/results/mri_band_audit.csv", index=False)

    print("\nMRI BAND AUDIT (PDR/RAS-adjusted sensitivity):")
    print(audit_df.to_string(index=False))
    print(f"\nSaved → outputs/results/mri_sensitivity.csv")
    return out


if __name__ == "__main__":
    run_sensitivity()
