"""
Build a single reconciliation table: classifier-flagged rate vs Rogan-Gladen
prevalence-corrected rate (Faculty / submission clarity).

Output:
  outputs/results/rate_reconciliation.csv
  outputs/results/rate_reconciliation.json

Run: PYTHONPATH=src python src/export_rate_reconciliation.py
"""

from __future__ import annotations

import json
import os

import pandas as pd


EXPLANATORY_NOTE = (
    "Two different quantities appear in results tables and must not be conflated. "
    "(1) classifier_flagged_rate = share of live comments the model labels "
    "perpetrator_attack (measurement on the scrape). "
    "(2) prevalence_corrected_rate = Rogan-Gladen estimate of true perpetrator "
    "prevalence using corpus human sensitivity/specificity. "
    "Corrected rates are typically lower when specificity < 1 (false positives "
    "inflate the classifier rate). For Telegram, observed rate is below estimated "
    "FPR, so Rogan-Gladen is censored at 0 — report classifier_flagged_rate as "
    "primary for that platform."
)


def build_reconciliation(
    prevalence_path: str = "outputs/results/prevalence_corrected_rates.csv",
    whsi_path: str = "data/processed/whsi_scores.csv",
) -> pd.DataFrame:
    prev = pd.read_csv(prevalence_path)
    rows = []
    for _, r in prev.iterrows():
        plat = r["platform"]
        note = str(r.get("reporting_note", "") or "")
        status = str(r.get("validation_status", ""))
        rows.append(
            {
                "platform": plat,
                "n_comments": int(r["n_comments"]),
                "classifier_flagged_rate_pct": r["classifier_flagged_rate_pct"],
                "prevalence_corrected_rate_pct": r["prevalence_corrected_pct"],
                "prevalence_combined_ci": r["combined_ci_display"],
                "sensitivity_used": r["sensitivity_used"],
                "specificity_used": r["specificity_used"],
                "validation_status": status,
                "why_they_differ": (
                    "RG censored (obs < FPR); use classifier rate as primary"
                    if "censored" in status
                    else (
                        "Corrected < classifier because Sp<1 implies false positives "
                        "inflate the flagged rate; Se/Sp from corpus gold labels"
                    )
                ),
                "reporting_note": note,
            }
        )

    out = pd.DataFrame(rows)

    # Attach WHSI_raw for side-by-side reading if available
    if os.path.exists(whsi_path):
        whsi = pd.read_csv(whsi_path)
        if "platform" in whsi.columns and "WHSI_raw" in whsi.columns:
            keep = ["platform", "WHSI_raw"]
            if "WHSI_literature_adjusted" in whsi.columns:
                keep.append("WHSI_literature_adjusted")
            if "WHSI_lit_sensitivity_low" in whsi.columns:
                keep += ["WHSI_lit_sensitivity_low", "WHSI_lit_sensitivity_high"]
            out = out.merge(whsi[keep], on="platform", how="left")

    return out


def main() -> None:
    os.makedirs("outputs/results", exist_ok=True)
    df = build_reconciliation()
    csv_path = "outputs/results/rate_reconciliation.csv"
    json_path = "outputs/results/rate_reconciliation.json"
    df.to_csv(csv_path, index=False)
    payload = {
        "purpose": "Reconcile classifier-flagged rates with prevalence-corrected rates",
        "explanatory_note": EXPLANATORY_NOTE,
        "thesis_paragraph": (
            "Throughout this thesis, two rates are reported for each live platform. "
            "The classifier-flagged perpetrator rate is the share of scraped comments "
            f"labelled perpetrator_attack by the corpus-tuned model "
            f"(YouTube {df.loc[df.platform=='YouTube','classifier_flagged_rate_pct'].iloc[0]}%, "
            f"Reddit {df.loc[df.platform=='Reddit','classifier_flagged_rate_pct'].iloc[0]}%, "
            f"Telegram {df.loc[df.platform=='Telegram','classifier_flagged_rate_pct'].iloc[0]}%). "
            "The prevalence-corrected rate applies the Rogan-Gladen estimator with "
            "sensitivity and specificity from dual human annotation on 360 stratified "
            "comments, yielding true-rate estimates of "
            f"YouTube {df.loc[df.platform=='YouTube','prevalence_corrected_rate_pct'].iloc[0]}% "
            f"{df.loc[df.platform=='YouTube','prevalence_combined_ci'].iloc[0]}, "
            f"Reddit {df.loc[df.platform=='Reddit','prevalence_corrected_rate_pct'].iloc[0]}% "
            f"{df.loc[df.platform=='Reddit','prevalence_combined_ci'].iloc[0]}, "
            "and a censored (non-informative) correction for Telegram where the "
            "observed rate lies below the estimated false-positive rate—so Telegram "
            "results lead with the classifier-flagged rate. These quantities answer "
            "different questions and are never substituted for one another in rankings "
            "without an explicit label."
        ),
        "rows": df.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(EXPLANATORY_NOTE)
    print()
    print(df.to_string(index=False))
    print(f"\nSaved → {csv_path}")
    print(f"Saved → {json_path}")


if __name__ == "__main__":
    main()
