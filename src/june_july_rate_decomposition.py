"""
June→July 2026 harm-rate decomposition (Faculty Step 3).

Separates threshold change vs corpus/denominator effects by running BOTH
legacy (0.52/0.45/0.38) and production (0.40/0.34/0.28) rules on the CURRENT
live master corpus.

Also defines the denominator for all rate claims:
  live_scrape ∩ women-relevant master panel (master_women_relevant.csv)

Run: PYTHONPATH=src python src/june_july_rate_decomposition.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from corpus_config import platform_analysis_mask
from gendered_harm_model import (
    PERP_SEXIST_THRESH,
    PERP_SEXIST_WITH_THREAT,
    PERP_THREAT_MIN,
    PERPETRATOR_ATTACK,
    VICTIM_DISCLOSURE,
)


JUNE_RATES = {"YouTube": 19.3, "Reddit": 17.4, "Telegram": 3.4}  # progress report June 2026


def _perp_mask(texts, sexist, threat, t_sex, t_sex_th, t_th) -> np.ndarray:
    out = []
    for text, sp, tp in zip(texts, sexist, threat):
        text = str(text) if isinstance(text, str) else ""
        victim = bool(VICTIM_DISCLOSURE.search(text))
        attack_kw = bool(PERPETRATOR_ATTACK.search(text))
        is_perp = (
            attack_kw
            or (sp >= t_sex and not victim)
            or (sp >= t_sex_th and tp >= t_th and not victim)
        )
        out.append(int(is_perp))
    return np.array(out)


def run(master_path: str = "data/processed/master_women_relevant.csv") -> dict:
    df = pd.read_csv(master_path, low_memory=False)
    live = df[platform_analysis_mask(df)].copy()
    live = live[live["dataset_split"] == "live_scrape"].copy()

    texts = live["comment_text"].fillna("").tolist()
    sexist = pd.to_numeric(live["gendered_harm_proba"], errors="coerce").fillna(0).values
    # threat_score in master is post-processed; use as threat_proba proxy
    threat = pd.to_numeric(live["threat_score"], errors="coerce").fillna(0).values

    legacy = _perp_mask(texts, sexist, threat, 0.52, 0.45, 0.38)
    prod = _perp_mask(
        texts, sexist, threat, PERP_SEXIST_THRESH, PERP_SEXIST_WITH_THREAT, PERP_THREAT_MIN
    )
    live["_legacy"] = legacy
    live["_prod"] = prod

    rows = []
    for plat, g in live.groupby("platform"):
        leg = float(g["_legacy"].mean() * 100)
        pr = float(g["_prod"].mean() * 100)
        june = JUNE_RATES.get(plat)
        rows.append(
            {
                "platform": plat,
                "n_live_women_relevant": len(g),
                "june_2026_reported_pct": june,
                "july_legacy_thresh_on_current_corpus_pct": round(leg, 2),
                "july_production_thresh_on_current_corpus_pct": round(pr, 2),
                "delta_threshold_only_pp": round(pr - leg, 2),
                "delta_june_to_july_total_pp": round(pr - june, 2) if june is not None else None,
                "approx_residual_vs_june_pp": (
                    round(leg - june, 2) if june is not None else None
                ),
                "residual_note": (
                    "Residual vs June ≈ corpus/pipeline/relabel differences not explained by "
                    "threshold alone (master_women_relevant live panel + any prior relabel)."
                    if june is not None
                    else ""
                ),
            }
        )

    denominator = {
        "definition": (
            "All perpetrator rates in the July 2026 results are conditional on the "
            "women-relevant live-scrape analysis panel: rows in "
            "data/processed/master_women_relevant.csv with dataset_split=live_scrape "
            "that pass platform_analysis_mask (live platforms only). "
            "They are NOT unconditional rates over all YouTube/Reddit/Telegram comments."
        ),
        "correct_claim": (
            "X% of comments in our women-relevant live scrape are classifier-flagged "
            "perpetrator attacks."
        ),
        "incorrect_claim": (
            "X% of all platform comments attack women."
        ),
        "june_july_not_comparable": (
            "June and July rates are not directly comparable: (1) production thresholds "
            "were lowered (0.52→0.40), and (2) the analysis denominator is the "
            "women-relevant live panel. Decompose using this table before interpreting "
            "the doubling."
        ),
    }

    out = {
        "denominator": denominator,
        "thresholds": {
            "legacy_june": {"sexist": 0.52, "sexist_with_threat": 0.45, "threat_min": 0.38},
            "production_july": {
                "sexist": PERP_SEXIST_THRESH,
                "sexist_with_threat": PERP_SEXIST_WITH_THREAT,
                "threat_min": PERP_THREAT_MIN,
            },
        },
        "rows": rows,
        "thesis_subsection_blurb": (
            "Changes from June 2026. Classifier-flagged perpetrator rates rose "
            f"(YouTube {JUNE_RATES['YouTube']}→{next(r['july_production_thresh_on_current_corpus_pct'] for r in rows if r['platform']=='YouTube')}%, "
            f"Reddit {JUNE_RATES['Reddit']}→{next(r['july_production_thresh_on_current_corpus_pct'] for r in rows if r['platform']=='Reddit')}%, "
            f"Telegram {JUNE_RATES['Telegram']}→{next(r['july_production_thresh_on_current_corpus_pct'] for r in rows if r['platform']=='Telegram')}%). "
            "Re-running legacy thresholds on the current women-relevant live corpus isolates "
            "the threshold contribution; residual differences reflect panel/pipeline updates. "
            "All rates are shares of women-relevant live-scrape comments, not of all platform comments. "
            "June and July rates are therefore not interchangeable."
        ),
    }

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/june_july_rate_decomposition.json", "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(rows).to_csv("outputs/results/june_july_rate_decomposition.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n" + denominator["definition"])
    print("Saved → outputs/results/june_july_rate_decomposition.json")
    return out


if __name__ == "__main__":
    run()
