"""
Telegram literature-adjusted WHSI — threshold sensitivity (Faculty Step 4).

Recomputes Telegram lit-adj under legacy, production, and (when available) held-out
operating points. Softens the claim: direction under A1, not a precise 53.88.

Run: PYTHONPATH=src python src/telegram_lit_threshold_sensitivity.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from corpus_config import platform_analysis_mask
from gendered_harm_model import PERPETRATOR_ATTACK, VICTIM_DISCLOSURE
from structural_platform_evidence import PUBLIC_SCRAPABLE_RATIO
from whsi_literature_adjustment import wtshi_literature_construct


def _perp_rate(df: pd.DataFrame, t_sex: float, t_sex_th: float, t_th: float) -> float:
    n = 0
    hits = 0
    for _, row in df.iterrows():
        text = str(row.get("comment_text", ""))
        sp = float(row.get("gendered_harm_proba", 0) or 0)
        tp = float(row.get("threat_score", 0) or 0)
        victim = bool(VICTIM_DISCLOSURE.search(text))
        attack_kw = bool(PERPETRATOR_ATTACK.search(text))
        is_perp = (
            attack_kw
            or (sp >= t_sex and not victim)
            or (sp >= t_sex_th and tp >= t_th and not victim)
        )
        n += 1
        hits += int(is_perp)
    return hits / max(n, 1)


def _fuzzy_proxy(df: pd.DataFrame) -> float:
    """Use existing integrated fuzzy if available via WHSI file; else mid default."""
    path = "data/processed/whsi_scores_fair.csv"
    if os.path.exists(path):
        w = pd.read_csv(path)
        row = w[w["platform"] == "Telegram"]
        if not row.empty and "fuzzy_component" in row.columns:
            return float(row.iloc[0]["fuzzy_component"])
    return 30.64


def run() -> dict:
    master = pd.read_csv("data/processed/master_women_relevant.csv", low_memory=False)
    live = master[platform_analysis_mask(master)]
    live = live[(live["dataset_split"] == "live_scrape") & (live["platform"] == "Telegram")].copy()

    fuzzy = _fuzzy_proxy(live)
    w_share = float(
        (
            live.loc[live.get("harm_role", "") == "perpetrator_attack", "directed_at_women"]
            .fillna(0)
            .astype(float)
            .mean()
        )
        if "directed_at_women" in live.columns
        else 0.58
    )
    # directed share among perp under production labels in file
    perp_rows = live[live.get("harm_role", pd.Series(dtype=str)) == "perpetrator_attack"]
    if len(perp_rows) and "directed_at_women" in perp_rows.columns:
        w_share = float(pd.to_numeric(perp_rows["directed_at_women"], errors="coerce").fillna(0).mean())

    scenarios = {
        "legacy_june_thresholds": (0.52, 0.45, 0.38),
        "production_july_thresholds": (0.40, 0.34, 0.28),
    }

    # Optional held-out-validated point — same as production until Step 2 completes
    scenarios["heldout_operating_point_or_production_proxy"] = (0.40, 0.34, 0.28)

    rows = []
    for name, (ts, tst, th) in scenarios.items():
        obs = _perp_rate(live, ts, tst, th)
        wtshi_lit, eher, meta = wtshi_literature_construct("Telegram", obs, w_share * 100)
        whsi_lit = float(np.clip(0.70 * wtshi_lit + 0.30 * fuzzy, 0, 100))
        # scrapable ratio band
        band_vals = []
        for ratio in (0.05, 0.10, 0.20):
            wt2, _, _ = wtshi_literature_construct("Telegram", obs, w_share * 100, scrapable_ratio=ratio)
            band_vals.append(float(np.clip(0.70 * wt2 + 0.30 * fuzzy, 0, 100)))
        rows.append(
            {
                "scenario": name,
                "thresholds": {"sexist": ts, "sexist_with_threat": tst, "threat_min": th},
                "classifier_flagged_rate_pct": round(obs * 100, 2),
                "below_estimated_fpr": obs < 0.125,
                "EHER": eher,
                "WHSI_literature_adjusted": round(whsi_lit, 2),
                "scrapable_ratio_band": [round(min(band_vals), 2), round(max(band_vals), 2)],
            }
        )

    claim = {
        "headline_point_53_88": "ASTERISK_OR_OMIT_in_headline_tables",
        "conditional_claim": (
            "Under Assumption A1 and the AI Forensics visibility estimate (scrapable ratio ≈ 0.10), "
            "any plausible Telegram harm rate in our women-relevant public scrape implies elevated "
            "ecosystem risk relative to the observable WHSI_raw. Our scrape cannot measure the "
            "true private-channel rate directly (Rogan–Gladen censored: observed flag rate sits "
            "near/below estimated false-positive rate). The July point estimate 53.88 is "
            "threshold-sensitive and should not be read as a precise ecosystem score; the "
            "supported claim is directional elevation under A1, with the full threshold × "
            "scrapable-ratio sensitivity range reported in this table."
        ),
        "examiner_one_liner": (
            "53.88 is not a measurement of private Telegram GBV density; it is an A1/"
            "visibility extrapolation from a noisy public-scrape flag rate. Direction survives; "
            "precision does not."
        ),
    }

    out = {
        "platform": "Telegram",
        "n_live": len(live),
        "fuzzy_used": fuzzy,
        "women_targeting_share_used": round(w_share, 4),
        "public_scrapable_ratio_default": PUBLIC_SCRAPABLE_RATIO.get("Telegram"),
        "scenarios": rows,
        "claim": claim,
    }
    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/telegram_lit_threshold_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(rows, indent=2))
    print("\n" + claim["conditional_claim"])
    print("Saved → outputs/results/telegram_lit_threshold_sensitivity.json")
    return out


if __name__ == "__main__":
    run()
