"""
Faculty Round 3 Step 6 — Wilson intervals on small-n held-out validation cells.

Adds Wilson 95% CIs for precision/recall/specificity per platform (unweighted
sample cells) and writes a report-ready markdown table.

Run: PYTHONPATH=src python src/heldout_validation_intervals.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gendered_harm_model import _scores_from_proba

GOLD = "data/labelled/v2/gold_heldout.csv"
OUT_JSON = "outputs/results/heldout_validation_intervals.json"
OUT_MD = "outputs/results/heldout_validation_intervals.md"
OUT_CSV = "outputs/results/heldout_validation_intervals.csv"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(p), float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def fmt(p, lo, hi) -> str:
    if not np.isfinite(p):
        return "—"
    return f"{p:.3f} [{lo:.3f}–{hi:.3f}]"


def metrics_for(y: np.ndarray, yhat: np.ndarray) -> dict:
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    prec = wilson(tp, tp + fp)
    rec = wilson(tp, tp + fn)
    spec = wilson(tn, tn + fp)
    # F1 point only (no simple Wilson); mark indicative
    p, r = prec[0], rec[0]
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "n": int(len(y)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "n_flags": tp + fp,
        "precision": {"point": round(prec[0], 4), "wilson_low": round(prec[1], 4), "wilson_high": round(prec[2], 4)},
        "recall": {"point": round(rec[0], 4), "wilson_low": round(rec[1], 4), "wilson_high": round(rec[2], 4)},
        "specificity": {"point": round(spec[0], 4), "wilson_low": round(spec[1], 4), "wilson_high": round(spec[2], 4)},
        "f1_point": round(f1, 4),
        "label": "indicative_only_small_n" if len(y) < 80 or (tp + fp) < 25 else "report_with_CI",
    }


def main() -> None:
    gold = pd.read_csv(GOLD)
    yhats = []
    for _, row in gold.iterrows():
        out = _scores_from_proba(
            str(row.get("comment_text", "")),
            float(row.get("gendered_harm_proba", 0) or 0),
            float(row.get("threat_score", 0) or 0),
        )
        yhats.append(1 if out["harm_role"] == "perpetrator_attack" else 0)
    gold = gold.copy()
    gold["yhat"] = yhats
    gold["y"] = gold["human_label_perpetrator"].astype(int)

    by = {}
    rows = []
    overall = metrics_for(gold["y"].to_numpy(), gold["yhat"].to_numpy())
    by["OVERALL"] = overall
    rows.append(
        {
            "platform": "OVERALL",
            "n": overall["n"],
            "n_flags": overall["n_flags"],
            "precision_wilson": fmt(
                overall["precision"]["point"],
                overall["precision"]["wilson_low"],
                overall["precision"]["wilson_high"],
            ),
            "recall_wilson": fmt(
                overall["recall"]["point"],
                overall["recall"]["wilson_low"],
                overall["recall"]["wilson_high"],
            ),
            "specificity_wilson": fmt(
                overall["specificity"]["point"],
                overall["specificity"]["wilson_low"],
                overall["specificity"]["wilson_high"],
            ),
            "f1_point": overall["f1_point"],
            "cell_label": overall["label"],
        }
    )

    for plat, g in gold.groupby("platform"):
        m = metrics_for(g["y"].to_numpy(), g["yhat"].to_numpy())
        by[plat] = m
        rows.append(
            {
                "platform": plat,
                "n": m["n"],
                "n_flags": m["n_flags"],
                "precision_wilson": fmt(
                    m["precision"]["point"],
                    m["precision"]["wilson_low"],
                    m["precision"]["wilson_high"],
                ),
                "recall_wilson": fmt(
                    m["recall"]["point"],
                    m["recall"]["wilson_low"],
                    m["recall"]["wilson_high"],
                ),
                "specificity_wilson": fmt(
                    m["specificity"]["point"],
                    m["specificity"]["wilson_low"],
                    m["specificity"]["wilson_high"],
                ),
                "f1_point": m["f1_point"],
                "cell_label": m["label"],
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Wilson intervals on unweighted stratified held-out cells. "
            "These describe the sample; corpus estimates use IPW bootstrap CIs "
            "in heldout_ipw_metrics.json. Small-n cells (esp. Telegram n=40, "
            "Twitter n=20, YouTube ~15 flags) are labelled indicative_only."
        ),
        "by_platform": by,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    md = [
        "# Held-out validation cells with Wilson 95% intervals (Step 6)",
        "",
        payload["note"],
        "",
        "| Platform | n | flags | Precision (Wilson) | Recall (Wilson) | Specificity (Wilson) | F1 (point) | Label |",
        "|----------|---|-------|--------------------|-----------------|----------------------|------------|-------|",
    ]
    for r in rows:
        md.append(
            f"| {r['platform']} | {r['n']} | {r['n_flags']} | {r['precision_wilson']} | "
            f"{r['recall_wilson']} | {r['specificity_wilson']} | {r['f1_point']:.3f} | {r['cell_label']} |"
        )
    md += [
        "",
        "Corpus-level (IPW) intervals: see `outputs/results/heldout_ipw_metrics.json`.",
        "",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))

    print(table.to_string(index=False))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
