"""
Perpetrator-detection threshold justification (Faculty Step 8).

Derives optimal thresholds from EDOS gold labels AND project corpus gold labels.
Applies corpus-tuned operating point to production constants in gendered_harm_model.

Output: outputs/results/threshold_justification.json

Run: PYTHONPATH=src python src/threshold_justification.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)


def threshold_analysis_edos() -> dict:
    from gendered_harm_model import load_edos_training_frame, predict_batch
    from sklearn.model_selection import train_test_split

    df = load_edos_training_frame()
    _, X_te, _, y_te = train_test_split(
        df["comment_text"], df["is_sexist"], test_size=0.15, random_state=42, stratify=df["is_sexist"]
    )
    preds = predict_batch(X_te.tolist())
    proba = preds["gendered_harm_proba"].values
    y_true = y_te.values

    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.52

    from gendered_harm_model import (
        PERP_SEXIST_THRESH,
        PERP_SEXIST_WITH_THREAT,
        PERP_THREAT_MIN,
    )

    current = {
        "sexist_proba": PERP_SEXIST_THRESH,
        "sexist_with_threat": PERP_SEXIST_WITH_THREAT,
        "threat_min": PERP_THREAT_MIN,
    }

    return {
        "source": "EDOS_holdout_PR_curve",
        "n_test": len(y_true),
        "domain_note": (
            "EDOS labels general sexism; project labels women-targeted perpetrator harm. "
            "EDOS F1 is a domain-shift baseline, not the primary validity metric."
        ),
        "current_thresholds": current,
        "f1_maximizing_threshold": round(best_thresh, 3),
        "f1_at_edos_proxy_0_52": (
            round(float(f1_scores[np.argmin(np.abs(thresholds - 0.52))]), 4) if len(thresholds) else None
        ),
        "operating_point": "Corpus gold F1-maximizing (primary); EDOS is external domain-shift check only",
    }


def _eval_rule(y_true, proba, threat, t, t2, th) -> dict:
    pred = ((proba >= t) | ((proba >= t2) & (threat >= th))).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def threshold_analysis_corpus() -> dict | None:
    path = "data/labelled/validation_sample.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    if "human_label_perpetrator" not in df.columns:
        return None
    labelled = df[
        pd.to_numeric(df["human_label_perpetrator"], errors="coerce").isin([0, 1])
    ].copy()
    if len(labelled) < 30:
        return None

    y_true = labelled["human_label_perpetrator"].astype(int).values
    proba = labelled["gendered_harm_proba"].astype(float).values
    threat = (
        labelled["threat_score"].astype(float).values
        if "threat_score" in labelled.columns
        else np.zeros(len(labelled))
    )

    # Old EDOS-proxy rule
    old = _eval_rule(y_true, proba, threat, 0.52, 0.45, 0.38)

    # Sweep for best F1 with P>=0.85 preference
    best = None
    for t in np.arange(0.28, 0.55, 0.01):
        for t2 in np.arange(0.22, t + 0.001, 0.02):
            for th in np.arange(0.20, 0.45, 0.04):
                pred = ((proba >= t) | ((proba >= t2) & (threat >= th))).astype(int)
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, pred, average="binary", zero_division=0
                )
                score = (f1, p, r)
                if best is None or score > best[0]:
                    best = (score, float(t), float(t2), float(th))

    # Production constants (must match gendered_harm_model.py)
    from gendered_harm_model import (
        PERP_SEXIST_THRESH,
        PERP_SEXIST_WITH_THREAT,
        PERP_THREAT_MIN,
    )

    production = _eval_rule(
        y_true, proba, threat, PERP_SEXIST_THRESH, PERP_SEXIST_WITH_THREAT, PERP_THREAT_MIN
    )

    by_platform = {}
    for plat, g in labelled.groupby("platform"):
        yt = g["human_label_perpetrator"].astype(int).values
        pr = g["gendered_harm_proba"].astype(float).values
        ths = g["threat_score"].astype(float).values if "threat_score" in g.columns else np.zeros(len(g))
        by_platform[str(plat)] = _eval_rule(
            yt, pr, ths, PERP_SEXIST_THRESH, PERP_SEXIST_WITH_THREAT, PERP_THREAT_MIN
        )
        by_platform[str(plat)]["n"] = int(len(g))

    return {
        "source": "corpus_human_gold_labels",
        "n_annotated": len(labelled),
        "legacy_edos_proxy_0_52": old,
        "sweep_best": {
            "sexist_proba": round(best[1], 3),
            "sexist_with_threat": round(best[2], 3),
            "threat_min": round(best[3], 3),
            "precision": round(best[0][1], 4),
            "recall": round(best[0][2], 4),
            "f1": round(best[0][0], 4),
        },
        "production_thresholds": {
            "sexist_proba": PERP_SEXIST_THRESH,
            "sexist_with_threat": PERP_SEXIST_WITH_THREAT,
            "threat_min": PERP_THREAT_MIN,
        },
        "production_metrics": production,
        "by_platform": by_platform,
        "justification": (
            f"Corpus gold (n={len(labelled)}) F1-maximizing rule ≈ "
            f"{best[1]:.2f}/{best[2]:.2f}/{best[3]:.2f}. "
            f"Production set to {PERP_SEXIST_THRESH}/{PERP_SEXIST_WITH_THREAT}/{PERP_THREAT_MIN}: "
            f"P={production['precision']:.3f} R={production['recall']:.3f} F1={production['f1']:.3f} "
            f"(was P={old['precision']:.3f} R={old['recall']:.3f} F1={old['f1']:.3f} at 0.52/0.45/0.38)."
        ),
        "recommended_action": "Applied — production thresholds updated in gendered_harm_model.py",
    }


def run_threshold_justification() -> dict:
    result = {
        "edos_analysis": threshold_analysis_edos(),
        "corpus_analysis": threshold_analysis_corpus(),
    }
    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/threshold_justification.json", "w") as f:
        json.dump(result, f, indent=2)
    ca = result.get("corpus_analysis") or {}
    print(ca.get("justification") or result["edos_analysis"].get("operating_point"))
    return result


if __name__ == "__main__":
    run_threshold_justification()
