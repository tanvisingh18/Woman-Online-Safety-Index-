"""
Held-out evaluation of FROZEN perpetrator thresholds (Faculty July Step 2).

- In-sample metrics on June n=360 are labelled optimistic / tuning-set only.
- Primary metrics come from data/labelled/v2/gold_heldout.csv after Protocol v2.0.

Run: PYTHONPATH=src python src/heldout_threshold_eval.py
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from gendered_harm_model import (
    PERP_SEXIST_THRESH,
    PERP_SEXIST_WITH_THREAT,
    PERP_THREAT_MIN,
    _scores_from_proba,
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _eval(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp + 1e-9)
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "sensitivity": round(float(r), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }


def _pred_from_proba(df: pd.DataFrame) -> np.ndarray:
    preds = []
    for _, row in df.iterrows():
        text = str(row.get("comment_text", ""))
        sp = float(row.get("gendered_harm_proba", 0) or 0)
        tp = float(row.get("threat_score", 0) or 0)
        out = _scores_from_proba(text, sp, tp)
        preds.append(1 if out["harm_role"] == "perpetrator_attack" else 0)
    return np.array(preds)


def in_sample_june() -> dict:
    path = "data/labelled/validation_sample.csv"
    df = pd.read_csv(path, low_memory=False)
    y = pd.to_numeric(df["human_label_perpetrator"], errors="coerce")
    df = df[y.isin([0, 1])].copy()
    y = y.loc[df.index].astype(int).values
    pred = _pred_from_proba(df)
    metrics = _eval(y, pred)
    return {
        "label": "IN_SAMPLE_optimistic_tuning_set",
        "warning": (
            "Thresholds 0.40/0.34/0.28 were selected on this same June n=360 set. "
            "F1 here is optimistically biased and MUST NOT be presented as primary validity."
        ),
        "thresholds": {
            "sexist_proba": PERP_SEXIST_THRESH,
            "sexist_with_threat": PERP_SEXIST_WITH_THREAT,
            "threat_min": PERP_THREAT_MIN,
        },
        "metrics": metrics,
        "source_file": path,
    }


def heldout_v2() -> dict | None:
    path = "data/labelled/v2/gold_heldout.csv"
    partial = "data/labelled/v2/gold_heldout_partial.csv"
    use = path if os.path.exists(path) else (partial if os.path.exists(partial) else None)
    if use is None:
        return {
            "label": "HELD_OUT_PRIMARY",
            "status": "pending_protocol_v2_annotation",
            "instruction": (
                "Complete docs/WOSI_ANNOTATION_PROTOCOL_V2.md; write gold_heldout.csv; re-run this script."
            ),
        }

    gold = pd.read_csv(use, low_memory=False)
    # Need proba — merge from master sample if missing
    if "gendered_harm_proba" not in gold.columns:
        master = pd.read_csv("data/labelled/v2/annotation_sample_master.csv", low_memory=False)
        gold = gold.merge(
            master[["row_id", "comment_text", "platform", "gendered_harm_proba", "threat_score"]],
            on="row_id",
            how="left",
            suffixes=("", "_m"),
        )
        if "comment_text" not in gold.columns or gold["comment_text"].isna().all():
            gold["comment_text"] = gold.get("comment_text_m", gold.get("comment_text"))

    y = pd.to_numeric(gold["human_label_perpetrator"], errors="coerce")
    labelled = gold[y.isin([0, 1])].copy()
    if len(labelled) < 30:
        return {
            "label": "HELD_OUT_PRIMARY",
            "status": "insufficient_adjudicated_labels",
            "n_ready": int(len(labelled)),
        }

    y_true = y.loc[labelled.index].astype(int).values
    pred = _pred_from_proba(labelled)
    metrics = _eval(y_true, pred)
    by_plat = {}
    for plat, g in labelled.groupby("platform"):
        yt = pd.to_numeric(g["human_label_perpetrator"]).astype(int).values
        yp = _pred_from_proba(g)
        by_plat[str(plat)] = _eval(yt, yp)

    return {
        "label": "HELD_OUT_PRIMARY",
        "status": "ready",
        "metrics": metrics,
        "by_platform": by_plat,
        "source_file": use,
        "frozen_threshold_hash": _sha256("configs/perpetrator_thresholds_frozen.json"),
    }


def main() -> dict:
    os.makedirs("outputs/results", exist_ok=True)
    cfg_hash = _sha256("configs/perpetrator_thresholds_frozen.json")
    out = {
        "frozen_config": "configs/perpetrator_thresholds_frozen.json",
        "frozen_config_sha256": cfg_hash,
        "code_thresholds": {
            "PERP_SEXIST_THRESH": PERP_SEXIST_THRESH,
            "PERP_SEXIST_WITH_THREAT": PERP_SEXIST_WITH_THREAT,
            "PERP_THREAT_MIN": PERP_THREAT_MIN,
        },
        "in_sample_june_360": in_sample_june(),
        "held_out_v2": heldout_v2(),
        "reporting_rule": (
            "Every table/summary that shows in-sample F1 must show held-out P/R/F1 beside it. "
            "Until held-out is ready, primary validity is PENDING — do not lead with 0.93."
        ),
    }
    path = "outputs/results/heldout_vs_insample_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Frozen config SHA-256: {cfg_hash}")
    print(f"In-sample F1: {out['in_sample_june_360']['metrics']['f1']} (OPTIMISTIC — demote)")
    ho = out["held_out_v2"]
    if ho.get("status") == "ready":
        print(f"Held-out F1 (PRIMARY): {ho['metrics']['f1']}")
    else:
        print(f"Held-out: {ho.get('status')} — {ho.get('instruction', '')}")
    print(f"Saved → {path}")
    return out


if __name__ == "__main__":
    main()
