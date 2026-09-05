"""
Classifier validation — EDOS holdout + corpus human validation + Detoxify cross-check.

Faculty Step 1 workflow:
  1. export stratified sample (360 rows)
  2. dual human annotation (annotator_a / annotator_b)
  3. --adjudicate → gold labels + Cohen's κ
  4. --import-annotations → corpus-specific P/R/F1

Run:
  PYTHONPATH=src python src/validate_classifier.py
  PYTHONPATH=src python src/validate_classifier.py --export-only
  PYTHONPATH=src python src/validate_classifier.py --adjudicate
  PYTHONPATH=src python src/validate_classifier.py --import-annotations data/labelled/validation_sample.csv
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.insert(0, os.path.dirname(__file__))

# Faculty stratification: platform → (n_total, n_harmful, n_safe)
ANNOTATION_STRATA = {
    "YouTube": (120, 60, 60),
    "Reddit": (120, 60, 60),
    "Telegram": (80, 40, 40),
    "Twitter": (40, 20, 20),
}

SEVERE_LANGUAGE = re.compile(
    r"\b(?:fuck|fucking|shit|bitch|whore|slut|cunt|rape|raped|molest|assault|"
    r"kill her|deserve to die|stupid woman|dumb woman|all women|women are|feminazi|"
    r"horny|naked|slutty|misogyn|objectif|underwear|boobs|tits|sexual)\b",
    re.IGNORECASE,
)


def _classifier_harmful_flag(row: pd.Series) -> int:
    """Binary harmful flag for stratification (50/50 sampling)."""
    if int(row.get("is_gendered_harm", 0) or 0) == 1:
        return 1
    if str(row.get("harm_role", "")) == "perpetrator_attack":
        return 1
    if float(row.get("gendered_harm_proba", 0) or 0) >= 0.42:
        return 1
    return 0


def _classifier_perpetrator_flag(row: pd.Series) -> int:
    return int(str(row.get("harm_role", "")) == "perpetrator_attack")


def edos_holdout_validation() -> dict:
    from gendered_harm_model import load_edos_training_frame, predict_batch
    from sklearn.model_selection import train_test_split

    df = load_edos_training_frame()
    X = df["comment_text"]
    y = df["is_sexist"]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    preds = predict_batch(X_te.tolist())
    y_pred = preds["is_gendered_harm"].values
    y_perp_pred = (preds["harm_role"] == "perpetrator_attack").astype(int).values

    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary")
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "source": "EDOS_holdout",
        "n_test": len(y_te),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "sensitivity": round(float(rec), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
        "classification_report": classification_report(y_te, y_pred, output_dict=True),
        "confusion_matrix": cm.tolist(),
        "note": "EDOS sexist class — NOT corpus-specific; use corpus_validation when available",
    }


def _has_dual_annotations(path: str = "data/labelled/validation_sample.csv") -> bool:
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path, low_memory=False)
    for col in ("annotator_a_perpetrator", "annotator_b_perpetrator"):
        if col not in df.columns:
            return False
        filled = df[col].notna() & df[col].isin([0, 1, 0.0, 1.0, "0", "1", "0.0", "1.0"])
        if filled.sum() < len(df):
            return False
    return True


def import_numbers_annotations(
    numbers_path: str = "data/labelled/validation_sample.numbers",
    out_path: str = "data/labelled/validation_sample.csv",
) -> pd.DataFrame:
    """Import dual annotations from Apple Numbers into the pipeline CSV."""
    if not os.path.exists(numbers_path):
        raise FileNotFoundError(numbers_path)
    from numbers_parser import Document

    doc = Document(numbers_path)
    table = doc.sheets[0].tables[0]
    headers = [table.cell(0, c).value for c in range(table.num_cols)]
    rows = [
        [table.cell(r, c).value for c in range(table.num_cols)]
        for r in range(1, table.num_rows)
    ]
    df = pd.DataFrame(rows, columns=headers)
    for col in (
        "annotator_a_perpetrator",
        "annotator_b_perpetrator",
        "human_label_perpetrator",
        "classifier_perpetrator_flag",
        "classifier_harmful_flag",
        "is_gendered_harm",
        "has_severe_language",
    ):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: (
                    ""
                    if v is None or (isinstance(v, float) and pd.isna(v))
                    else str(int(float(v)))
                    if str(v).strip() in ("0", "1", "0.0", "1.0")
                    else str(v).strip()
                )
            )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    a = df["annotator_a_perpetrator"].isin(["0", "1"]).sum()
    b = df["annotator_b_perpetrator"].isin(["0", "1"]).sum()
    print(f"Imported Numbers annotations → {out_path} ({a}/{len(df)} A, {b}/{len(df)} B)")
    return df


def export_validation_sample(
    master_path: str = "data/processed/master_women_relevant.csv",
    out_path: str = "data/labelled/validation_sample.csv",
    force: bool = False,
) -> pd.DataFrame:
    """
    Faculty Step 1 stratification:
    - Platform counts per ANNOTATION_STRATA
    - 50% classifier-flagged harmful / 50% safe within each platform
    """
    if not force and _has_dual_annotations(out_path):
        print(f"Skipping export — dual annotations already in {out_path}")
        return pd.read_csv(out_path, low_memory=False)

    if not force and os.path.exists("data/labelled/validation_sample.numbers"):
        try:
            return import_numbers_annotations(out_path=out_path)
        except Exception as exc:
            print(f"  Numbers import failed ({exc}); exporting fresh sample")

    if not os.path.exists(master_path):
        print(f"Missing {master_path}")
        return pd.DataFrame()

    df = pd.read_csv(master_path, low_memory=False)
    df = df[df["comment_text"].astype(str).str.len() > 20].copy()
    df["_harmful_stratum"] = df.apply(_classifier_harmful_flag, axis=1)

    parts = []
    for platform, (n_total, n_harm, n_safe) in ANNOTATION_STRATA.items():
        plat_df = df[df["platform"] == platform].copy()
        if plat_df.empty:
            print(f"  Warning: no rows for {platform}")
            continue
        harmful = plat_df[plat_df["_harmful_stratum"] == 1]
        safe = plat_df[plat_df["_harmful_stratum"] == 0]
        h_sample = harmful.sample(min(len(harmful), n_harm), random_state=42) if len(harmful) else harmful
        s_sample = safe.sample(min(len(safe), n_safe), random_state=42) if len(safe) else safe
        parts.append(pd.concat([h_sample, s_sample], ignore_index=True))

    sample = pd.concat(parts, ignore_index=True).drop(columns=["_harmful_stratum"], errors="ignore")

    cols = [
        "comment_text", "platform", "community", "dataset_source", "dataset_split",
        "gendered_harm_proba", "is_gendered_harm", "harm_role",
        "has_severe_language", "content_severity",
        "toxicity_score", "threat_score",
    ]
    export = sample[[c for c in cols if c in sample.columns]].copy()

    # Dual-annotator columns (faculty requirement)
    for col in [
        "annotator_a_perpetrator", "annotator_a_role",
        "annotator_b_perpetrator", "annotator_b_role",
        "human_label_perpetrator", "human_label_role",
        "annotator_notes",
    ]:
        export[col] = ""

    export["classifier_perpetrator_flag"] = export.apply(_classifier_perpetrator_flag, axis=1)
    export["classifier_harmful_flag"] = export.apply(_classifier_harmful_flag, axis=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export.to_csv(out_path, index=False)
    print(f"Validation sample ({len(export)} rows) → {out_path}")
    print(export.groupby(["platform", "classifier_harmful_flag"]).size().to_string())
    return export


def adjudicate_annotations(path: str = "data/labelled/validation_sample.csv") -> dict:
    """Merge dual annotator labels → gold; compute inter-annotator κ."""
    df = pd.read_csv(path, low_memory=False)

    def _parse_label(s) -> int | None:
        s = str(s).strip()
        if s in ("0", "1", "0.0", "1.0"):
            return int(float(s))
        return None

    a = df["annotator_a_perpetrator"].apply(_parse_label) if "annotator_a_perpetrator" in df.columns else pd.Series([None] * len(df))
    b = df["annotator_b_perpetrator"].apply(_parse_label) if "annotator_b_perpetrator" in df.columns else pd.Series([None] * len(df))

    both = a.notna() & b.notna()
    if both.sum() == 0:
        return {"error": "No dual annotations found — fill annotator_a_perpetrator and annotator_b_perpetrator (0/1)"}

    kappa = cohen_kappa_score(a[both].astype(int), b[both].astype(int))

    gold = []
    disputed = []
    for i, row in df.iterrows():
        la, lb = _parse_label(row.get("annotator_a_perpetrator")), _parse_label(row.get("annotator_b_perpetrator"))
        ra = str(row.get("annotator_a_role", "")).strip()
        rb = str(row.get("annotator_b_role", "")).strip()
        if la is None and lb is None:
            gold.append(None)
            continue
        if la is not None and lb is not None:
            if la == lb:
                gold.append(la)
            else:
                gold.append(None)
                disputed.append(i)
        elif la is not None:
            gold.append(la)
        else:
            gold.append(lb)

    df["human_label_perpetrator"] = gold
    if "human_label_role" not in df.columns:
        df["human_label_role"] = ""
    for col in ("human_label_role", "annotator_a_role", "annotator_b_role", "annotator_notes"):
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("").astype(str)
    for i, row in df.iterrows():
        if pd.isna(row.get("human_label_perpetrator")):
            continue
        ra = str(row.get("annotator_a_role", "")).strip()
        rb = str(row.get("annotator_b_role", "")).strip()
        if ra and rb and ra == rb:
            df.at[i, "human_label_role"] = ra
        elif ra:
            df.at[i, "human_label_role"] = ra

    df.to_csv(path, index=False)
    os.makedirs("outputs/results", exist_ok=True)
    result = {
        "n_dual_annotated": int(both.sum()),
        "cohen_kappa_perpetrator": round(float(kappa), 4),
        "kappa_acceptable": kappa >= 0.60,
        "n_disputed_rows": len(disputed),
        "disputed_row_indices": disputed[:50],
        "action_if_kappa_low": "Revise codebook (docs/ANNOTATION_CODEBOOK.md), discuss disagreements, re-annotate disputed rows",
    }
    with open("outputs/results/annotation_agreement.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Cohen's κ = {kappa:.3f} ({'OK' if kappa >= 0.60 else 'BELOW 0.60 — revise codebook'})")
    print(f"Disputed rows: {len(disputed)}")
    return result


def validate_on_human_annotations(path: str) -> dict:
    """Corpus-specific validation against gold perpetrator labels."""
    df = pd.read_csv(path, low_memory=False)
    label_col = "human_label_perpetrator"
    if label_col not in df.columns:
        label_col = "human_label_gendered_harm"

    labelled = df[df[label_col].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    if labelled.empty:
        return {"error": "No completed human labels — fill human_label_perpetrator (0/1)"}

    y_true = labelled[label_col].astype(int)
    y_pred = (labelled.get("harm_role", "") == "perpetrator_attack").astype(int)
    if "classifier_perpetrator_flag" in labelled.columns:
        y_pred = labelled["classifier_perpetrator_flag"].astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    by_platform = {}
    for plat, g in labelled.groupby("platform"):
        yt = g[label_col].astype(int)
        yp = g["classifier_perpetrator_flag"].astype(int) if "classifier_perpetrator_flag" in g.columns else (g["harm_role"] == "perpetrator_attack").astype(int)
        pp, pr, pf, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        pcm = confusion_matrix(yt, yp)
        if pcm.size == 4:
            ptn, pfp, _, _ = pcm.ravel()
            psp = ptn / (ptn + pfp) if (ptn + pfp) else 0.0
        else:
            psp = 0.0
        by_platform[plat] = {
            "n": len(g),
            "precision": round(float(pp), 4),
            "recall": round(float(pr), 4),
            "sensitivity": round(float(pr), 4),
            "specificity": round(float(psp), 4),
            "f1": round(float(pf), 4),
        }

    return {
        "source": "corpus_human_validation",
        "n_annotated": len(labelled),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "sensitivity": round(float(rec), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "by_platform": by_platform,
        "note": "Gold labels from dual-annotator adjudication on project corpus",
    }


def detoxify_independent_validation(
    master_path: str = "data/processed/master_women_relevant.csv",
    n: int = 500,
) -> dict:
    from hybrid_classifier import _load_detoxify
    from corpus_config import platform_analysis_mask

    if not os.path.exists(master_path):
        return {"error": f"Missing {master_path}"}

    df = pd.read_csv(master_path, low_memory=False)
    live = df[platform_analysis_mask(df)].copy()
    live = live[live["comment_text"].astype(str).str.len() > 20]
    if live.empty:
        return {"error": "No live_scrape rows"}

    per_plat = max(50, n // max(live["platform"].nunique(), 1))
    parts = [g.sample(min(len(g), per_plat), random_state=42) for _, g in live.groupby("platform")]
    sample = pd.concat(parts, ignore_index=True).head(n)

    detox = _load_detoxify()
    if detox is None:
        return {"error": "Detoxify unavailable"}

    texts = sample["comment_text"].fillna("").astype(str).tolist()
    tox_scores, id_attack, threat = [], [], []
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        out = detox.predict(batch)
        for j in range(len(batch)):
            tox_scores.append(float(out["toxicity"][j]))
            id_attack.append(float(out["identity_attack"][j]))
            threat.append(float(out["threat"][j]))

    sample = sample.copy()
    sample["detoxify_toxicity"] = tox_scores
    sample["detoxify_identity_attack"] = id_attack
    sample["detoxify_threat"] = threat
    sample["detoxify_harm_flag"] = (
        (sample["detoxify_toxicity"] >= 0.55)
        | (sample["detoxify_identity_attack"] >= 0.45)
        | (sample["detoxify_threat"] >= 0.45)
    ).astype(int)

    y_model = sample["is_gendered_harm"].astype(int)
    y_detox = sample["detoxify_harm_flag"].astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_model, y_detox, average="binary")
    kappa = cohen_kappa_score(y_model, y_detox)

    disagree = sample[y_model != y_detox].copy()
    disagree_path = "data/labelled/validation_disagreements_full.csv"
    os.makedirs(os.path.dirname(disagree_path), exist_ok=True)
    disagree.to_csv(disagree_path, index=False)

    return {
        "source": "detoxify_independent_crosscheck",
        "n_sample": len(sample),
        "note": "Detoxify is NOT human ground truth — see docs/DETOXIFY_DISAGREEMENT_ANALYSIS.md",
        "precision_model_vs_detox": round(float(prec), 4),
        "recall_model_vs_detox": round(float(rec), 4),
        "f1_model_vs_detox": round(float(f1), 4),
        "cohen_kappa": round(float(kappa), 4),
        "model_harm_rate": round(float(y_model.mean()), 4),
        "detoxify_harm_rate": round(float(y_detox.mean()), 4),
        "disagreement_export": disagree_path,
        "n_disagreements": len(disagree),
        "confusion_matrix": confusion_matrix(y_model, y_detox).tolist(),
    }


def run_validation(import_path: str | None = None, adjudicate: bool = False) -> dict:
    os.makedirs("outputs/results", exist_ok=True)
    results = {"edos_holdout": edos_holdout_validation()}
    results["detoxify_independent"] = detoxify_independent_validation()

    sample_path = "data/labelled/validation_sample.csv"
    if not os.path.exists(sample_path) or os.path.getmtime(sample_path) < 0:
        export_validation_sample(out_path=sample_path)

    if adjudicate:
        results["annotation_agreement"] = adjudicate_annotations(sample_path)

    if import_path and os.path.exists(import_path):
        results["corpus_validation"] = validate_on_human_annotations(import_path)
    else:
        # Check if any labels exist in default sample
        sample = pd.read_csv(sample_path, low_memory=False)
        if "human_label_perpetrator" in sample.columns:
            partial = validate_on_human_annotations(sample_path)
            if "error" not in partial:
                results["corpus_validation"] = partial
            else:
                results["corpus_validation"] = {
                    "status": "pending",
                    "error": partial["error"],
                    "instruction": "Complete dual annotation per docs/ANNOTATION_CODEBOOK.md",
                }

    out = "outputs/results/classifier_validation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    ed = results["edos_holdout"]
    print(f"\nEDOS holdout: P={ed['precision']:.3f} R={ed['recall']:.3f} F1={ed['f1']:.3f}")
    if "corpus_validation" in results and results["corpus_validation"].get("f1"):
        cv = results["corpus_validation"]
        print(f"Corpus validation: P={cv['precision']:.3f} R={cv['recall']:.3f} F1={cv['f1']:.3f} (n={cv['n_annotated']})")
    elif results.get("corpus_validation", {}).get("status") == "pending":
        print("Corpus validation: PENDING — complete human annotation")
    print(f"Saved → {out}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--import-annotations", default=None)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--adjudicate", action="store_true")
    parser.add_argument("--import-numbers", action="store_true", help="Import data/labelled/validation_sample.numbers → CSV")
    args = parser.parse_args()

    if args.import_numbers:
        import_numbers_annotations()
        run_validation(adjudicate=True)
    elif args.export_only:
        export_validation_sample()
    else:
        run_validation(import_path=args.import_annotations, adjudicate=args.adjudicate)
