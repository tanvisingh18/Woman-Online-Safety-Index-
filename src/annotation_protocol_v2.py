"""
WOSI Annotation Protocol v2.0 — Sampling Lead tools.

Exports a fresh 200-row sample (excluding June validation_sample rows),
strips classifier columns for annotators, writes pre-annotation hashes.

Run:
  PYTHONPATH=src python src/annotation_protocol_v2.py --export-sample
  PYTHONPATH=src python src/annotation_protocol_v2.py --adjudicate --file-a ... --file-b ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

SEED = 20260712
OUT_DIR = "data/labelled/v2"
JUNE_PATH = "data/labelled/validation_sample.csv"
MASTER = "data/processed/master_women_relevant.csv"
THRESH_CFG = "configs/perpetrator_thresholds_frozen.json"

STRATA = {
    # platform: (n_total, n_high, n_mid, n_low)
    "YouTube": (70, 28, 21, 21),
    "Reddit": (70, 28, 21, 21),
    "Telegram": (40, 16, 12, 12),
    "Twitter": (20, 8, 6, 12),  # last bucket = low; table said 6+6 for mid+low but 8+6+6=20
}

# Fix Twitter to 8+6+6=20
STRATA["Twitter"] = (20, 8, 6, 6)

ANNOTATOR_KEEP = ["row_id", "comment_text", "platform"]
CLASSIFIER_STRIP = [
    "gendered_harm_proba",
    "is_gendered_harm",
    "harm_role",
    "toxicity_score",
    "threat_score",
    "classifier_perpetrator_flag",
    "classifier_harmful_flag",
    "has_severe_language",
    "content_severity",
    "directed_at_women",
    "targets_women",
]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _band(proba: float) -> str:
    if proba >= 0.55:
        return "high"
    if proba >= 0.35:
        return "mid"
    return "low"


def _text_key(s: str) -> str:
    return " ".join(str(s).lower().split())[:500]


def export_sample() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    master = pd.read_csv(MASTER, low_memory=False)
    june_texts = set()
    if os.path.exists(JUNE_PATH):
        june = pd.read_csv(JUNE_PATH, low_memory=False)
        june_texts = set(june["comment_text"].map(_text_key))

    # Live + limited Twitter hist
    live = master[master["dataset_split"] == "live_scrape"].copy()
    tw = master[master["dataset_split"] == "historical_twitter"].copy()
    pool = pd.concat([live, tw], ignore_index=True)
    pool = pool[pool["comment_text"].astype(str).str.len() > 20].copy()
    pool["_tkey"] = pool["comment_text"].map(_text_key)
    pool = pool[~pool["_tkey"].isin(june_texts)].copy()
    pool["gendered_harm_proba"] = pd.to_numeric(pool.get("gendered_harm_proba"), errors="coerce").fillna(0)
    pool["_band"] = pool["gendered_harm_proba"].map(_band)

    parts = []
    for plat, (n_tot, n_hi, n_mid, n_lo) in STRATA.items():
        plat_df = pool[pool["platform"] == plat]
        if plat_df.empty:
            print(f"WARNING: no rows for {plat}")
            continue
        for band, n_need in [("high", n_hi), ("mid", n_mid), ("low", n_lo)]:
            sub = plat_df[plat_df["_band"] == band]
            take = min(len(sub), n_need)
            if take < n_need:
                print(f"  WARNING: {plat}/{band} only {take}/{n_need}")
            if take == 0:
                continue
            idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
            parts.append(sub.loc[idx])

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "row_id", [f"v2_{i:04d}" for i in range(len(sample))])

    master_path = f"{OUT_DIR}/annotation_sample_master.csv"
    sample.to_csv(master_path, index=False)

    blank_a = sample[ANNOTATOR_KEEP].copy()
    blank_a["annotator_a_perpetrator"] = ""
    blank_a["annotator_a_role"] = ""
    blank_a["annotator_notes"] = ""
    path_a = f"{OUT_DIR}/annotator_A_blank.csv"
    blank_a.to_csv(path_a, index=False)

    blank_b = sample[ANNOTATOR_KEEP].copy()
    blank_b["annotator_b_perpetrator"] = ""
    blank_b["annotator_b_role"] = ""
    blank_b["annotator_notes"] = ""
    path_b = f"{OUT_DIR}/annotator_B_blank.csv"
    blank_b.to_csv(path_b, index=False)

    # Independence statement templates
    for who in ("A", "B"):
        with open(f"{OUT_DIR}/independence_{who}_TEMPLATE.txt", "w") as f:
            f.write(
                f"I, [NAME], labelled the WOSI v2 annotation file assigned to Annotator {who} "
                "without access to the other annotator's labels and without access to classifier "
                "score columns. I did not discuss individual cases with the other annotator before "
                "submission of my raw file.\n\nSignature: _______________  Date: _______________\n"
            )

    thresh_hash = _sha256_file(THRESH_CFG) if os.path.exists(THRESH_CFG) else None
    manifest = {
        "protocol": "docs/WOSI_ANNOTATION_PROTOCOL_V2.md",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling_seed": SEED,
        "n_rows": len(sample),
        "excluded_june_validation_texts": len(june_texts),
        "strata": {k: list(v) for k, v in STRATA.items()},
        "platform_counts": sample["platform"].value_counts().to_dict(),
        "band_counts": sample["_band"].value_counts().to_dict() if "_band" in sample.columns else {},
        "sha256": {
            "annotation_sample_master.csv": _sha256_file(master_path),
            "annotator_A_blank.csv": _sha256_file(path_a),
            "annotator_B_blank.csv": _sha256_file(path_b),
            "perpetrator_thresholds_frozen.json": thresh_hash,
        },
        "classifier_columns_stripped_from_blank_files": True,
        "june_360_status": "tuning/in-sample only — not for primary held-out metrics",
    }
    # recompute band on sample for counts
    if "gendered_harm_proba" in sample.columns:
        sample["_band"] = sample["gendered_harm_proba"].map(_band)
        manifest["band_counts"] = sample["_band"].value_counts().to_dict()

    man_path = f"{OUT_DIR}/PRE_ANNOTATION_MANIFEST.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log_path = f"{OUT_DIR}/annotation_log.md"
    with open(log_path, "w") as f:
        f.write("# WOSI v2 Annotation Log\n\n")
        f.write(f"- Created (UTC): {manifest['created_at']}\n")
        f.write(f"- Sampling seed: `{SEED}`\n")
        f.write(f"- n rows: {len(sample)}\n")
        f.write(f"- Frozen threshold config SHA-256: `{thresh_hash}`\n")
        f.write(f"- Blank A SHA-256: `{manifest['sha256']['annotator_A_blank.csv']}`\n")
        f.write(f"- Blank B SHA-256: `{manifest['sha256']['annotator_B_blank.csv']}`\n")
        f.write(f"- Master SHA-256: `{manifest['sha256']['annotation_sample_master.csv']}`\n\n")
        f.write("## Stratum counts\n\n")
        f.write(sample.groupby(["platform", "_band"]).size().to_string() + "\n\n")
        f.write("## Session logs (fill during annotation)\n\n")
        f.write("### Annotator A\n- Start:\n- End:\n- Classifier columns visible? NO\n- Interruptions:\n\n")
        f.write("### Annotator B\n- Start:\n- End:\n- Classifier columns visible? NO\n- Interruptions:\n\n")
        f.write("## Post-submission (Sampling Lead)\n")
        f.write("- Raw A SHA-256:\n- Raw B SHA-256:\n- Cohen kappa:\n- n_disputed:\n")
        f.write("- If kappa > 0.95: STOP — escalate to faculty with raw files\n")
        f.write("- If kappa < 0.50: revise codebook and re-run\n")

    print(f"Exported {len(sample)} rows → {OUT_DIR}/")
    print(f"Manifest → {man_path}")
    print(f"Log → {log_path}")
    print("Sampling Lead: distribute ONLY annotator_*_blank.csv files. Do not annotate.")


def _parse01(s) -> int | None:
    s = str(s).strip()
    if s in ("0", "1", "0.0", "1.0"):
        return int(float(s))
    return None


def adjudicate(file_a: str, file_b: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    a = pd.read_csv(file_a, low_memory=False)
    b = pd.read_csv(file_b, low_memory=False)
    m = a.merge(b, on="row_id", suffixes=("_Afile", "_Bfile"))
    # normalize column names
    la = m["annotator_a_perpetrator"].map(_parse01) if "annotator_a_perpetrator" in m.columns else m.filter(like="annotator_a").iloc[:, 0].map(_parse01)
    lb = m["annotator_b_perpetrator"].map(_parse01) if "annotator_b_perpetrator" in m.columns else None
    if "annotator_b_perpetrator" in m.columns:
        lb = m["annotator_b_perpetrator"].map(_parse01)
    else:
        # B file may use annotator_b on its own
        cand = [c for c in b.columns if "perpetrator" in c]
        lb = m.merge(b[["row_id"] + cand], on="row_id")[cand[0]].map(_parse01)

    both = la.notna() & lb.notna()
    kappa = float(cohen_kappa_score(la[both].astype(int), lb[both].astype(int))) if both.sum() else float("nan")
    disputed = []
    finals = []
    reasons = []
    for i in range(len(m)):
        va, vb = la.iloc[i], lb.iloc[i]
        if va is None or vb is None or (isinstance(va, float) and np.isnan(va)):
            finals.append(None)
            reasons.append("missing_label")
            disputed.append(False)
            continue
        if int(va) == int(vb):
            finals.append(int(va))
            reasons.append("agree")
            disputed.append(False)
        else:
            finals.append(None)
            reasons.append("DISPUTE_needs_adjudicator")
            disputed.append(True)

    adj = pd.DataFrame(
        {
            "row_id": m["row_id"],
            "label_A": la.values,
            "label_B": lb.values,
            "final_label": finals,
            "reason": reasons,
        }
    )
    adj_path = f"{OUT_DIR}/adjudication_table.csv"
    adj.to_csv(adj_path, index=False)

    result = {
        "n": int(len(m)),
        "n_dual": int(both.sum()),
        "cohen_kappa": round(kappa, 4),
        "n_disputed": int(sum(disputed)),
        "action": (
            "STOP_escalate_faculty"
            if kappa > 0.95
            else ("revise_codebook" if kappa < 0.50 else "adjudicate_disputes_then_heldout_eval")
        ),
        "sha256_file_a": _sha256_file(file_a),
        "sha256_file_b": _sha256_file(file_b),
        "adjudication_table": adj_path,
    }
    with open(f"{OUT_DIR}/agreement_v2.json", "w") as f:
        json.dump(result, f, indent=2)

    # Partial gold where agreed
    gold = m[["row_id"]].copy()
    gold["human_label_perpetrator"] = finals
    # attach text from A
    if "comment_text" in a.columns:
        gold = gold.merge(a[["row_id", "comment_text", "platform"]], on="row_id", how="left")
    gold_path = f"{OUT_DIR}/gold_heldout_partial.csv"
    gold.to_csv(gold_path, index=False)

    print(json.dumps(result, indent=2))
    if kappa > 0.95:
        print("\n*** κ > 0.95 — STOP. Do not compute downstream metrics. Escalate with raw files. ***")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export-sample", action="store_true")
    p.add_argument("--adjudicate", action="store_true")
    p.add_argument("--file-a", default=None)
    p.add_argument("--file-b", default=None)
    args = p.parse_args()
    if args.export_sample:
        export_sample()
    elif args.adjudicate:
        if not args.file_a or not args.file_b:
            raise SystemExit("Need --file-a and --file-b")
        adjudicate(args.file_a, args.file_b)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
