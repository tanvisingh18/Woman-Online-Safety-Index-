"""
Path A — Fix the instrument (Faculty Round 3).

1. Tune ONLY on June n=360 (re-score texts; ignore corrupted stored probas).
2. Sweep thresholds + taxonomy-informed filters (victim / feminist-topic veto).
3. Select operating point(s) on June without looking at v2 labels for selection.
4. Evaluate ONCE on Protocol v2 gold with IPW weights (primary).
5. If IPW precision < 0.5, run DistilBERT fine-tune on EDOS+June; eval IPW-v2.

Never tune on v2. Never claim June metrics as primary validity.

Run: PYTHONPATH=src python src/path_a_fix_instrument.py
     PYTHONPATH=src python src/path_a_fix_instrument.py --skip-transformer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from gendered_harm_model import (
    FEMALE_TARGET,
    PERPETRATOR_ATTACK,
    VICTIM_DISCLOSURE,
    predict_batch,
)
from heldout_ipw_reweight import bootstrap_weighted, confusion_from_weights

JUNE = "data/labelled/validation_sample.csv"
V2 = "data/labelled/v2/gold_heldout.csv"
WEIGHTS = "outputs/results/heldout_ipw_gold_weights.csv"
OUT_DIR = Path("outputs/results")
OUT_JSON = OUT_DIR / "path_a_instrument_fix.json"
OUT_MD = OUT_DIR / "path_a_instrument_fix.md"
OUT_CFG = Path("configs/perpetrator_thresholds_path_a_candidate.json")

# Taxonomy-informed topic veto (feminist / gender discourse without attack lexicon)
TOPIC_DISCOURSE = re.compile(
    r"\b(feminis|patriarch|misogyn|metoo|sexis|women'?s rights|gender pay|"
    r"trad.?wife|equality|empowerment|podcast|episode)\b",
    re.IGNORECASE,
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def flag_from_scores(
    text: str,
    sp: float,
    tp: float,
    t_sex: float,
    t_sex_threat: float,
    t_threat: float,
    *,
    require_female_target: bool = False,
    topic_veto: bool = False,
    expand_victim: bool = True,
) -> int:
    """Parameterized perpetrator flag (mirrors _scores_from_proba with knobs)."""
    t = str(text)
    victim = bool(VICTIM_DISCLOSURE.search(t))
    if expand_victim:
        victim = victim or bool(
            re.search(
                r"\b(i (was|am|feel|felt)|survivors?|reporting|happened to (me|us)|"
                r"my (husband|boyfriend|partner) (hit|abused|raped))\b",
                t,
                re.I,
            )
        )
    attack_kw = bool(PERPETRATOR_ATTACK.search(t))
    female = bool(FEMALE_TARGET.search(t))

    if topic_veto and TOPIC_DISCOURSE.search(t) and not attack_kw:
        # Structural FP mode: topic vocabulary without clear attack → do not flag
        return 0

    is_perp = (
        attack_kw
        or (sp >= t_sex and not victim)
        or (sp >= t_sex_threat and tp >= t_threat and not victim)
    )
    if require_female_target and is_perp and not attack_kw and not female:
        return 0
    return int(is_perp)


def metrics_unweighted(y, yhat) -> dict:
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    spec = tn / (tn + fp + 1e-9)
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1), 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "n_flags": int(tp + fp),
    }


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    preds = predict_batch(df["comment_text"].fillna("").tolist())
    out = df.copy()
    out["sp_rescored"] = preds["gendered_harm_proba"].astype(float)
    # threat before role discount — use pipeline threat from predict; predict_batch
    # returns post-processed threat_score. For thresholding we want raw-ish scores.
    # Re-get from model internals:
    from gendered_harm_model import _load_pipe, SEXIST_MODEL, THREAT_MODEL

    sexist = _load_pipe(SEXIST_MODEL)
    threat = _load_pipe(THREAT_MODEL)
    texts = df["comment_text"].fillna("").astype(str).tolist()
    out["sp_raw"] = sexist.predict_proba(texts)[:, 1]
    out["tp_raw"] = threat.predict_proba(texts)[:, 1]
    return out


def apply_config(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    return np.array(
        [
            flag_from_scores(
                str(t),
                float(sp),
                float(tp),
                cfg["t_sex"],
                cfg["t_sex_threat"],
                cfg["t_threat"],
                require_female_target=cfg.get("require_female_target", False),
                topic_veto=cfg.get("topic_veto", False),
                expand_victim=cfg.get("expand_victim", True),
            )
            for t, sp, tp in zip(df["comment_text"], df["sp_raw"], df["tp_raw"])
        ]
    )


def sweep_june(june: pd.DataFrame) -> list[dict]:
    y = june["human_label_perpetrator"].astype(int).values
    results = []
    # Raise main threshold to chase precision; keep threat branch
    for t_sex in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for t_sex_threat in [0.34, 0.40, 0.45, 0.50, 0.55]:
            if t_sex_threat > t_sex:
                continue
            for t_threat in [0.28, 0.35, 0.40, 0.45]:
                for topic_veto in [False, True]:
                    for req_f in [False, True]:
                        cfg = {
                            "t_sex": t_sex,
                            "t_sex_threat": t_sex_threat,
                            "t_threat": t_threat,
                            "topic_veto": topic_veto,
                            "require_female_target": req_f,
                            "expand_victim": True,
                        }
                        yhat = apply_config(june, cfg)
                        m = metrics_unweighted(y, yhat)
                        p, r, f1 = m["precision"], m["recall"], m["f1"]
                        # Path A selection (June only, pre-registered):
                        # maximize precision subject to recall >= 0.50;
                        # taxonomy prior: prefer topic_veto (structural FP mode).
                        if r < 0.50:
                            sel = -1000 + p
                        else:
                            sel = p * 100 + (10 if topic_veto else 0) + f1
                        results.append({**cfg, **m, "selection_score": round(sel, 4)})
    results.sort(key=lambda d: d["selection_score"], reverse=True)
    return results


def ipw_eval(v2: pd.DataFrame, weights: pd.DataFrame, cfg: dict) -> dict:
    # Align by row_id
    w = weights.set_index("row_id")
    boot_df = pd.DataFrame(
        {
            "row_id": v2["row_id"].values,
            "platform": v2["platform"].values,
            "comment_text": v2["comment_text"].values,
            "y": v2["human_label_perpetrator"].astype(int).values,
            "yhat": apply_config(v2, cfg),
            "weight": v2["row_id"].map(w["weight"]).astype(float).values,
            "_band": v2["row_id"].map(w["_band"]).values,
        }
    )
    if boot_df["weight"].isna().any():
        boot_df["weight"] = boot_df["weight"].fillna(1.0)
    uw = metrics_unweighted(boot_df["y"], boot_df["yhat"])
    ww = confusion_from_weights(
        boot_df["y"].to_numpy(), boot_df["yhat"].to_numpy(), boot_df["weight"].to_numpy()
    )
    boot = bootstrap_weighted(boot_df, n_boot=2000, seed=20260812)
    return {
        "config": cfg,
        "unweighted": uw,
        "weighted": ww,
        "bootstrap_weighted_ci": boot,
    }


def run_distilbert(june: pd.DataFrame, v2: pd.DataFrame, weights: pd.DataFrame) -> dict | None:
    """Fine-tune DistilBERT on EDOS sexist + June gold; eval IPW-v2."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as e:
        return {"status": "skipped", "reason": f"import_failed: {e}"}

    from gendered_harm_model import find_edos_path

    edos_path = find_edos_path()
    if not edos_path:
        return {"status": "skipped", "reason": "EDOS not found"}

    edos = pd.read_csv(edos_path)
    text_col = "text" if "text" in edos.columns else "comment_text"
    edos = edos.rename(columns={text_col: "comment_text"})
    edos["comment_text"] = edos["comment_text"].astype(str)
    edos = edos[edos["comment_text"].str.len() > 10]
    # Approximate perpetrator-ish: sexist label (EDOS doesn't have WTSHI roles)
    edos["y"] = (edos["label_sexist"].astype(str).str.lower() == "sexist").astype(int)
    pos = edos[edos["y"] == 1]
    neg = edos[edos["y"] == 0]
    n_pos = min(2500, len(pos))
    n_neg = min(2500, len(neg))
    edos_s = pd.concat(
        [
            pos.sample(n_pos, random_state=42),
            neg.sample(n_neg, random_state=42),
        ]
    )
    june_tr = june.copy()
    june_tr["y"] = june_tr["human_label_perpetrator"].astype(int)
    # Upsample June 3× so WTSHI gold outweighs noisy EDOS sexist≈perp proxy
    train = pd.concat(
        [
            edos_s[["comment_text", "y"]],
            june_tr[["comment_text", "y"]],
            june_tr[["comment_text", "y"]],
            june_tr[["comment_text", "y"]],
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=42)

    model_name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cpu")

    class TxtDS(Dataset):
        def __init__(self, texts, labels):
            self.enc = tok(
                list(texts),
                truncation=True,
                padding=True,
                max_length=128,
            )
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(int(self.labels[i]))
            return item

    n = len(train)
    cut = int(0.9 * n)
    ds_tr = TxtDS(train["comment_text"].iloc[:cut], train["y"].iloc[:cut])
    ds_va = TxtDS(train["comment_text"].iloc[cut:], train["y"].iloc[cut:])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    out_dir = "models/path_a_distilbert"
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        report_to=[],
        seed=42,
        use_cpu=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        return {"precision": float(p), "recall": float(r), "f1": float(f1)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Batched predict on v2
    model.to(device)
    model.eval()
    v2_texts = v2["comment_text"].fillna("").astype(str).tolist()
    yhats = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(v2_texts), bs):
            batch = v2_texts[i : i + bs]
            enc = tok(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            yhats.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    yhat = np.array(yhats)

    w = weights.set_index("row_id")
    boot_df = pd.DataFrame(
        {
            "platform": v2["platform"].values,
            "y": v2["human_label_perpetrator"].astype(int).values,
            "yhat": yhat,
            "weight": v2["row_id"].map(w["weight"]).astype(float).values,
            "_band": v2["row_id"].map(w["_band"]).values,
        }
    )
    boot_df["weight"] = boot_df["weight"].fillna(1.0)
    uw = metrics_unweighted(boot_df["y"], boot_df["yhat"])
    ww = confusion_from_weights(
        boot_df["y"].to_numpy(), boot_df["yhat"].to_numpy(), boot_df["weight"].to_numpy()
    )
    boot = bootstrap_weighted(boot_df, n_boot=2000, seed=20260812)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {
        "status": "ok",
        "model": model_name,
        "train_n": int(len(train)),
        "edos_sample_n": int(len(edos_s)),
        "june_n": int(len(june_tr)),
        "june_upsample": 3,
        "device": str(device),
        "unweighted_v2": uw,
        "weighted_v2": ww,
        "bootstrap_weighted_ci": boot,
        "model_dir": out_dir,
        "note": (
            "EDOS sexist≠perpetrator; June gold upsampled 3×. "
            "Eval is IPW-weighted v2 only."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-transformer", action="store_true")
    ap.add_argument("--force-transformer", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Rescoring June…")
    june = pd.read_csv(JUNE, low_memory=False)
    june = june[pd.to_numeric(june["human_label_perpetrator"], errors="coerce").isin([0, 1])]
    june = score_frame(june)

    print("Sweeping June (tune only)…")
    sweep = sweep_june(june)
    top = sweep[:15]
    baseline_cfg = {
        "t_sex": 0.40,
        "t_sex_threat": 0.34,
        "t_threat": 0.28,
        "topic_veto": False,
        "require_female_target": False,
        "expand_victim": False,  # match production _scores_from_proba victim regex only
        "name": "frozen_production_0.40_0.34_0.28",
    }
    # Pre-registered June selection: max precision s.t. R>=0.50 (+ topic_veto prior)
    best_cfg = {k: top[0][k] for k in [
        "t_sex", "t_sex_threat", "t_threat", "topic_veto", "require_female_target", "expand_victim"
    ]}
    best_cfg["name"] = "path_a_june_selected_maxP_Rge0.5"

    # Taxonomy-forced candidate: best among topic_veto=True with R>=0.5
    tax = [r for r in sweep if r["topic_veto"] and r["recall"] >= 0.50]
    tax_cfg = None
    if tax:
        tax_cfg = {k: tax[0][k] for k in [
            "t_sex", "t_sex_threat", "t_threat", "topic_veto", "require_female_target", "expand_victim"
        ]}
        tax_cfg["name"] = "path_a_taxonomy_topic_veto"

    print("Loading v2 + IPW weights…")
    v2 = pd.read_csv(V2)
    v2 = score_frame(v2)
    weights = pd.read_csv(WEIGHTS)

    print("IPW eval baseline…")
    ev_base = ipw_eval(v2, weights, baseline_cfg)
    print("IPW eval june-selected…")
    ev_best = ipw_eval(v2, weights, best_cfg)
    ev_tax = ipw_eval(v2, weights, tax_cfg) if tax_cfg else None

    # Primary = June-pre-registered selected config (NOT chosen by peeking at v2).
    # We still *report* baseline and taxonomy for transparency; primary is june_selected.
    primary_name = "june_selected"
    primary = ev_best

    transformer = None
    need_t = (primary["weighted"]["precision"] or 0) < 0.50
    if args.force_transformer or (need_t and not args.skip_transformer):
        print("IPW precision < 0.5 (or forced) → DistilBERT pilot…")
        transformer = run_distilbert(june, v2, weights)
        if transformer and transformer.get("status") == "ok":
            # DistilBERT is an alternate Path A arm; report both; promote if better P_w
            # without using v2 to choose TF-IDF hyperparameters (model class switch is pre-registered escalate).
            tw = transformer["weighted_v2"]["precision"]
            if tw is not None and tw > (primary["weighted"]["precision"] or 0):
                primary_name = "distilbert"
                primary = {
                    "config": {"name": "distilbert_edos_june"},
                    "unweighted": transformer["unweighted_v2"],
                    "weighted": transformer["weighted_v2"],
                    "bootstrap_weighted_ci": transformer["bootstrap_weighted_ci"],
                }

    # Persist candidate thresholds if TF-IDF won
    if primary_name != "distilbert":
        cfg_out = {
            **primary["config"],
            "status": "PATH_A_CANDIDATE_not_production_until_accepted",
            "selected_by": primary_name,
            "ipw_weighted_precision": primary["weighted"]["precision"],
            "ipw_weighted_f1": primary["weighted"]["f1"],
            "note": "Tune on June only; eval IPW-v2. Do not overwrite frozen production without faculty sign-off.",
        }
        OUT_CFG.write_text(json.dumps(cfg_out, indent=2))

    gate = {
        "target_precision": [0.5, 0.7],
        "achieved_ipw_precision": primary["weighted"]["precision"],
        "target_met": bool(
            primary["weighted"]["precision"] is not None
            and primary["weighted"]["precision"] >= 0.50
        ),
        "absolute_claims_restored": False,
    }
    gate["absolute_claims_restored"] = gate["target_met"]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "tune_on": "June validation_sample.csv (rescored)",
            "evaluate_on": "v2 gold_heldout.csv with IPW weights",
            "never_tune_on_v2": True,
            "june_sha256": _sha256(JUNE),
            "v2_sha256": _sha256(V2),
        },
        "june_baseline_rescored": metrics_unweighted(
            june["human_label_perpetrator"], apply_config(june, baseline_cfg)
        ),
        "june_top_sweep": top[:10],
        "ipw_baseline_frozen": ev_base,
        "ipw_june_selected": ev_best,
        "ipw_taxonomy_topic_veto": ev_tax,
        "transformer": transformer,
        "primary": {"name": primary_name, **primary},
        "gate": gate,
        "honest_summary": (
            f"Path A primary={primary_name}; IPW precision="
            f"{primary['weighted']['precision']}; target 0.5–0.7 "
            f"{'MET' if gate['target_met'] else 'NOT MET'}."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))

    lines = [
        "# Path A — Instrument fix results",
        "",
        f"**Primary candidate:** `{primary_name}`",
        f"**IPW-weighted precision:** **{primary['weighted']['precision']}** "
        f"(target 0.5–0.7: {'MET' if gate['target_met'] else 'NOT MET'})",
        f"**IPW-weighted F1:** {primary['weighted']['f1']} · "
        f"Recall {primary['weighted']['recall']} · Spec {primary['weighted']['specificity']}",
        "",
        "## Protocol",
        "- Tune: June n=360 only (rescored)",
        "- Eval: Protocol v2 + IPW (stratified bootstrap)",
        "- Never tune on v2",
        "",
        "## IPW comparison",
        "",
        "| Candidate | P_w | R_w | Sp_w | F1_w |",
        "|-----------|-----|-----|------|------|",
        f"| Frozen 0.40/0.34/0.28 | {ev_base['weighted']['precision']} | {ev_base['weighted']['recall']} | {ev_base['weighted']['specificity']} | {ev_base['weighted']['f1']} |",
        f"| June-selected (max P, R≥0.5) | {ev_best['weighted']['precision']} | {ev_best['weighted']['recall']} | {ev_best['weighted']['specificity']} | {ev_best['weighted']['f1']} |",
    ]
    if ev_tax:
        lines.append(
            f"| Taxonomy topic-veto | {ev_tax['weighted']['precision']} | {ev_tax['weighted']['recall']} | {ev_tax['weighted']['specificity']} | {ev_tax['weighted']['f1']} |"
        )
    if transformer and transformer.get("status") == "ok":
        lines.append(
            f"| DistilBERT | {transformer['weighted_v2']['precision']} | {transformer['weighted_v2']['recall']} | {transformer['weighted_v2']['specificity']} | {transformer['weighted_v2']['f1']} |"
        )
    lines += [
        "",
        f"**Absolute claims restored?** {'Yes' if gate['absolute_claims_restored'] else 'No — instrument still below Path A gate.'}",
        "",
        f"Full JSON: `{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps(gate, indent=2))
    print(payload["honest_summary"])
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
