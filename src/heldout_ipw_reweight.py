"""
Faculty August Round 3 Step 1 — Inverse-probability weighting for Protocol v2 gold.

Band definitions MUST match src/annotation_protocol_v2.py (seed 20260712):
  high: gendered_harm_proba >= 0.55
  mid:  0.35 <= proba < 0.55
  low:  proba < 0.35

Weight for gold row i in platform p, band b:
  w_i = N[p][b] / n[p][b]

Run: PYTHONPATH=src python src/heldout_ipw_reweight.py
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from annotation_protocol_v2 import SEED, JUNE_PATH, MASTER, STRATA, _band, _text_key
from gendered_harm_model import (
    PERP_SEXIST_THRESH,
    PERP_SEXIST_WITH_THREAT,
    PERP_THREAT_MIN,
    _scores_from_proba,
)
from prevalence_correction import rogan_gladen

OUT = "outputs/results"
GOLD = "data/labelled/v2/gold_heldout.csv"
SAMPLE_MASTER = "data/labelled/v2/annotation_sample_master.csv"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def reconstruct_sampling_pool() -> pd.DataFrame:
    """Same pool construction as annotation_protocol_v2.export_sample."""
    master = pd.read_csv(MASTER, low_memory=False)
    june_texts = set()
    if os.path.exists(JUNE_PATH):
        june = pd.read_csv(JUNE_PATH, low_memory=False)
        june_texts = set(june["comment_text"].map(_text_key))

    live = master[master["dataset_split"] == "live_scrape"].copy()
    tw = master[master["dataset_split"] == "historical_twitter"].copy()
    pool = pd.concat([live, tw], ignore_index=True)
    pool = pool[pool["comment_text"].astype(str).str.len() > 20].copy()
    pool["_tkey"] = pool["comment_text"].map(_text_key)
    pool = pool[~pool["_tkey"].isin(june_texts)].copy()
    pool["gendered_harm_proba"] = pd.to_numeric(
        pool.get("gendered_harm_proba"), errors="coerce"
    ).fillna(0)
    pool["_band"] = pool["gendered_harm_proba"].map(_band)
    return pool


def band_count_tables(pool: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for plat in sorted(set(pool["platform"].unique()) | set(sample["platform"].unique())):
        for band in ("low", "mid", "high"):
            N = int(((pool["platform"] == plat) & (pool["_band"] == band)).sum())
            n = int(((sample["platform"] == plat) & (sample["_band"] == band)).sum())
            w = (N / n) if n > 0 else None
            rows.append(
                {
                    "platform": plat,
                    "band": band,
                    "band_definition": {
                        "low": "gendered_harm_proba < 0.35",
                        "mid": "0.35 <= gendered_harm_proba < 0.55",
                        "high": "gendered_harm_proba >= 0.55",
                    }[band],
                    "corpus_N": N,
                    "sample_n": n,
                    "weight_N_over_n": round(w, 6) if w is not None else None,
                }
            )
    return pd.DataFrame(rows)


def classifier_flag(row) -> int:
    text = str(row.get("comment_text", ""))
    sp = float(row.get("gendered_harm_proba", 0) or 0)
    tp = float(row.get("threat_score", 0) or 0)
    out = _scores_from_proba(text, sp, tp)
    return 1 if out["harm_role"] == "perpetrator_attack" else 0


def confusion_from_weights(y, yhat, w) -> dict:
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    w = np.asarray(w).astype(float)
    tp = float(w[(y == 1) & (yhat == 1)].sum())
    fp = float(w[(y == 0) & (yhat == 1)].sum())
    fn = float(w[(y == 1) & (yhat == 0)].sum())
    tn = float(w[(y == 0) & (yhat == 0)].sum())
    se = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    sp = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = se
    f1 = (
        2 * prec * rec / (prec + rec)
        if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0
        else float("nan")
    )
    flag_rate = (tp + fp) / w.sum() if w.sum() > 0 else float("nan")
    return {
        "TP_w": round(tp, 4),
        "FP_w": round(fp, 4),
        "FN_w": round(fn, 4),
        "TN_w": round(tn, 4),
        "sensitivity": round(float(se), 4) if np.isfinite(se) else None,
        "specificity": round(float(sp), 4) if np.isfinite(sp) else None,
        "precision": round(float(prec), 4) if np.isfinite(prec) else None,
        "recall": round(float(rec), 4) if np.isfinite(rec) else None,
        "f1": round(float(f1), 4) if np.isfinite(f1) else None,
        "weighted_flag_rate": round(float(flag_rate), 4) if np.isfinite(flag_rate) else None,
        "sum_weights": round(float(w.sum()), 4),
        "n_rows": int(len(y)),
    }


def metrics_unweighted(y, yhat) -> dict:
    return confusion_from_weights(y, yhat, np.ones(len(y)))


def bootstrap_weighted(
    df: pd.DataFrame, n_boot: int = 2000, seed: int = 20260804
) -> dict:
    """Resample within platform×band strata (design-respecting)."""
    rng = np.random.default_rng(seed)
    keys = ["precision", "recall", "sensitivity", "specificity", "f1", "weighted_flag_rate"]
    store = {k: [] for k in keys}
    groups = list(df.groupby(["platform", "_band"], sort=False))
    for _ in range(n_boot):
        parts = []
        for (_, _), g in groups:
            if len(g) == 0:
                continue
            idx = rng.choice(g.index.to_numpy(), size=len(g), replace=True)
            parts.append(df.loc[idx])
        boot = pd.concat(parts)
        m = confusion_from_weights(
            boot["y"].values, boot["yhat"].values, boot["weight"].values
        )
        for k in keys:
            store[k].append(m[k] if m[k] is not None else np.nan)
    out = {}
    for k, vals in store.items():
        arr = np.array(vals, dtype=float)
        out[k] = {
            "mean": round(float(np.nanmean(arr)), 4),
            "ci_low": round(float(np.nanpercentile(arr, 2.5)), 4),
            "ci_high": round(float(np.nanpercentile(arr, 97.5)), 4),
        }
    return out


def corpus_flag_rate(pool: pd.DataFrame, platform: str | None = None) -> float:
    sub = pool if platform is None else pool[pool["platform"] == platform]
    if sub.empty:
        return float("nan")
    flags = []
    for _, row in sub.iterrows():
        flags.append(classifier_flag(row))
    return float(np.mean(flags)) if flags else float("nan")


def run() -> dict:
    os.makedirs(OUT, exist_ok=True)
    pool = reconstruct_sampling_pool()
    sample = pd.read_csv(SAMPLE_MASTER, low_memory=False)
    sample["gendered_harm_proba"] = pd.to_numeric(
        sample["gendered_harm_proba"], errors="coerce"
    ).fillna(0)
    sample["_band"] = sample["gendered_harm_proba"].map(_band)

    band_table = band_count_tables(pool, sample)
    band_table.to_csv(f"{OUT}/heldout_ipw_band_table.csv", index=False)

    gold = pd.read_csv(GOLD, low_memory=False)
    if "gendered_harm_proba" not in gold.columns or gold["gendered_harm_proba"].isna().all():
        gold = gold.drop(columns=["gendered_harm_proba", "threat_score"], errors="ignore")
        gold = gold.merge(
            sample[
                [
                    "row_id",
                    "gendered_harm_proba",
                    "threat_score",
                    "comment_text",
                    "platform",
                ]
            ],
            on="row_id",
            how="left",
            suffixes=("", "_s"),
        )
        if "comment_text" not in gold.columns or gold["comment_text"].isna().all():
            gold["comment_text"] = gold.get("comment_text_s")

    gold["gendered_harm_proba"] = pd.to_numeric(
        gold["gendered_harm_proba"], errors="coerce"
    ).fillna(0)
    gold["threat_score"] = pd.to_numeric(gold.get("threat_score"), errors="coerce").fillna(0)
    gold["_band"] = gold["gendered_harm_proba"].map(_band)
    gold["y"] = pd.to_numeric(gold["human_label_perpetrator"], errors="coerce").astype(int)
    gold["yhat"] = gold.apply(classifier_flag, axis=1)

    # map weights
    wmap = {
        (r.platform, r.band): r.weight_N_over_n
        for r in band_table.itertuples()
        if r.weight_N_over_n is not None
    }
    gold["weight"] = gold.apply(lambda r: wmap[(r.platform, r._band)], axis=1)

    # overall + per platform
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling_seed": SEED,
        "band_definitions": {
            "source": "src/annotation_protocol_v2.py::_band",
            "high": "gendered_harm_proba >= 0.55",
            "mid": "0.35 <= gendered_harm_proba < 0.55",
            "low": "gendered_harm_proba < 0.35",
            "note": (
                "Appendix A illustrative cutpoints (0.34/0.40) are NOT used; "
                "weights follow the actual Protocol v2 sampling script."
            ),
        },
        "protocol_omission_note": (
            "Protocol v2.0 specified confidence-band stratification for IAA but omitted "
            "inverse-probability reweighting for corpus metric estimation. Faculty Round 3 "
            "identified the omission; this worksheet corrects it."
        ),
        "overall": {},
        "by_platform": {},
        "rogan_gladen": {},
        "files": {
            "band_table": f"{OUT}/heldout_ipw_band_table.csv",
            "gold_with_weights": f"{OUT}/heldout_ipw_gold_weights.csv",
        },
        "sha256_inputs": {
            "gold_heldout.csv": _sha256(GOLD),
            "annotation_sample_master.csv": _sha256(SAMPLE_MASTER),
            "this_script": _sha256(__file__) if os.path.exists(__file__) else None,
        },
    }

    # unweighted / weighted overall
    uw = metrics_unweighted(gold["y"], gold["yhat"])
    ww = confusion_from_weights(gold["y"], gold["yhat"], gold["weight"])
    # sanity: compare weighted flag rate to pool overall flag rate (live+twitter pool)
    # For overall, restrict sanity to live platforms YT/Reddit/Telegram if possible
    live_platforms = ["YouTube", "Reddit", "Telegram"]
    corpus_flag_overall = {}
    for plat in live_platforms + ["Twitter"]:
        corpus_flag_overall[plat] = round(corpus_flag_rate(pool, plat), 4)

    boot = bootstrap_weighted(gold)
    results["overall"] = {
        "unweighted": uw,
        "weighted": ww,
        "bootstrap_weighted_ci": boot,
        "sample_flag_rate_unweighted": round(float(gold["yhat"].mean()), 4),
        "explanation_paragraph": (
            "Protocol v2 stratified the gold set by classifier-confidence band so that "
            "ambiguous mid/high-score cases appear in inter-annotator agreement. That design "
            "over-represents high-score rows relative to the corpus, which inflates the "
            "apparent false-positive rate. Inverse-probability weights w = N_band/n_band "
            "restore each row to the mass of corpus comments it represents. Weighted Se/Sp/"
            "precision/F1 are therefore the corpus estimates; unweighted metrics remain "
            "valid descriptions of the stratified sample only (and of IAA design), not of "
            "platform-wide classifier performance."
        ),
    }

    for plat, g in gold.groupby("platform"):
        uw_p = metrics_unweighted(g["y"], g["yhat"])
        ww_p = confusion_from_weights(g["y"], g["yhat"], g["weight"])
        boot_p = bootstrap_weighted(g) if len(g) >= 8 else {}
        corp = corpus_flag_overall.get(plat)
        sanity_ok = (
            corp is not None
            and ww_p["weighted_flag_rate"] is not None
            and abs(ww_p["weighted_flag_rate"] - corp) < 0.08
        )
        results["by_platform"][plat] = {
            "unweighted": uw_p,
            "weighted": ww_p,
            "bootstrap_weighted_ci": boot_p,
            "corpus_flag_rate_on_sampling_pool": corp,
            "sanity_check_weighted_flag_near_corpus": sanity_ok,
            "sanity_gap": (
                round(abs(ww_p["weighted_flag_rate"] - corp), 4)
                if corp is not None and ww_p["weighted_flag_rate"] is not None
                else None
            ),
        }

        # Rogan–Gladen on live platforms using observed live scrape rates
        if plat in live_platforms and ww_p["sensitivity"] and ww_p["specificity"]:
            # observed rate from live women-relevant panel (not sampling pool with twitter)
            live_master = pd.read_csv(MASTER, low_memory=False)
            live_plat = live_master[
                (live_master["dataset_split"] == "live_scrape")
                & (live_master["platform"] == plat)
            ]
            # use existing perp flags if present
            if "harm_role" in live_plat.columns:
                obs = float((live_plat["harm_role"] == "perpetrator_attack").mean())
            else:
                obs = corp
            se_w, sp_w = ww_p["sensitivity"], ww_p["specificity"]
            fpr = 1.0 - sp_w
            if (se_w + sp_w - 1) <= 1e-9:
                true_r = None
                status = "undefined_youden_Se_plus_Sp_le_1"
            elif obs > fpr:
                true_r = rogan_gladen(obs, se_w, sp_w)
                status = "corrected"
            else:
                true_r = None
                status = "censored_observed_le_implied_fpr"
            results["rogan_gladen"][plat] = {
                "observed_corpus_flag_rate": round(obs, 4),
                "Se_w": se_w,
                "Sp_w": sp_w,
                "implied_FPR_1_minus_Sp_w": round(fpr, 4),
                "status": status,
                "prevalence_corrected_rate": (
                    round(true_r, 4) if true_r is not None else None
                ),
            }

    gold[
        [
            "row_id",
            "platform",
            "_band",
            "y",
            "yhat",
            "weight",
            "gendered_harm_proba",
        ]
    ].to_csv(f"{OUT}/heldout_ipw_gold_weights.csv", index=False)

    out_path = f"{OUT}/heldout_ipw_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # human-readable worksheet
    lines = [
        "# Held-out IPW worksheet (Faculty Round 3 Step 1)",
        "",
        f"Created: {results['created_at']}",
        f"Sampling seed: {SEED}",
        "",
        "## Band definitions (from sampling script, NOT appendix illustration)",
        "- low: gendered_harm_proba < 0.35",
        "- mid: 0.35 ≤ proba < 0.55",
        "- high: proba ≥ 0.55",
        "",
        results["protocol_omission_note"],
        "",
        "## Band table",
        band_table.to_string(index=False),
        "",
        "## Overall metrics",
        f"Unweighted: {json.dumps(uw)}",
        f"Weighted:   {json.dumps(ww)}",
        f"Bootstrap CI: {json.dumps(boot)}",
        "",
        results["overall"]["explanation_paragraph"],
        "",
        "## Per-platform sanity (weighted flag ≈ corpus flag)",
    ]
    for plat, d in results["by_platform"].items():
        lines.append(
            f"- {plat}: weighted_flag={d['weighted']['weighted_flag_rate']} "
            f"corpus={d['corpus_flag_rate_on_sampling_pool']} "
            f"ok={d['sanity_check_weighted_flag_near_corpus']} "
            f"Sp_w={d['weighted']['specificity']} F1_w={d['weighted']['f1']}"
        )
    lines += ["", "## Rogan–Gladen (weighted Se/Sp)"]
    for plat, d in results["rogan_gladen"].items():
        lines.append(f"- {plat}: {json.dumps(d)}")

    with open(f"{OUT}/heldout_ipw_worksheet.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved → {out_path}")
    print(f"Band table → {OUT}/heldout_ipw_band_table.csv")
    print(f"Worksheet → {OUT}/heldout_ipw_worksheet.md")
    print("\nOVERALL unweighted F1", uw["f1"], "Sp", uw["specificity"])
    print("OVERALL weighted   F1", ww["f1"], "Sp", ww["specificity"], "flag", ww["weighted_flag_rate"])
    for plat, d in results["by_platform"].items():
        print(
            plat,
            "Sp_w",
            d["weighted"]["specificity"],
            "P_w",
            d["weighted"]["precision"],
            "sanity",
            d["sanity_check_weighted_flag_near_corpus"],
            "gap",
            d["sanity_gap"],
        )
    print("\nRG:", json.dumps(results["rogan_gladen"], indent=2))
    return results


if __name__ == "__main__":
    run()
