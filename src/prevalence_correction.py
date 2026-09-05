"""
Prevalence correction for classifier-estimated harm rates (Faculty Step 2).

Rogan-Gladen estimator:
  true_rate = (observed_rate + specificity - 1) / (sensitivity + specificity - 1)

Combined CI: bootstrap over comment sample AND annotation-derived Se/Sp uncertainty.

Outputs:
  outputs/results/prevalence_corrected_rates.csv
  outputs/results/prevalence_correction_methodology.json

Run: PYTHONPATH=src python src/prevalence_correction.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from corpus_config import platform_analysis_mask


def rogan_gladen(observed_rate: float, sensitivity: float, specificity: float) -> float:
    denom = sensitivity + specificity - 1.0
    if denom <= 0:
        return float("nan")
    return float(np.clip((observed_rate + specificity - 1.0) / denom, 0.0, 1.0))


def load_corpus_validation_metrics() -> dict:
    """Prefer Protocol v2 held-out Se/Sp; fall back to June in-sample corpus_validation."""
    held = "outputs/results/heldout_vs_insample_metrics.json"
    if os.path.exists(held):
        with open(held) as f:
            data = json.load(f)
        ho = data.get("held_out_v2") or {}
        if ho.get("status") == "ready" and ho.get("metrics"):
            m = ho["metrics"]
            return {
                "sensitivity": m.get("sensitivity", m.get("recall")),
                "specificity": m.get("specificity"),
                "precision": m.get("precision"),
                "f1": m.get("f1"),
                "n_annotated": m.get("n"),
                "by_platform": ho.get("by_platform") or {},
                "source": "protocol_v2_heldout",
                "note": "Primary Se/Sp from v2 gold_heldout (Faculty July Step 6.3)",
            }
    path = "outputs/results/classifier_validation.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("corpus_validation") or {}


def _platform_metrics(corpus_val: dict, platform: str) -> tuple[float, float, str]:
    """Return (sensitivity, specificity, provenance)."""
    prov = (
        "protocol_v2_heldout"
        if corpus_val.get("source") == "protocol_v2_heldout"
        else "corpus_human_validation"
    )
    by_plat = corpus_val.get("by_platform", {})
    if platform in by_plat:
        p = by_plat[platform]
        se = p.get("sensitivity", p.get("recall"))
        sp = p.get("specificity")
        if se is not None and sp is not None:
            return float(se), float(sp), prov
    if corpus_val.get("sensitivity") and corpus_val.get("specificity"):
        return float(corpus_val["sensitivity"]), float(corpus_val["specificity"]), f"{prov}_pooled"
    # EDOS holdout fallback — clearly labeled interim only
    edos_path = "outputs/results/classifier_validation.json"
    if os.path.exists(edos_path):
        with open(edos_path) as f:
            ed = json.load(f).get("edos_holdout", {})
        if ed.get("sensitivity") and ed.get("specificity"):
            return float(ed["sensitivity"]), float(ed["specificity"]), "INTERIM_EDOS_holdout_NOT_corpus_specific"
    return 0.7912, 0.6991, "INTERIM_EDOS_default"


def combined_prevalence_bootstrap(
    perp_flags: np.ndarray,
    sensitivity: float,
    specificity: float,
    n_boot: int = 2000,
    se_sp_noise: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Bootstrap comment resampling + perturb Se/Sp within annotation uncertainty.
    se_sp_noise: std dev for Gaussian perturbation of Se/Sp (0 if gold fixed).
    """
    flags = perp_flags.astype(int)
    n = len(flags)
    if n == 0:
        return {"observed_rate": 0.0, "corrected_rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    observed = float(flags.mean())
    corrected_point = rogan_gladen(observed, sensitivity, specificity)

    corrected_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_obs = flags[idx].mean()
        se = np.clip(sensitivity + rng.normal(0, se_sp_noise), 0.05, 0.99)
        sp = np.clip(specificity + rng.normal(0, se_sp_noise), 0.05, 0.99)
        cr = rogan_gladen(boot_obs, se, sp)
        if not np.isnan(cr):
            corrected_samples.append(cr)

    if not corrected_samples:
        return {
            "observed_rate": round(observed, 4),
            "corrected_rate": round(corrected_point, 4) if not np.isnan(corrected_point) else None,
            "ci_low": None,
            "ci_high": None,
        }

    arr = np.array(corrected_samples)
    return {
        "observed_rate": round(observed, 4),
        "corrected_rate": round(float(np.nanmean(arr)), 4),
        "ci_low": round(float(np.percentile(arr, 2.5)), 4),
        "ci_high": round(float(np.percentile(arr, 97.5)), 4),
    }


def compute_prevalence_corrected(
    master_path: str = "data/processed/master_women_relevant.csv",
) -> pd.DataFrame:
    df = pd.read_csv(master_path, low_memory=False)
    df = df[platform_analysis_mask(df)].copy()

    corpus_val = load_corpus_validation_metrics()
    has_corpus = corpus_val.get("n_annotated", 0) > 0
    se_sp_noise = 0.0 if has_corpus else 0.08  # wider interim band without gold labels

    rows = []
    for plat, g in df.groupby("platform"):
        perp = (g.get("harm_role", "") == "perpetrator_attack").astype(int).values
        sens, spec, prov = _platform_metrics(corpus_val, plat)
        stats = combined_prevalence_bootstrap(perp, sens, spec, se_sp_noise=se_sp_noise)

        # Rogan-Gladen is uninformative when observed rate < estimated FPR (1−Sp):
        # corrected estimate hits the 0 floor. Flag explicitly; keep classifier rate primary.
        fpr = max(0.0, 1.0 - spec)
        rg_censored = bool(
            stats["observed_rate"] is not None
            and stats["observed_rate"] < fpr
            and (stats["corrected_rate"] is None or stats["corrected_rate"] <= 0.0)
        )
        status = "corpus_validated" if has_corpus else "INTERIM_until_human_annotation"
        if rg_censored:
            status = "corpus_validated_rg_censored_low_prevalence"
            # Report upper-bound style: corrected cannot exceed observed under Sp<1
            # when RG floors; keep corrected_rate at 0 and surface note in display.
            stats["corrected_rate"] = 0.0

        rows.append(
            {
                "platform": plat,
                "n_comments": len(g),
                "classifier_flagged_rate": stats["observed_rate"],
                "classifier_flagged_rate_pct": round(stats["observed_rate"] * 100, 1),
                "prevalence_corrected_rate": stats["corrected_rate"],
                "prevalence_corrected_pct": (
                    round(stats["corrected_rate"] * 100, 1)
                    if stats["corrected_rate"] is not None
                    else None
                ),
                "combined_ci_low": stats["ci_low"],
                "combined_ci_high": stats["ci_high"],
                "combined_ci_display": (
                    f"[{stats['ci_low']*100:.1f}–{stats['ci_high']*100:.1f}%]"
                    if stats["ci_low"] is not None
                    else "pending"
                ),
                "sensitivity_used": round(sens, 4),
                "specificity_used": round(spec, 4),
                "metrics_provenance": prov,
                "validation_status": status,
                "reporting_note": (
                    f"Rogan-Gladen censored: observed {stats['observed_rate']*100:.1f}% < FPR≈{fpr*100:.1f}% "
                    "(1−Sp); report classifier rate as primary for this platform"
                    if rg_censored
                    else ""
                ),
            }
        )

    out = pd.DataFrame(rows)
    os.makedirs("outputs/results", exist_ok=True)
    out.to_csv("outputs/results/prevalence_corrected_rates.csv", index=False)

    methodology = {
        "formula": "Rogan-Gladen: true_rate = (observed + specificity - 1) / (sensitivity + specificity - 1)",
        "combined_ci": "Bootstrap over comment resampling + Se/Sp perturbation",
        "reporting_rule": "Thesis must show BOTH classifier-flagged rate AND prevalence-corrected estimate with combined CI",
        "validation_status": "corpus_validated" if has_corpus else "INTERIM — complete Step 1 human annotation",
        "rows": out.to_dict(orient="records"),
    }
    with open("outputs/results/prevalence_correction_methodology.json", "w") as f:
        json.dump(methodology, f, indent=2)

    print("\nPREVALENCE-CORRECTED HARM RATES:")
    print(out[["platform", "classifier_flagged_rate_pct", "prevalence_corrected_pct", "combined_ci_display", "validation_status"]].to_string(index=False))
    return out


if __name__ == "__main__":
    compute_prevalence_corrected()
