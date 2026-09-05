"""
Merge bootstrap CIs + display labels into thesis score tables.

Run: python src/scoring_exports.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from corpus_config import platform_analysis_mask
from platform_labels import add_display_columns, display_name, platform_is_proxy


def bootstrap_whshi_ci(
    master_path: str = "data/processed/master_women_relevant.csv",
    fuzzy_by_platform: dict[str, float] | None = None,
    n_boot: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap WHSI_raw CI from row-level perpetrator/directed flags."""
    df = pd.read_csv(master_path, low_memory=False)
    df = df[platform_analysis_mask(df)].copy()

    rng = np.random.default_rng(seed)
    rows = []

    for plat, g in df.groupby("platform"):
        if "harm_role" not in g.columns:
            continue
        perp = (g["harm_role"] == "perpetrator_attack").astype(int).values
        directed = g.get("directed_at_women", pd.Series(0, index=g.index)).astype(int).values
        perp_mask = perp == 1
        n = len(g)
        fuzzy = (fuzzy_by_platform or {}).get(plat, 25.0)
        scores = []

        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_perp = perp[idx]
            boot_dir = directed[idx]
            p_rate = boot_perp.mean()
            if boot_perp.sum() > 0:
                d_share = boot_dir[boot_perp == 1].mean()
            else:
                d_share = 0.0
            wtshi = 100.0 * p_rate * (0.5 + 0.5 * d_share)
            scores.append(0.70 * wtshi + 0.30 * fuzzy)

        scores = np.array(scores)
        rows.append(
            {
                "platform": plat,
                "whsi_raw_ci_low": round(float(np.percentile(scores, 2.5)), 1),
                "whsi_raw_ci_high": round(float(np.percentile(scores, 97.5)), 1),
            }
        )

    return pd.DataFrame(rows)


def mri_uncertainty_bands(
    sensitivity_path: str = "outputs/results/mri_sensitivity.csv",
    mri_path: str = "data/processed/mri_scores.csv",
) -> pd.DataFrame:
    """Min/max MRI from sensitivity; extended to include actual reported MRI (Faculty Step 9)."""
    if not os.path.exists(sensitivity_path):
        return pd.DataFrame(columns=["platform", "mri_ci_low", "mri_ci_high"])

    sens = pd.read_csv(sensitivity_path)
    actual = pd.read_csv(mri_path).set_index("platform")["MRI_score"].to_dict() if os.path.exists(mri_path) else {}

    rows = []
    for plat, grp in sens.groupby("platform"):
        lo = float(grp["MRI_score"].min())
        hi = float(grp["MRI_score"].max())
        act = actual.get(plat)
        if act is not None:
            lo = min(lo, float(act))
            hi = max(hi, float(act))
        rows.append({
            "platform": plat,
            "mri_ci_low": round(lo, 1),
            "mri_ci_high": round(hi, 1),
        })
    return pd.DataFrame(rows)


def format_ci(low: float, high: float, decimals: int = 1) -> str:
    return f"[{low:.{decimals}f}–{high:.{decimals}f}]"


def enrich_whshi_scores(
    whsi_path: str = "data/processed/whsi_scores.csv",
    ci_path: str = "outputs/results/harm_rate_confidence_intervals.csv",
    fuzzy_path: str = "data/processed/whsi_scores_integrated.csv",
) -> pd.DataFrame:
    whsi = pd.read_csv(whsi_path)
    ci = pd.read_csv(ci_path) if os.path.exists(ci_path) else pd.DataFrame()

    fuzzy_map = {}
    if os.path.exists(fuzzy_path):
        full = pd.read_csv(fuzzy_path)
        if "fuzzy_component" in full.columns:
            fuzzy_map = full.set_index("platform")["fuzzy_component"].to_dict()

    drop_cols = [
        c
        for c in whsi.columns
        if c.endswith("_x")
        or c.endswith("_y")
        or c
        in {
            "whsi_raw_ci_low",
            "whsi_raw_ci_high",
            "perpetrator_ci_low",
            "perpetrator_ci_high",
            "women_targeted_perpetrator_rate",
            "women_targeted_ci_low",
            "women_targeted_ci_high",
            "is_proxy",
            "is_historical",
            "platform_display",
            "perpetrator_rate_pct",
            "perpetrator_rate_ci",
            "WHSI_score_display",
            "WHSI_literature_display",
            "WTSHI_display",
        }
    ]
    whsi = whsi.drop(columns=drop_cols, errors="ignore")

    whsi_ci = bootstrap_whshi_ci(fuzzy_by_platform=fuzzy_map)
    whsi = whsi.merge(whsi_ci, on="platform", how="left", suffixes=("", "_drop"))
    whsi = whsi[[c for c in whsi.columns if not c.endswith("_drop")]]
    if not ci.empty:
        ci_cols = [
            "platform",
            "perpetrator_ci_low",
            "perpetrator_ci_high",
            "is_proxy_data",
            "is_historical_data",
            "women_targeted_perpetrator_rate",
            "women_targeted_ci_low",
            "women_targeted_ci_high",
        ]
        ci_sub = ci[[c for c in ci_cols if c in ci.columns]].copy()
        overlap = [c for c in ci_sub.columns if c != "platform" and c in whsi.columns]
        if overlap:
            whsi = whsi.drop(columns=overlap, errors="ignore")
        whsi = whsi.merge(ci_sub, on="platform", how="left")
        if "is_historical_data" in whsi.columns:
            whsi["is_historical"] = whsi["is_historical_data"].fillna(0).astype(int)

    whsi = add_display_columns(whsi, proxy_col="is_proxy_data" if "is_proxy_data" in whsi.columns else "is_proxy")

    whsi["perpetrator_rate_pct"] = (whsi["perpetrator_harm_rate"] * 100).round(1)
    whsi["perpetrator_rate_ci"] = whsi.apply(
        lambda r: format_ci(r["perpetrator_ci_low"] * 100, r["perpetrator_ci_high"] * 100)
        if pd.notna(r.get("perpetrator_ci_low"))
        else "",
        axis=1,
    )
    whsi["WHSI_score_display"] = whsi.apply(
        lambda r: f"{r['WHSI_raw']:.1f} {format_ci(r['whsi_raw_ci_low'], r['whsi_raw_ci_high'])}"
        if pd.notna(r.get("whsi_raw_ci_low"))
        else f"{r['WHSI_raw']:.1f}",
        axis=1,
    )
    if "WHSI_literature_adjusted" in whsi.columns:
        whsi["WHSI_literature_display"] = whsi["WHSI_literature_adjusted"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else ""
        )
    whsi["WTSHI_display"] = whsi.apply(
        lambda r: f"{r['wtshi_construct']:.1f}",
        axis=1,
    )
    whsi["measurement_epistemology"] = (
        "classifier_output_unvalidated — perpetrator rates are model estimates, not human-validated harm rates"
    )
    whsi["perpetrator_rate_label"] = "classifier_estimated_perpetrator_share"

    # Prevalence-corrected rates (Faculty Step 2)
    prev_path = "outputs/results/prevalence_corrected_rates.csv"
    if os.path.exists(prev_path):
        prev = pd.read_csv(prev_path)
        whsi = whsi.merge(
            prev[
                [
                    "platform",
                    "prevalence_corrected_rate",
                    "prevalence_corrected_pct",
                    "combined_ci_display",
                    "validation_status",
                ]
            ],
            on="platform",
            how="left",
        )
        whsi["perpetrator_rate_reporting"] = whsi.apply(
            lambda r: (
                f"classifier {r['perpetrator_rate_pct']}% [{r.get('perpetrator_rate_ci', '')}]; "
                f"corrected {r.get('prevalence_corrected_pct', 'pending')}% {r.get('combined_ci_display', '')}"
                if pd.notna(r.get("prevalence_corrected_pct"))
                else f"classifier {r['perpetrator_rate_pct']}% — prevalence correction pending human validation"
            ),
            axis=1,
        )

    whsi.to_csv(whsi_path, index=False)
    return whsi


def enrich_safety_matrix(
    matrix_path: str = "outputs/results/safety_matrix_data.csv",
    whsi_path: str = "data/processed/whsi_scores.csv",
    mri_path: str = "data/processed/mri_scores.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    whsi = pd.read_csv(whsi_path)
    mri = pd.read_csv(mri_path)[["platform", "MRI_score", "MRI_label", "source"]]
    matrix = whsi.merge(mri, on="platform", how="inner")

    mri_bands = mri_uncertainty_bands()
    if not mri_bands.empty:
        matrix = matrix.drop(columns=["mri_ci_low", "mri_ci_high"], errors="ignore")
        matrix = matrix.merge(mri_bands, on="platform", how="left")

    if "quadrant" not in matrix.columns:
        from safety_matrix import assign_quadrant

        matrix["quadrant"] = matrix.apply(
            lambda r: assign_quadrant(r["WHSI_raw"], r["MRI_score"]), axis=1
        )

    matrix["quadrant_note"] = (
        "Descriptive coordinates only — WHSI_raw is classifier-estimated harm, not validated harm rate"
    )
    matrix["WHSI_display"] = matrix.get(
        "WHSI_score_display",
        matrix["WHSI_raw"].apply(lambda x: f"{x:.1f}"),
    )
    if "WHSI_literature_adjusted" in matrix.columns:
        matrix["WHSI_literature_display"] = matrix.get(
            "WHSI_literature_display",
            matrix["WHSI_literature_adjusted"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
        )
        matrix["risk_rank_literature"] = matrix["WHSI_literature_adjusted"] - matrix["MRI_score"]
    matrix["MRI_display"] = matrix.apply(
        lambda r: f"{r['MRI_score']:.1f} {format_ci(r['mri_ci_low'], r['mri_ci_high'])}"
        if pd.notna(r.get("mri_ci_low"))
        else f"{r['MRI_score']:.1f}",
        axis=1,
    )
    matrix["risk_rank"] = matrix["WHSI_raw"] - matrix["MRI_score"]

    keep = [c for c in matrix.columns if not (c.endswith("_x") or c.endswith("_y"))]
    matrix = matrix[keep]
    matrix.to_csv(matrix_path, index=False)
    live_primary = matrix[(matrix["is_proxy"] != 1) & (matrix.get("is_historical", 0) != 1)].copy()
    historical = matrix[matrix.get("is_historical", 0) == 1].copy()
    proxy = matrix[matrix["is_proxy"] == 1].copy()
    live_primary.to_csv("outputs/results/safety_matrix_primary.csv", index=False)
    historical.to_csv("outputs/results/safety_matrix_historical.csv", index=False)
    proxy.to_csv("outputs/results/safety_matrix_exploratory_proxy.csv", index=False)
    return live_primary, historical if not historical.empty else proxy


if __name__ == "__main__":
    enrich_whshi_scores()
    enrich_safety_matrix()
    print("Enriched whsi_scores.csv and safety_matrix_data.csv with CIs + display labels")
