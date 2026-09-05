"""
Integrated WHSI/MRI — folds hotspots, threads, sentiment/influence INTO platform scores.

Novelty layers merged into final WHSI (not separate side outputs):
  - Base T/Th/F/N from feature_extraction
  - Hotspot severity (top community WHSI per platform)
  - Reply-thread pile-on factor (YouTube/Reddit nested comments)
  - Sentiment + incitement influence weighting
  - Gendered harm prevalence calibration

Output:
  data/processed/platform_features_integrated.csv
  data/processed/whsi_scores_integrated.csv
  data/processed/mri_scores_integrated.csv

Run: python src/integrated_scoring.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from corpus_config import platform_analysis_mask
from feature_extraction import extract_platform_features, _harm_column
from fuzzy_engine import compute_whsi, run_whsi_for_all_platforms
from hotspot_detection import compute_community_whsi, _hotspot_corpus


def _live_corpus(master: pd.DataFrame) -> pd.DataFrame:
    return master[platform_analysis_mask(master)].copy()


def hotspot_platform_summary(community_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate top community WHSI into platform-level hotspot severity."""
    if community_df.empty:
        return pd.DataFrame(columns=["platform", "hotspot_whsi_mean", "hotspot_whsi_max", "n_hotspot_communities"])

    rows = []
    for plat, grp in community_df.groupby("platform"):
        top = grp.nlargest(min(5, len(grp)), "WHSI")
        rows.append(
            {
                "platform": plat,
                "hotspot_whsi_mean": round(float(top["WHSI"].mean()), 2),
                "hotspot_whsi_max": round(float(top["WHSI"].max()), 2),
                "n_hotspot_communities": int(len(grp)),
                "top_community": str(top.iloc[0]["community"]),
                "top_community_whsi": round(float(top.iloc[0]["WHSI"]), 2),
            }
        )
    return pd.DataFrame(rows)


def thread_reply_factors(master: pd.DataFrame) -> pd.DataFrame:
    """Reply-thread pile-on: harmful nested comments amplify frequency/threat."""
    live = _live_corpus(master)
    rows = []
    for plat, g in live.groupby("platform"):
        depth = g.get("thread_depth", pd.Series(0, index=g.index)).fillna(0).astype(float)
        harm_col = _harm_column(g)
        replies = g[depth > 0]
        top_level = g[depth <= 0]
        reply_harm = float(replies[harm_col].mean()) if len(replies) else 0.0
        top_harm = float(top_level[harm_col].mean()) if len(top_level) else 0.0
        reply_share = len(replies) / max(len(g), 1)
        pileon_score = min(
            100.0,
            100.0 * reply_harm * (0.5 + reply_share) + 20.0 * max(0.0, reply_harm - top_harm),
        )
        rows.append(
            {
                "platform": plat,
                "reply_pileon_score": round(pileon_score, 2),
                "reply_harm_rate": round(reply_harm, 4),
                "n_replies": int(len(replies)),
                "reply_share": round(reply_share, 4),
            }
        )
    return pd.DataFrame(rows)


def sentiment_platform_factors(master: pd.DataFrame) -> pd.DataFrame:
    """Platform-level sentiment/influence aggregates for WHSI integration."""
    live = _live_corpus(master)
    rows = []
    for plat, g in live.groupby("platform"):
        harm_col = _harm_column(g)
        harmful = g[g[harm_col] == 1] if harm_col in g.columns else g.iloc[0:0]

        if "influence_score" in g.columns:
            inf_mean = float(g["influence_score"].mean())
            incite = float(g.get("incitement_score", pd.Series(0)).mean())
            manip = float(g.get("manipulation_score", pd.Series(0)).mean())
            neg_sent = float((g.get("sentiment_compound", 0) < -0.2).mean()) * 100
        else:
            inf_mean = incite = manip = neg_sent = 0.0

        harm_inf = float(harmful["influence_score"].mean()) if len(harmful) and "influence_score" in harmful.columns else inf_mean

        rows.append(
            {
                "platform": plat,
                "influence_mean": round(inf_mean, 2),
                "harmful_influence_mean": round(harm_inf, 2),
                "incitement_mean": round(incite, 2),
                "manipulation_mean": round(manip, 2),
                "negative_sentiment_pct": round(neg_sent, 2),
            }
        )
    return pd.DataFrame(rows)


def women_specific_platform_metrics(master: pd.DataFrame) -> pd.DataFrame:
    """
    Women-Online-Safety construct metrics (EIGE-style victim/perpetrator distinction).

    Primary severity signal = perpetrator attacks targeting women, NOT victim disclosure
    in support spaces (addresses TwoXChromosomes false-positive critique).
    """
    live = _live_corpus(master)
    rows = []
    for plat, g in live.groupby("platform"):
        n = len(g)
        harm_col = _harm_column(g)
        gendered = g[harm_col].astype(int) if harm_col in g.columns else pd.Series(0, index=g.index)

        if "harm_role" in g.columns:
            perpetrator = g["harm_role"] == "perpetrator_attack"
            victim_disc = g["harm_role"] == "victim_disclosure"
        else:
            perpetrator = gendered == 1
            victim_disc = pd.Series(False, index=g.index)

        perp_rate = float(perpetrator.mean())
        victim_rate = float(victim_disc.mean())
        gendered_rate = float(gendered.mean())

        perp_df = g[perpetrator]
        if len(perp_df) and "directed_at_women" in perp_df.columns:
            women_targeted_perp = float(perp_df["directed_at_women"].mean())
        elif len(perp_df) and "targets_women" in perp_df.columns:
            women_targeted_perp = float(perp_df["targets_women"].mean())
        else:
            women_targeted_perp = 0.0

        wtshi = min(100.0, 100.0 * perp_rate * (0.50 + 0.50 * women_targeted_perp))

        rows.append(
            {
                "platform": plat,
                "perpetrator_harm_rate": round(perp_rate, 4),
                "victim_disclosure_rate": round(victim_rate, 4),
                "gendered_harm_rate_all": round(gendered_rate, 4),
                "women_targeted_perpetrator_pct": round(women_targeted_perp * 100, 2),
                "wtshi_construct": round(wtshi, 2),
            }
        )
    return pd.DataFrame(rows)


def build_integrated_features(
    master: pd.DataFrame,
    community_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge base + hotspot + thread + sentiment into integrated feature vector."""
    base = extract_platform_features(master, live_only=True)
    if community_df is None or community_df.empty:
        community_df = compute_community_whsi(master, min_comments=100)

    hot = hotspot_platform_summary(community_df)
    thr = thread_reply_factors(master)
    sent = sentiment_platform_factors(master)
    women = women_specific_platform_metrics(master)

    out = (
        base.merge(hot, on="platform", how="left")
        .merge(thr, on="platform", how="left")
        .merge(sent, on="platform", how="left")
        .merge(women, on="platform", how="left")
    )

    for col in ["hotspot_whsi_mean", "hotspot_whsi_max", "reply_pileon_score", "harmful_influence_mean", "incitement_mean"]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    # Integrated dimensions — novelty layers folded in
    out["toxicity_integrated"] = np.clip(
        0.55 * out["toxicity"]
        + 0.25 * out.get("hotspot_whsi_mean", 0) * 0.6
        + 0.20 * out.get("harmful_influence_mean", 0),
        0,
        100,
    )
    out["threat_integrated"] = np.clip(
        np.maximum(out["threat"], out.get("incitement_mean", 0) * 0.45 + out.get("manipulation_mean", 0) * 0.25),
        0,
        100,
    )
    out["frequency_integrated"] = np.clip(
        0.65 * out["frequency"] + 0.35 * out.get("reply_pileon_score", 0),
        0,
        100,
    )
    out["normalization_integrated"] = out["normalization"].fillna(0)

    live = _live_corpus(master)
    hist_map = {
        plat: int(g.get("dataset_split", pd.Series([""])).astype(str).eq("historical_twitter").any())
        for plat, g in live.groupby("platform")
    }
    out["is_historical"] = out["platform"].map(hist_map).fillna(0).astype(int)

    out.to_csv("data/processed/platform_features_integrated.csv", index=False)
    return out


def run_integrated_whsi(features: pd.DataFrame, export_main_outputs: bool = True) -> pd.DataFrame:
    """
    Dual WHSI outputs:
      WHSI_raw        — 70% WTSHI + 30% fuzzy (pure measurement, no rank inflation)
      WHSI_score      — raw + capped structural bonuses (hotspot/pileon)
    Primary thesis score = WHSI_raw; WHSI_score adds thread/hotspot tail risk.
    """
    results = []
    for idx, row in features.iterrows():
        fuzzy_score, _ = compute_whsi(
            row["toxicity_integrated"],
            row["threat_integrated"],
            row["frequency_integrated"],
            row["normalization_integrated"],
            gendered_harm_rate=float(row.get("perpetrator_harm_rate", row.get("harm_rate", 0))),
            gendered_targeting_ratio=float(row.get("women_targeted_perpetrator_pct", row.get("gendered_targeting_ratio", 0))),
        )

        wtshi = float(row.get("wtshi_construct", 0) or 0)
        hotspot = min(6.0, float(row.get("hotspot_whsi_max", 0) or 0) * 0.10)
        pileon = min(4.0, float(row.get("reply_pileon_score", 0) or 0) * 0.06)

        whsi_raw = round(float(np.clip(0.70 * wtshi + 0.30 * fuzzy_score, 0, 100)), 2)

        from whsi_literature_adjustment import whsi_literature_adjusted, scrapable_ratio_sensitivity_band

        whsi_lit, lit_meta = whsi_literature_adjusted(
            platform=row["platform"],
            observed_perpetrator_rate=float(row.get("perpetrator_harm_rate", row.get("harm_rate", 0)) or 0),
            observed_women_targeted_pct=float(row.get("women_targeted_perpetrator_pct", 0) or 0),
            fuzzy_score=fuzzy_score,
        )
        lit_band = scrapable_ratio_sensitivity_band(
            row["platform"],
            float(row.get("perpetrator_harm_rate", row.get("harm_rate", 0)) or 0),
            float(row.get("women_targeted_perpetrator_pct", 0) or 0),
            fuzzy_score,
        )
        lit_vals = [b["WHSI_literature_adjusted"] for b in lit_band]

        score = round(float(np.clip(whsi_raw + hotspot + pileon, 0, 100)), 2)

        if score < 25:
            category = "Safe"
        elif score < 45:
            category = "Moderately Unsafe"
        elif score < 65:
            category = "Unsafe"
        else:
            category = "Critically Unsafe"

        results.append(
            {
                "platform": row["platform"],
                "WHSI_score": score,
                "WHSI_raw": whsi_raw,
                "WHSI_literature_adjusted": whsi_lit,
                "WHSI_lit_sensitivity_low": round(min(lit_vals), 2) if lit_vals else whsi_lit,
                "WHSI_lit_sensitivity_high": round(max(lit_vals), 2) if lit_vals else whsi_lit,
                "ecosystem_harm_exposure_rate": lit_meta.get("ecosystem_harm_exposure_rate"),
                "wtshi_literature": lit_meta.get("wtshi_literature"),
                "literature_women_targeting_provenance": lit_meta.get("women_targeting_provenance"),
                "assumption_A1": lit_meta.get("assumption_A1"),
                "WHSI_category": category,
                "wtshi_construct": wtshi,
                "fuzzy_component": fuzzy_score,
                "hotspot_component": hotspot,
                "thread_pileon_component": pileon,
                "toxicity_integrated": row["toxicity_integrated"],
                "threat_integrated": row["threat_integrated"],
                "frequency_integrated": row["frequency_integrated"],
                "normalization_integrated": row["normalization_integrated"],
                "hotspot_whsi_max": row.get("hotspot_whsi_max"),
                "top_community": row.get("top_community"),
                "n_comments": row.get("n_comments"),
                "harm_rate": row.get("perpetrator_harm_rate", row.get("harm_rate")),
                "perpetrator_harm_rate": row.get("perpetrator_harm_rate"),
                "victim_disclosure_rate": row.get("victim_disclosure_rate"),
                "women_targeted_perpetrator_pct": row.get("women_targeted_perpetrator_pct"),
            }
        )

    df = pd.DataFrame(results).sort_values("WHSI_score", ascending=False)

    export = df[
        [
            "platform",
            "WHSI_score",
            "WHSI_raw",
            "WHSI_literature_adjusted",
            "WHSI_lit_sensitivity_low",
            "WHSI_lit_sensitivity_high",
            "ecosystem_harm_exposure_rate",
            "wtshi_literature",
            "WHSI_category",
            "n_comments",
            "harm_rate",
            "perpetrator_harm_rate",
            "wtshi_construct",
            "women_targeted_perpetrator_pct",
        ]
    ].copy()
    if export_main_outputs:
        export.to_csv("data/processed/whsi_scores.csv", index=False)
        df.to_csv("data/processed/whsi_scores_integrated.csv", index=False)

        try:
            from bootstrap_metrics import compute_platform_cis

            compute_platform_cis()
        except Exception as exc:
            print(f"[WHSI] Bootstrap CI skipped: {exc}")

        try:
            from scoring_exports import enrich_whshi_scores

            enrich_whshi_scores()
        except Exception as exc:
            print(f"[WHSI] CI export skipped: {exc}")

        try:
            from whsi_sensitivity import run_whsi_sensitivity

            run_whsi_sensitivity()
        except Exception as exc:
            print(f"[WHSI] Sensitivity skipped: {exc}")

        try:
            from prevalence_correction import compute_prevalence_corrected

            compute_prevalence_corrected()
        except Exception as exc:
            print(f"[WHSI] Prevalence correction skipped: {exc}")

        try:
            from validation_corpus_scoring import score_validation_corpus

            score_validation_corpus()
        except Exception as exc:
            print(f"[WHSI] Validation corpus scoring skipped: {exc}")

        try:
            from structural_platform_evidence import export_evidence_config, export_mri_input_derivations
            from triangulated_risk_table import build_triangulated_table, print_triangulated_summary
            from whsi_literature_adjustment import export_literature_sources

            export_evidence_config()
            export_mri_input_derivations()
            export_literature_sources()
            tri = build_triangulated_table()
            print_triangulated_summary(tri)
        except Exception as exc:
            print(f"[WHSI] Triangulated risk table skipped: {exc}")

        os.makedirs("outputs/results", exist_ok=True)
        import json

        methodology = {
            "whsi_formula_primary": "WHSI_raw = 0.70*WTSHI + 0.30*fuzzy (no rank inflation)",
            "whsi_formula_final": "WHSI_score = WHSI_raw + hotspot_bonus(<=6) + pileon_bonus(<=4)",
            "epistemology": (
                "WHSI and perpetrator_harm_rate are classifier outputs until human validation "
                "on the project corpus is complete. EDOS holdout F1≈0.58 does not equal corpus-specific accuracy."
            ),
            "wtshi_construct": "100 * perpetrator_attack_rate * (0.50 + 0.50 * directed_at_women_share among perpetrators)",
            "harm_rate_reported": "perpetrator_harm_rate — classifier-estimated, not human-validated",
            "references": [
                "EIGE CVAWG measurement framework — perpetrator/victim distinction",
                "GLAAD SMSI — platform policy scorecard methodology (transparency MRI layer)",
                "Ranking Digital Rights — moderation transparency indicators",
            ],
            "scores": df[
                [
                    "platform",
                    "WHSI_score",
                    "wtshi_construct",
                    "perpetrator_harm_rate",
                    "women_targeted_perpetrator_pct",
                ]
            ].to_dict(orient="records"),
        }
        with open("outputs/results/measurement_methodology.json", "w") as f:
            json.dump(methodology, f, indent=2)

    return df


def run_integrated_pipeline(master_path: str = "data/processed/master_dataset.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_parquet(master_path) if master_path.endswith(".parquet") else pd.read_csv(master_path)
    community = compute_community_whsi(master, min_comments=100)
    features = build_integrated_features(master, community)
    whsi = run_integrated_whsi(features)
    print("\nCALIBRATED WHSI (women-targeted perpetrator + hotspots + threads + sentiment):")
    print(
        whsi[
            [
                "platform",
                "WHSI_score",
                "WHSI_category",
                "wtshi_construct",
                "perpetrator_harm_rate",
                "hotspot_whsi_max",
            ]
        ].to_string(index=False)
    )
    return features, whsi


if __name__ == "__main__":
    path = "data/processed/master_with_influence.parquet"
    if not os.path.exists(path):
        path = "data/processed/master_dataset.csv"
    run_integrated_pipeline(path)
