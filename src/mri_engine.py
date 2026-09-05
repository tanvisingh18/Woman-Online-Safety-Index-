"""
SECTION 7 + SECTION 13 — MRI ENGINE
Women Safety Index | mri_engine.py

Computes MRI (Moderation Responsiveness Index) per platform:
  MRI = 0.45 × RemovalRate + 0.30 × SpeedScore + 0.25 × ConsistencyScore
  SpeedScore = max(0, 1 − ResponseHours / 720) × 100

Also implements Section 13: empirical reaction speed measurement by
comparing two scraping waves separated in time.

Output: data/processed/mri_scores.csv

Run: python src/mri_engine.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MRI FORMULA
# ─────────────────────────────────────────────────────────────

def speed_score(avg_response_hours: float, max_hours: float = 720.0) -> float:
    """
    SpeedScore ∈ [0, 100].
    720 hours = 30 days. Platforms taking ≥ 30 days score 0.
    Fast platforms (< 1 hour) score near 100.
    """
    return max(0.0, (1.0 - avg_response_hours / max_hours)) * 100.0


def compute_mri(removal_rate: float,
                avg_response_hours: float,
                consistency_score: float) -> float:
    """
    Moderation Responsiveness Index.

    Parameters
    ----------
    removal_rate       : proportion of harmful content removed (0–1)
    avg_response_hours : mean hours from report to removal
    consistency_score  : consistency of enforcement (0–1, from transparency report)

    Returns
    -------
    MRI score ∈ [0, 100]
    """
    R  = float(np.clip(removal_rate,    0.0, 1.0)) * 100.0
    S  = speed_score(float(avg_response_hours))
    C  = float(np.clip(consistency_score, 0.0, 1.0)) * 100.0
    return round(0.45 * R + 0.30 * S + 0.25 * C, 2)


def mri_label(score: float) -> str:
    """Categorical label for MRI score."""
    if score >= 75:
        return "Highly Accountable"
    elif score >= 50:
        return "Moderately Accountable"
    elif score >= 25:
        return "Poorly Accountable"
    else:
        return "Unaccountable"


# ─────────────────────────────────────────────────────────────
# TRANSPARENCY REPORT → MRI BATCH
# ─────────────────────────────────────────────────────────────

def compute_mri_from_transparency(report_path: str = "data/transparency/moderation_reports.csv") -> pd.DataFrame:
    """
    Load transparency CSV and compute MRI for each platform.
    Applies PDR (proactive detection) and RAS (regulatory accountability) sub-adjustments.
    """
    from structural_platform_evidence import adjust_consistency_score, adjust_removal_rate

    df = pd.read_csv(report_path)
    results = []
    for _, row in df.iterrows():
        plat = row["platform"]
        removal_adj, pdr = adjust_removal_rate(float(row["removal_rate"]), plat)
        consistency_adj, ras = adjust_consistency_score(float(row["consistency_score"]), plat)
        score = compute_mri(
            removal_adj,
            row["avg_response_hours"],
            consistency_adj,
        )
        results.append({
            "platform":           plat,
            "removal_rate_pct":   round(removal_adj * 100, 1),
            "removal_reactive_pct": round(float(row["removal_rate"]) * 100, 1),
            "proactive_detection_pct": round(pdr * 100, 1),
            "avg_response_hours": row["avg_response_hours"],
            "consistency_pct":    round(consistency_adj * 100, 1),
            "consistency_reported_pct": round(float(row["consistency_score"]) * 100, 1),
            "regulatory_accountability_pct": round(ras * 100, 1),
            "speed_score":        round(speed_score(row["avg_response_hours"]), 2),
            "MRI_score":          score,
            "MRI_label":          mri_label(score),
            "source":             row.get("source", ""),
            "year":               row.get("year", 2023),
        })

    df_out = pd.DataFrame(results).sort_values("MRI_score", ascending=False)
    out_path = "data/processed/mri_scores.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("MRI SCORES — MODERATION RESPONSIVENESS INDEX")
    print(df_out[["platform","MRI_score","MRI_label","removal_rate_pct","avg_response_hours"]].to_string(index=False))
    print(f"\nSaved → {out_path}")
    print(f"{'='*60}\n")
    return df_out


# ─────────────────────────────────────────────────────────────
# SECTION 13: EMPIRICAL REACTION SPEED MEASUREMENT
# Compare two scraping waves to measure what was actually removed
# ─────────────────────────────────────────────────────────────

def measure_empirical_reaction_speed(scrape_wave_1_path: str,
                                      scrape_wave_2_path: str,
                                      hours_between: float = 48.0) -> dict:
    """
    Compare wave 1 and wave 2 scraped data (separated by hours_between hours).
    Identifies harmful posts that disappeared → empirical removal rate.

    Usage:
      1. Run reddit_scraper.py → save to data/scraped/reddit_wave1.parquet
      2. Wait 48 hours.
      3. Re-run same scraper → save to data/scraped/reddit_wave2.parquet
      4. Call this function to measure platform reaction speed.

    Returns dict with empirical_removal_rate, avg_response_hours_estimate, survivor_rate
    """
    if not os.path.exists(scrape_wave_1_path) or not os.path.exists(scrape_wave_2_path):
        print(f"[MRI Empirical] Wave files not found. Need:\n  {scrape_wave_1_path}\n  {scrape_wave_2_path}")
        return {}

    w1 = pd.read_parquet(scrape_wave_1_path)
    w2 = pd.read_parquet(scrape_wave_2_path)

    # Unique comment identifier: post_id + author
    def make_ids(df):
        return set(df["post_id"].astype(str) + "||" + df["author"].astype(str))

    w1_ids = make_ids(w1)
    w2_ids = make_ids(w2)
    removed_ids = w1_ids - w2_ids

    # Among harmful wave-1 posts, how many are gone in wave 2?
    if "is_harmful" not in w1.columns:
        print("[MRI Empirical] 'is_harmful' column not in wave 1. Run classifier first.")
        return {}

    harmful_w1     = w1[w1["is_harmful"] == 1]
    harmful_ids    = make_ids(harmful_w1)
    n_harmful      = len(harmful_ids)
    n_removed      = len(removed_ids & harmful_ids)

    empirical_removal_rate  = n_removed / (n_harmful + 1e-9)
    survivor_count          = n_harmful - n_removed
    avg_response_estimate   = hours_between / 2.0  # midpoint estimate

    result = {
        "empirical_removal_rate":       round(empirical_removal_rate, 4),
        "avg_response_hours_estimate":  avg_response_estimate,
        "survivor_rate":                round(1 - empirical_removal_rate, 4),
        "n_harmful_wave1":              n_harmful,
        "n_removed_in_window":          n_removed,
        "n_still_live":                 survivor_count,
        "hours_between_waves":          hours_between,
    }

    print(f"\n[MRI EMPIRICAL MEASUREMENT]")
    print(f"  Harmful posts in wave 1   : {n_harmful}")
    print(f"  Removed by wave 2 ({hours_between}h)  : {n_removed}")
    print(f"  Empirical removal rate    : {empirical_removal_rate:.2%}")
    print(f"  Survivor rate             : {1-empirical_removal_rate:.2%}")
    return result


def measure_empirical_mri_from_removal_flags(
    reddit_path: str = "data/scraped/reddit_harassment_dataset.csv",
) -> dict:
    """
    Estimate empirical moderation responsiveness from is_removed / is_deleted flags
  in an existing Reddit scrape (no 48h wave wait required).
    """
    if not os.path.exists(reddit_path):
        print(f"[MRI Empirical] Missing {reddit_path}")
        return {}

    df = pd.read_csv(reddit_path, low_memory=False)

    if "is_gendered_harm" not in df.columns:
        try:
            from gendered_harm_model import label_dataframe
            df = label_dataframe(df, text_col="comment_text", batch_size=256)
        except Exception:
            pass

    harm_col = "is_gendered_harm" if "is_gendered_harm" in df.columns else None
    if harm_col:
        harmful = df[df[harm_col] == 1]
    else:
        harmful = df  # use all if not labelled yet

    n_total = len(harmful)
    if n_total == 0:
        return {}

    removed = harmful[
        (harmful.get("is_removed", 0) == 1) | (harmful.get("is_deleted", 0) == 1)
    ]
    n_removed = len(removed)
    removal_rate = n_removed / (n_total + 1e-9)

    # Heuristic response hours: removed faster on high-score toxic posts
    avg_hours = 72.0 if removal_rate > 0.05 else 168.0

    result = {
        "platform": "Reddit",
        "empirical_removal_rate": round(removal_rate, 4),
        "avg_response_hours_estimate": avg_hours,
        "n_harmful_or_total": n_total,
        "n_removed_flagged": n_removed,
        "method": "removal_flags_in_scrape",
    }

    print(f"\n[MRI EMPIRICAL — Reddit removal flags]")
    print(f"  Flagged removed/deleted: {n_removed}/{n_total} ({removal_rate:.2%})")
    return result


def run_empirical_mri_and_blend(
    wave1: str = "data/scraped/reddit_wave1.parquet",
    wave2: str = "data/scraped/reddit_wave2.parquet",
    report_path: str = "data/transparency/moderation_reports.csv",
) -> pd.DataFrame:
    """Blend transparency MRI with empirical evidence when available."""
    empirical = {}
    if os.path.exists(wave1) and os.path.exists(wave2):
        empirical["Reddit"] = measure_empirical_reaction_speed(wave1, wave2)
    else:
        flag_data = measure_empirical_mri_from_removal_flags()
        if flag_data:
            empirical["Reddit"] = flag_data

    trans = pd.read_csv(report_path)
    rows = []
    for _, row in trans.iterrows():
        plat = row["platform"]
        emp = empirical.get(plat, {})
        base = compute_mri(row["removal_rate"], row["avg_response_hours"], row["consistency_score"])
        blended = compute_enhanced_mri(row, emp) if emp else base
        rows.append(
            {
                "platform": plat,
                "MRI_score": blended,
                "MRI_label": mri_label(blended),
                "MRI_transparency_only": base,
                "empirical_removal_rate": emp.get("empirical_removal_rate"),
                "empirical_method": emp.get("method", "transparency_only"),
            }
        )

    out = pd.DataFrame(rows).sort_values("MRI_score", ascending=False)
    out.to_csv("data/processed/mri_scores_empirical.csv", index=False)
    print(f"Blended MRI saved → data/processed/mri_scores_empirical.csv")
    return out


def compute_enhanced_mri(transparency_row: pd.Series,
                          empirical_data:    dict = None,
                          empirical_weight:  float = 0.40) -> float:
    """
    Blend self-reported transparency MRI with empirically measured MRI.
    empirical_weight: how much scrape evidence overrides reported values (0–1).
    """
    base_mri = compute_mri(
        transparency_row["removal_rate"],
        transparency_row["avg_response_hours"],
        transparency_row["consistency_score"]
    )
    if not empirical_data:
        return base_mri

    empirical_mri = compute_mri(
        empirical_data["empirical_removal_rate"],
        empirical_data["avg_response_hours_estimate"],
        transparency_row["consistency_score"]   # keep reported consistency
    )
    blended = (1 - empirical_weight) * base_mri + empirical_weight * empirical_mri
    return round(blended, 2)


# ─────────────────────────────────────────────────────────────
# MRI VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_mri_breakdown(mri_df: pd.DataFrame,
                       save_path: str = "outputs/plots/mri_breakdown.png"):
    """
    Horizontal bar chart showing MRI components per platform.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    platforms = mri_df["platform"].tolist()
    y         = np.arange(len(platforms))
    bar_h     = 0.25

    ax.barh(y + bar_h,  mri_df["removal_rate_pct"] * 0.45,
            bar_h, label="Removal Rate (×0.45)", color="#2196F3", alpha=0.85)
    ax.barh(y,          mri_df["speed_score"] * 0.30,
            bar_h, label="Speed Score (×0.30)",   color="#4CAF50", alpha=0.85)
    ax.barh(y - bar_h,  mri_df["consistency_pct"] * 0.25,
            bar_h, label="Consistency (×0.25)",   color="#FF9800", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(platforms, fontsize=11)
    ax.set_xlabel("Weighted Component Score", fontsize=11)
    ax.set_title("MRI Component Breakdown by Platform", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate total MRI score
    for i, row in mri_df.iterrows():
        idx = platforms.index(row["platform"])
        ax.text(row["MRI_score"] + 0.5, idx, f"  MRI={row['MRI_score']:.0f}",
                va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"MRI breakdown plot saved → {save_path}")


if __name__ == "__main__":
    import json

    mri_df = compute_mri_from_transparency()
    plot_mri_breakdown(mri_df)

    # Faculty July Step 5 option (b): DO NOT blend proactive wave-2 into reactive MRI.
    # Keep transparency-only MRI in mri_scores.csv; export proactive probe separately.
    emp_path = "outputs/results/empirical_mri_all.json"
    proactive = {
        "policy": "unblended",
        "rationale": (
            "Wave-2 0% removal measures unreported/proactive survival, not reactive "
            "removal-after-report. Blending it into MRI's removal term is a category error. "
            "MRI_score in mri_scores.csv is transparency+PDR/RAS only. The 0% probe is "
            "reported beside coded PDR as a standalone empirical check."
        ),
        "reddit_MRI_transparency_only": None,
        "proactive_probe": None,
    }
    if "Reddit" in mri_df["platform"].values:
        proactive["reddit_MRI_transparency_only"] = float(
            mri_df.loc[mri_df["platform"] == "Reddit", "MRI_score"].iloc[0]
        )
    if os.path.exists(emp_path):
        with open(emp_path) as f:
            emp_all = json.load(f)
        reddit_emp = emp_all.get("Reddit") or emp_all.get("results", {}).get("Reddit")
        if reddit_emp:
            proactive["proactive_probe"] = {
                "platform": "Reddit",
                "empirical_removal_rate": reddit_emp.get("empirical_removal_rate", 0.0),
                "n_harmful_wave1": reddit_emp.get("n_harmful_wave1"),
                "n_removed_in_window": reddit_emp.get("n_removed_in_window"),
                "interpretation": (
                    "Reported proactive detection is not observable in our harassment sample "
                    f"(coded PDR for Reddit = 38%; empirical unreported removal = "
                    f"{100 * float(reddit_emp.get('empirical_removal_rate') or 0):.1f}%)."
                ),
                "former_blended_MRI": reddit_emp.get("MRI_blended"),
                "MRI_now_in_matrix": proactive["reddit_MRI_transparency_only"],
            }
        # Ensure source strings do not claim blend; keep transparency-only MRI
        mri_df = compute_mri_from_transparency()
        mri_df["source"] = mri_df["source"].astype(str).str.replace(
            r" \+ empirical wave2 blend", "", regex=True
        )
        mri_df.to_csv("data/processed/mri_scores.csv", index=False)
        if "Reddit" in mri_df["platform"].values:
            proactive["reddit_MRI_transparency_only"] = float(
                mri_df.loc[mri_df["platform"] == "Reddit", "MRI_score"].iloc[0]
            )
            if proactive.get("proactive_probe"):
                proactive["proactive_probe"]["MRI_now_in_matrix"] = proactive[
                    "reddit_MRI_transparency_only"
                ]

    with open("outputs/results/mri_proactive_probe_standalone.json", "w") as f:
        json.dump(proactive, f, indent=2)
    print("UNBLENDED: MRI_score = transparency+PDR/RAS only (no proactive blend)")
    print(f"Standalone proactive probe → outputs/results/mri_proactive_probe_standalone.json")
    print(f"Reddit MRI (matrix): {proactive['reddit_MRI_transparency_only']}")

    empirical = measure_empirical_reaction_speed(
        "data/scraped/reddit_wave1.parquet",
        "data/scraped/reddit_wave2.parquet",
        hours_between=48,
    )
    if empirical:
        print(
            f"\n[Standalone proactive probe] empirical_removal_rate="
            f"{empirical.get('empirical_removal_rate')} — NOT blended into MRI_score"
        )