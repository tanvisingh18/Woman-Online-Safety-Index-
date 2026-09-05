"""
SECTION 5 — FEATURE EXTRACTION
Women Safety Index | feature_extraction.py

Computes 4 WHSI input dimensions per platform:
  T   (Toxicity)      → [0, 100]
  Th  (Threat)        → [0, 100]
  F   (Frequency)     → [0, 100]  uses HHI-based concentration index
  N   (Normalization) → [0, 100]  upvote-weighted harm ratio

Output: data/processed/platform_features.csv

Run: python src/feature_extraction.py
"""

import os, warnings
import pandas as pd
import numpy as np
from typing import Tuple

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)


def _harm_column(df: pd.DataFrame) -> str:
    """Prefer gendered-harm flag (WHSI construct) over generic is_harmful."""
    if "whsi_harm_flag" in df.columns:
        return "whsi_harm_flag"
    if "is_gendered_harm" in df.columns:
        return "is_gendered_harm"
    return "is_harmful"

# ─────────────────────────────────────────────────────────────
# DIMENSION 1 — TOXICITY (T)
# Weighted mean of per-comment toxicity scores.
# UCBerkeley hate_speech_score feeds this directly (continuous).
# All other datasets use pre-mapped toxicity_score column.
# ─────────────────────────────────────────────────────────────

def compute_toxicity(df_platform: pd.DataFrame) -> float:
    """
    Aggregate toxicity — weighted by gendered-harm among harmful comments.
    Women-targeted harm weighted higher (WHSI construct).
    """
    if df_platform.empty:
        return 0.0

    scores = df_platform["toxicity_score"].fillna(0.0).clip(0, 1)

    if "targets_women" in df_platform.columns:
        women_w = 1.0 + 0.5 * df_platform["targets_women"].fillna(0)
    elif "is_gendered_harm" in df_platform.columns:
        women_w = 1.0 + 0.4 * df_platform["is_gendered_harm"].fillna(0)
    else:
        women_w = np.ones(len(scores))

    if "score" in df_platform.columns:
        engagement = df_platform["score"].fillna(0).clip(lower=0)
        weights = (1 + np.log1p(engagement)) * women_w
    else:
        weights = women_w

    weighted_mean = float(np.average(scores, weights=weights))
    return round(weighted_mean * 100, 2)


# ─────────────────────────────────────────────────────────────
# DIMENSION 2 — THREAT (Th)
# Combines:
#   a) threat_score column (from annotations / Davidson)
#   b) keyword-based threat detection (regex on raw text)
#   c) presence of doxing / call-to-action language
# ─────────────────────────────────────────────────────────────

# Extended threat lexicon — curated from literature
THREAT_KEYWORDS = [
    # Physical harm
    r"\bkill\b", r"\bmurder\b", r"\bhurt\b", r"\bbeat\b", r"\battack\b",
    r"\bstab\b", r"\bshoot\b", r"\braped?\b", r"\bbash\b", r"\bslap\b",
    # Doxing / coordinated harassment
    r"\baddress\b.{0,20}\bfind\b", r"\bfind\b.{0,20}\baddress\b",
    r"\bdox\b", r"\bpost\s+her\b", r"\bleak\b",
    # Silencing
    r"\bshut\s+up\b", r"\bsilence\b.{0,15}\bher\b",
    r"\bgo\s+kill\s+yourself\b", r"\bkys\b",
    # Incitement
    r"\bsomeone\s+should\b", r"\bteach\s+her\s+a\s+lesson\b",
    r"\bdeserves?\s+(to|what)", r"\bfloods?\s+her\b",
]

import re as _re
_THREAT_PATTERN = _re.compile("|".join(THREAT_KEYWORDS), _re.IGNORECASE)

def _keyword_threat_flag(text: str) -> float:
    """Return 1.0 if text contains threat keywords, else 0.0"""
    if not isinstance(text, str):
        return 0.0
    return 1.0 if _THREAT_PATTERN.search(text) else 0.0


def compute_threat(df_platform: pd.DataFrame) -> float:
    """
    Compute Threat dimension (0–100).
    Combines annotation-based threat_score with keyword detection.
    Final = 60% annotation + 40% keyword detection.
    """
    if df_platform.empty:
        return 0.0

    ann_score = df_platform["threat_score"].fillna(0).clip(0, 1)

    keyword_flags = df_platform["comment_text"].apply(_keyword_threat_flag)
    keyword_score = keyword_flags.mean()

    combined = 0.60 * ann_score.mean() + 0.40 * keyword_score
    return round(float(combined) * 100, 2)


# ─────────────────────────────────────────────────────────────
# DIMENSION 3 — FREQUENCY (F)
# Herfindahl-Hirschman Index (HHI) based concentration.
# Measures how concentrated harm is — not just how often it occurs.
# HHI = Σ (n_i / N)² where n_i = harmful posts from user i
#
# Pure rate: harmful_rate = harmful / total
# Concentration: HHI across users (repeat offenders inflate this)
# F = α × harmful_rate + β × HHI_normalised
# ─────────────────────────────────────────────────────────────

def compute_frequency(df_platform: pd.DataFrame,
                      alpha: float = 0.70,
                      beta:  float = 0.30) -> float:
    """
    Frequency dimension (0–100): gendered-harm prevalence + offender concentration.
    Uses gendered_harm_rate when available (women-specific construct).
    """
    if df_platform.empty:
        return 0.0

    harm_col = _harm_column(df_platform)
    harmful = df_platform[df_platform[harm_col] == 1]
    N       = len(df_platform)
    n_harm  = len(harmful)

    if N == 0:
        return 0.0

    # Women-specific prevalence (primary signal)
    if "is_gendered_harm" in df_platform.columns:
        harmful_rate = float(df_platform["is_gendered_harm"].mean())
    else:
        harmful_rate = n_harm / N

    if "author" in df_platform.columns and n_harm > 0:
        author_counts = harmful["author"].value_counts()
        shares        = author_counts / n_harm
        hhi           = float((shares ** 2).sum())
    else:
        hhi = 0.5 if harmful_rate > 0 else 0.0

    frequency_score = alpha * harmful_rate + beta * hhi
    return round(float(frequency_score) * 100, 2)


# ─────────────────────────────────────────────────────────────
# DIMENSION 4 — NORMALIZATION (N)
# The degree to which harassment is socially ACCEPTED on the platform.
# Measured by:
#   a) Upvote ratio of harmful vs. non-harmful posts (if 'score' available)
#   b) Controversiality — low controversiality on harmful posts = it's "normal"
#   c) Proportion of high-upvote harmful posts vs total harmful
# ─────────────────────────────────────────────────────────────

def compute_normalization(df_platform: pd.DataFrame) -> float:
    """
    Normalization dimension (0–100).
    High = harmful content receives positive social signals (upvotes, engagement).
    """
    if df_platform.empty:
        return 0.0

    harm_col    = _harm_column(df_platform)
    harmful     = df_platform[df_platform[harm_col] == 1]
    non_harmful = df_platform[df_platform[harm_col] == 0]

    if len(harmful) == 0:
        return 0.0

    # Component 1: Engagement ratio (harmful vs non-harmful average score)
    if "score" in df_platform.columns:
        # Robust against NaN-only score columns (common in raw exports)
        harm_eng     = harmful["score"].fillna(0).clip(lower=0).mean()
        non_harm_eng = (
            non_harmful["score"].fillna(0).clip(lower=0).mean()
            if len(non_harmful) > 0
            else 1.0
        )
        # If harmful posts get MORE engagement than normal posts → high normalization
        eng_ratio    = min(harm_eng / (non_harm_eng + 1e-9), 3.0) / 3.0  # cap at 3×
    else:
        eng_ratio    = 0.5  # neutral default

    # Component 2: Low controversiality on harmful posts (they go unchallenged)
    if "controversiality" in harmful.columns:
        # Low controversiality for harmful = community accepts it = high normalization
        low_controver = 1.0 - harmful["controversiality"].mean()
    else:
        low_controver = 0.5

    # Component 3: Proportion of harmful posts that are net-positive (score > 0)
    if "score" in harmful.columns:
        pos_ratio = (harmful["score"].fillna(0) > 0).mean()
    else:
        pos_ratio = 0.5

    normalization = (eng_ratio * 0.40 + low_controver * 0.35 + pos_ratio * 0.25)
    return round(float(normalization) * 100, 2)


# ─────────────────────────────────────────────────────────────
# EXTENDED METRICS (Section 14)
# ─────────────────────────────────────────────────────────────

def compute_viral_harm_score(df_platform: pd.DataFrame) -> float:
    """
    Viral Harm Score = mean(toxicity × engagement_weight) for harmful posts.
    Measures how far harmful content spreads.
    """
    harm_col = _harm_column(df_platform)
    harmful = df_platform[df_platform[harm_col] == 1].copy()
    if "likes" not in harmful.columns or len(harmful) == 0:
        return 0.0
    mean_eng             = df_platform["likes"].mean() + 1e-9
    harmful["vw"]        = harmful["likes"] / mean_eng
    return float((harmful["toxicity_score"] * harmful["vw"]).mean() * 100)


def compute_gendered_targeting_ratio(df_platform: pd.DataFrame) -> float:
    """Proportion (%) of harmful comments that explicitly target women."""
    harm_col = _harm_column(df_platform)
    harmful = df_platform[df_platform[harm_col] == 1]
    if len(harmful) == 0:
        return 0.0

    if "targets_women" in harmful.columns:
        return round(float(harmful["targets_women"].mean()) * 100, 2)

    if "target_gender" in harmful.columns:
        female_flag = harmful["target_gender"].astype(str).str.contains(
            "female|woman|women", case=False, na=False
        )
        return round(float(female_flag.mean()) * 100, 2)

    return 0.0


def compute_echo_chamber_index(df_community: pd.DataFrame) -> float:
    """
    Low std-dev of sentiment = echo chamber (everyone agrees).
    Returns 0–100, high = strong echo chamber.
    """
    if "sentiment_compound" not in df_community.columns:
        return 0.0
    std = df_community["sentiment_compound"].std()
    return round((1 - min(std, 1.0)) * 100, 2)


# ─────────────────────────────────────────────────────────────
# PLATFORM AGGREGATION — runs all 4 dimensions per platform
# ─────────────────────────────────────────────────────────────

def extract_platform_features(master_df: pd.DataFrame,
                            live_only: bool = False) -> pd.DataFrame:
    """
    For each platform, compute [T, Th, F, N] and extended metrics.
    live_only=True excludes EDOS training anchor only (live + historical platform corpora).
    """
    if live_only and "dataset_split" in master_df.columns:
        from corpus_config import platform_analysis_mask

        master_df = master_df[platform_analysis_mask(master_df)].copy()
        print(f"[Features] Platform analysis corpus: {len(master_df)} rows")

    records = []
    harm_col = _harm_column(master_df)
    for platform, group in master_df.groupby("platform"):
        T  = compute_toxicity(group)
        Th_raw = compute_threat(group)
        F  = compute_frequency(group)
        N_raw = compute_normalization(group)

        from structural_platform_evidence import (
            adjust_normalization_with_vis,
            adjust_threat_with_mbr,
        )

        Th, mbr_pct = adjust_threat_with_mbr(Th_raw, platform)
        N, vis_score = adjust_normalization_with_vis(N_raw, platform)

        GTR  = compute_gendered_targeting_ratio(group)
        VHS  = compute_viral_harm_score(group)

        records.append({
            "platform":                    platform,
            "n_comments":                  len(group),
            "n_harmful":                   int(group[harm_col].sum()),
            "harm_rate":                   round(group[harm_col].mean(), 4),
            "gendered_harm_rate":          round(group.get("is_gendered_harm", group[harm_col]).mean(), 4),
            # Core WHSI inputs
            "toxicity":                    T,
            "threat":                      Th,
            "threat_raw_scrape":           Th_raw,
            "mbr_subinput":                mbr_pct,
            "frequency":                   F,
            "normalization":               N,
            "normalization_raw_scrape":    N_raw,
            "vis_subinput":                vis_score,
            # Extended metrics
            "gendered_targeting_ratio":    GTR,
            "viral_harm_score":            VHS,
        })

    df_out = pd.DataFrame(records).sort_values("toxicity", ascending=False)
    out_path = "data/processed/platform_features.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("PLATFORM FEATURE EXTRACTION COMPLETE")
    print(df_out[["platform","toxicity","threat","frequency","normalization"]].to_string(index=False))
    print(f"\nSaved → {out_path}")
    print(f"{'='*60}\n")
    return df_out


if __name__ == "__main__":
    master = pd.read_csv("data/processed/master_dataset.csv")
    features = extract_platform_features(master)