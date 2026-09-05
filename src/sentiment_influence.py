"""
SECTION 10 — SENTIMENT & INFLUENCE ANALYSIS
Women Safety Index | src/sentiment_influence.py

3-layer analysis framework:
  Layer 1: Polarity       — VADER (social-media-tuned, fast)
  Layer 2: Subjectivity   — TextBlob
  Layer 3: Influence type — Sentence-BERT semantic similarity to
                             manipulation + incitement seed sentences

Classifies each comment as: neutral / influential / harmful / manipulative / incitement

Influence scores feed back into WHSI dimensions (Section 10.4):
  - Toxicity gets weighted by influence_score
  - Threat gets boosted by incitement_score
  - Normalization gets boosted by high-influence harmful comments

Output: data/processed/master_with_influence.parquet

Run: python src/sentiment_influence.py
"""

import os, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SEED SENTENCES — define the semantic target space
# ─────────────────────────────────────────────────────────────

MANIPULATION_SEEDS = [
    "you are overreacting, it was just a joke",
    "women always lie about this",
    "she deserved what happened to her",
    "stop being so sensitive",
    "real women do not complain",
    "men are the real victims here",
    "you are imagining things",
    "that never happened to you",
    "you brought it on yourself",
    "she is asking for attention",
    "typical feminist overreaction",
    "why are you making this political",
]

INCITEMENT_SEEDS = [
    "someone should teach her a lesson",
    "she needs to be silenced",
    "go find her address",
    "flood her mentions",
    "she deserves to be doxxed",
    "everyone report her account",
    "we should all message her",
    "make her pay for this",
    "go harass her",
    "raid her stream",
]

HARMFUL_DISCOURSE_SEEDS = [
    "women are inferior",
    "women should not be allowed",
    "she is worthless",
    "females ruin everything",
    "women belong in the kitchen",
    "typical female behavior",
    "women are too emotional to lead",
]


# ─────────────────────────────────────────────────────────────
# LAZY MODEL LOADING — only load once, reuse across calls
# ─────────────────────────────────────────────────────────────

_vader   = None
_sbert   = None
_seed_emb_manip   = None
_seed_emb_incite  = None
_seed_emb_harmful = None


def _load_models():
    global _vader, _sbert, _seed_emb_manip, _seed_emb_incite, _seed_emb_harmful

    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
        print("[Sentiment] VADER loaded.")

    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer("all-MiniLM-L6-v2")   # 80MB, fast
        print("[Sentiment] SentenceBERT (all-MiniLM-L6-v2) loaded.")

        _seed_emb_manip   = _sbert.encode(MANIPULATION_SEEDS,   show_progress_bar=False)
        _seed_emb_incite  = _sbert.encode(INCITEMENT_SEEDS,     show_progress_bar=False)
        _seed_emb_harmful = _sbert.encode(HARMFUL_DISCOURSE_SEEDS, show_progress_bar=False)


# ─────────────────────────────────────────────────────────────
# CORE SCORING FUNCTION — single comment
# ─────────────────────────────────────────────────────────────

def compute_influence_score(text: str) -> dict:
    """
    Compute full influence analysis for a single comment.

    Returns
    -------
    dict with keys:
      sentiment_compound   : float [-1, 1]
      sentiment_pos        : float [0, 1]
      sentiment_neg        : float [0, 1]
      subjectivity         : float [0, 1]
      polarity             : float [-1, 1]  (TextBlob)
      influence_score      : float [0, 100]
      manipulation_score   : float [0, 100]
      incitement_score     : float [0, 100]
      harmful_discourse_score : float [0, 100]
      influence_type       : str
    """
    _load_models()

    from sklearn.metrics.pairwise import cosine_similarity
    from textblob import TextBlob

    if not isinstance(text, str) or len(text.strip()) < 3:
        return {
            "sentiment_compound": 0.0, "sentiment_pos": 0.0, "sentiment_neg": 0.0,
            "subjectivity": 0.0, "polarity": 0.0,
            "influence_score": 0.0, "manipulation_score": 0.0,
            "incitement_score": 0.0, "harmful_discourse_score": 0.0,
            "influence_type": "neutral"
        }

    # ── Layer 1: VADER Polarity ─────────────────────────────
    vs       = _vader.polarity_scores(text)
    compound = vs["compound"]
    pos      = vs["pos"]
    neg      = vs["neg"]

    # ── Layer 2: TextBlob ────────────────────────────────────
    blob        = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    polarity     = blob.sentiment.polarity

    # ── Layer 3: Semantic Similarity to Seeds ─────────────────
    emb = _sbert.encode([text], show_progress_bar=False)

    manip_sim   = float(cosine_similarity(emb, _seed_emb_manip).max())
    incite_sim  = float(cosine_similarity(emb, _seed_emb_incite).max())
    harmful_sim = float(cosine_similarity(emb, _seed_emb_harmful).max())

    # ── Composite Scores ─────────────────────────────────────
    # Influence = how much this text COULD persuade someone (subjectivity × |polarity|)
    influence_score      = subjectivity * abs(compound) * 100

    # Manipulation = semantic closeness to gaslighting/victim-blaming language
    manipulation_score   = manip_sim * 100

    # Incitement = semantic closeness to call-to-action harassment language
    incitement_score     = incite_sim * 100

    # Harmful discourse = general misogynistic content
    harmful_discourse_score = harmful_sim * 100

    # ── Classification ───────────────────────────────────────
    if incitement_score > 55:
        influence_type = "incitement"
    elif manipulation_score > 50:
        influence_type = "manipulative"
    elif harmful_discourse_score > 55:
        influence_type = "harmful_discourse"
    elif compound < -0.30 and influence_score > 40:
        influence_type = "harmful"
    elif influence_score > 30 and compound < 0.0:
        influence_type = "influential_negative"
    elif influence_score > 30:
        influence_type = "influential"
    else:
        influence_type = "neutral"

    return {
        "sentiment_compound":      round(compound, 3),
        "sentiment_pos":           round(pos, 3),
        "sentiment_neg":           round(neg, 3),
        "subjectivity":            round(subjectivity, 3),
        "polarity":                round(polarity, 3),
        "influence_score":         round(influence_score, 2),
        "manipulation_score":      round(manipulation_score, 2),
        "incitement_score":        round(incitement_score, 2),
        "harmful_discourse_score": round(harmful_discourse_score, 2),
        "influence_type":          influence_type,
    }


# ─────────────────────────────────────────────────────────────
# BATCH ANALYSIS — full DataFrame
# ─────────────────────────────────────────────────────────────

def analyze_dataframe_vader_only(df: pd.DataFrame,
                                text_col: str = "comment_text",
                                batch_size: int = 512) -> pd.DataFrame:
    """Fast VADER-only influence layer (no SBERT) — used when HF offline."""
    if "sentiment_compound" in df.columns and "influence_score" in df.columns:
        return df

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob

    vader = SentimentIntensityAnalyzer()
    rows = []
    texts = df[text_col].fillna("").astype(str).tolist()
    for text in texts:
        if len(text.strip()) < 3:
            rows.append(
                {
                    "sentiment_compound": 0.0,
                    "sentiment_pos": 0.0,
                    "sentiment_neg": 0.0,
                    "subjectivity": 0.0,
                    "polarity": 0.0,
                    "influence_score": 0.0,
                    "manipulation_score": 0.0,
                    "incitement_score": 0.0,
                    "harmful_discourse_score": 0.0,
                    "influence_type": "neutral",
                }
            )
            continue
        vs = vader.polarity_scores(text)
        blob = TextBlob(text)
        compound = vs["compound"]
        subj = blob.sentiment.subjectivity
        influence = subj * abs(compound) * 100
        # Keyword incitement boost
        incite_kw = bool(
            __import__("re").search(
                r"\b(kill|dox|raid|flood|harass|address|silence her|teach her)\b", text, __import__("re").I
            )
        )
        incite = 65.0 if incite_kw and compound < 0 else max(0.0, -compound * 30)
        rows.append(
            {
                "sentiment_compound": round(compound, 3),
                "sentiment_pos": round(vs["pos"], 3),
                "sentiment_neg": round(vs["neg"], 3),
                "subjectivity": round(subj, 3),
                "polarity": round(blob.sentiment.polarity, 3),
                "influence_score": round(influence, 2),
                "manipulation_score": round(max(0, -compound * 40) if subj > 0.5 else 0, 2),
                "incitement_score": round(incite, 2),
                "harmful_discourse_score": round(max(0, -compound * 50), 2),
                "influence_type": "incitement" if incite > 55 else ("harmful" if compound < -0.4 else "neutral"),
            }
        )

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def analyze_dataframe(df: pd.DataFrame,
                       text_col:   str = "comment_text",
                       batch_size: int = 256,
                       vader_only: bool = False) -> pd.DataFrame:
    """
    Add influence columns to a DataFrame.
    Processes in batches for memory efficiency.
    Skips if influence columns already exist.
    """
    if "influence_score" in df.columns:
        print("[Sentiment] Influence columns already present. Skipping recomputation.")
        return df

    if vader_only:
        print("[Sentiment] VADER-only mode (fast, no SBERT).")
        return analyze_dataframe_vader_only(df, text_col=text_col, batch_size=batch_size)

    try:
        _load_models()   # warm up models before batch
    except Exception as e:
        print(f"[Sentiment] SBERT unavailable ({e}) — falling back to VADER-only.")
        return analyze_dataframe_vader_only(df, text_col=text_col, batch_size=batch_size)

    results = []
    texts   = df[text_col].fillna("").tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="Influence scoring"):
        batch = texts[i:i+batch_size]
        results.extend([compute_influence_score(t) for t in batch])

    scores_df = pd.DataFrame(results)
    enriched  = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    return enriched


# ─────────────────────────────────────────────────────────────
# SECTION 10.4 — INTEGRATE INFLUENCE INTO WHSI FEATURES
# Enriches the feature vector BEFORE fuzzy engine
# ─────────────────────────────────────────────────────────────

def apply_influence_weighting(platform_features_df: pd.DataFrame,
                               master_enriched_df:   pd.DataFrame) -> pd.DataFrame:
    """
    For each platform, recompute Toxicity and Threat
    weighted by influence scores from master_enriched_df.

    Returns platform_features_df with two new columns:
      toxicity_influence_weighted
      threat_incitement_weighted
    """
    records = []
    for platform, group in master_enriched_df.groupby("platform"):
        if platform not in platform_features_df["platform"].values:
            continue

        # Influence-weighted toxicity
        harmful   = group[group["is_harmful"] == 1]
        if len(harmful) > 0 and "influence_score" in harmful.columns:
            inf_weights = 1 + harmful["influence_score"].fillna(0) / 100
            tox_weighted = float(np.average(
                harmful["toxicity_score"].fillna(0), weights=inf_weights
            )) * 100
        else:
            tox_weighted = None

        # Incitement-boosted threat
        if "incitement_score" in group.columns:
            incite_mean  = group["incitement_score"].mean()
            base_threat  = platform_features_df.loc[
                platform_features_df["platform"] == platform, "threat"
            ].values[0]
            threat_boosted = min(100, base_threat + 0.30 * incite_mean)
        else:
            threat_boosted = None

        records.append({
            "platform":                     platform,
            "toxicity_influence_weighted":  round(tox_weighted, 2) if tox_weighted else None,
            "threat_incitement_weighted":   round(threat_boosted, 2) if threat_boosted else None,
        })

    boost_df = pd.DataFrame(records)
    return platform_features_df.merge(boost_df, on="platform", how="left")


# ─────────────────────────────────────────────────────────────
# VISUALISE INFLUENCE DISTRIBUTION
# ─────────────────────────────────────────────────────────────

def plot_influence_distribution(enriched_df: pd.DataFrame,
                                 save_path:   str = "outputs/plots/influence_distribution.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Type distribution
    type_counts = enriched_df["influence_type"].value_counts()
    axes[0].barh(type_counts.index, type_counts.values,
                 color=["#E53935","#FB8C00","#FF7043","#43A047","#1976D2","#9C27B0"])
    axes[0].set_title("Influence Type Distribution", fontweight="bold")
    axes[0].set_xlabel("Comment Count")

    # Influence score histogram
    axes[1].hist(enriched_df["influence_score"].dropna(), bins=40,
                 color="#1976D2", edgecolor="white", alpha=0.8)
    axes[1].set_title("Influence Score Distribution", fontweight="bold")
    axes[1].set_xlabel("Influence Score (0–100)")
    axes[1].set_ylabel("Frequency")

    # Sentiment vs influence scatter
    sample = enriched_df.sample(min(2000, len(enriched_df)), random_state=42)
    color_map = {
        "incitement": "#B71C1C", "manipulative": "#E53935",
        "harmful_discourse": "#FF7043", "harmful": "#FB8C00",
        "influential_negative": "#FDD835", "influential": "#43A047", "neutral": "#90A4AE"
    }
    for itype, grp in sample.groupby("influence_type"):
        axes[2].scatter(grp["sentiment_compound"], grp["influence_score"],
                        c=color_map.get(itype, "gray"), label=itype,
                        s=15, alpha=0.6)
    axes[2].set_xlabel("Sentiment Compound")
    axes[2].set_ylabel("Influence Score")
    axes[2].set_title("Sentiment vs Influence Space", fontweight="bold")
    axes[2].legend(fontsize=7, markerscale=2)
    axes[2].axvline(0, color="gray", linestyle="--", alpha=0.4)
    axes[2].axhline(30, color="gray", linestyle="--", alpha=0.4)

    plt.suptitle("Sentiment & Influence Analysis — Women Safety Index",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Influence distribution plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    master   = pd.read_csv("data/processed/master_dataset.csv")
    enriched = analyze_dataframe(master)

    out_path = "data/processed/master_with_influence.parquet"
    enriched.to_parquet(out_path, index=False)
    print(f"\nSaved enriched dataset → {out_path}")
    print(f"\nInfluence type breakdown:")
    print(enriched["influence_type"].value_counts().to_string())

    plot_influence_distribution(enriched)