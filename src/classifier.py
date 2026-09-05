"""
HARM CLASSIFIER — labels scraped / unlabelled comments
Women Safety Index | src/classifier.py

Used when is_harmful is None (live scrapes). Combines:
  1. Transformer hate-speech probability (optional, --use-model)
  2. Keyword threat lexicon (from feature_extraction)
  3. Semantic harmful-discourse similarity (Sentence-BERT seeds)

Output columns on input DataFrame:
  is_harmful, toxicity_score, threat_score, target_gender (heuristic)

Run:
  python src/classifier.py --input data/scraped/reddit_live.parquet
  python src/classifier.py --input data/scraped/reddit_live.parquet --use-model
"""

from __future__ import annotations

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Reuse threat patterns from feature extraction
from feature_extraction import THREAT_KEYWORDS, _keyword_threat_flag

_HF_MODEL = "unitary/toxic-bert"
_pipeline = None
_sbert = None
_harmful_seed_emb = None
_SBERT_AVAILABLE = None

HARMFUL_DISCOURSE_SEEDS = [
    "women are inferior",
    "she is worthless",
    "typical female behavior",
    "women belong in the kitchen",
    "females ruin everything",
]

FEMALE_TARGET_PATTERNS = re.compile(
    r"\b(she|her|woman|women|girl|girls|female|females|bitch|slut|whore)\b",
    re.IGNORECASE,
)


def _load_transformer():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from transformers import pipeline

    print(f"[Classifier] Loading {_HF_MODEL} …")
    _pipeline = pipeline(
        "text-classification",
        model=_HF_MODEL,
        top_k=None,
        truncation=True,
        max_length=512,
        device=-1,
    )
    return _pipeline


def _load_sbert():
    global _sbert, _harmful_seed_emb, _SBERT_AVAILABLE
    if _sbert is not None:
        return _sbert, _harmful_seed_emb
    if _SBERT_AVAILABLE is False:
        raise RuntimeError("SBERT disabled (previous load failure).")
    try:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer("all-MiniLM-L6-v2")
        _harmful_seed_emb = _sbert.encode(HARMFUL_DISCOURSE_SEEDS, show_progress_bar=False)
        _SBERT_AVAILABLE = True
        return _sbert, _harmful_seed_emb
    except Exception as e:
        # Offline / restricted environments (common in lab networks) should not block the pipeline.
        _SBERT_AVAILABLE = False
        raise RuntimeError(f"Could not load SBERT model: {e}") from e


def _semantic_harm_score(text: str) -> float:
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 0.0
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        sbert, seeds = _load_sbert()
        emb = sbert.encode([text], show_progress_bar=False)
        return float(cosine_similarity(emb, seeds).max())
    except Exception:
        # Semantic model not available → return neutral semantic score.
        return 0.0


def _model_toxicity(text: str) -> float:
    """Return P(toxic) from toxic-bert."""
    pipe = _load_transformer()
    out = pipe(text[:512])[0]
    # toxic-bert labels: toxic / severe_toxic / obscene / threat / insult / identity_hate
    toxic_labels = {"toxic", "severe_toxic", "threat", "identity_hate", "insult"}
    scores = {d["label"].lower(): d["score"] for d in out}
    return max(scores.get(l, 0.0) for l in toxic_labels)


def classify_comment(text: str, use_model: bool = False) -> dict:
    """
    Classify a single comment. Returns is_harmful, toxicity_score, threat_score.
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {"is_harmful": 0, "toxicity_score": 0.0, "threat_score": 0.0}

    keyword_threat = _keyword_threat_flag(text)
    semantic_harm = _semantic_harm_score(text)

    if use_model:
        try:
            tox = _model_toxicity(text)
        except Exception:
            tox = semantic_harm
    else:
        tox = max(semantic_harm, 0.35 if keyword_threat else 0.0)

    threat = max(keyword_threat, tox * 0.6 if keyword_threat else tox * 0.25)
    is_harmful = int(tox >= 0.45 or threat >= 0.55 or semantic_harm >= 0.55)

    return {
        "is_harmful": is_harmful,
        "toxicity_score": round(float(np.clip(tox, 0, 1)), 4),
        "threat_score": round(float(np.clip(threat, 0, 1)), 4),
    }


def classify_dataframe(
    df: pd.DataFrame,
    text_col: str = "comment_text",
    use_model: bool = False,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Add gendered-harm labels (EDOS-trained) with keyword fallback."""
    try:
        from gendered_harm_model import label_dataframe

        return label_dataframe(df, text_col=text_col, batch_size=batch_size)
    except Exception as e:
        print(f"[Classifier] Gendered-harm model unavailable ({e}); using keyword fallback.")

    df = df.copy()
    texts = df[text_col].fillna("").tolist()
    results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
        batch = texts[i : i + batch_size]
        for t in batch:
            results.append(classify_comment(t, use_model=use_model))

    scores = pd.DataFrame(results)
    for col in scores.columns:
        df[col] = scores[col].values

    if "target_gender" not in df.columns or df["target_gender"].isna().all():
        df["target_gender"] = df[text_col].apply(
            lambda t: "female" if isinstance(t, str) and FEMALE_TARGET_PATTERNS.search(t) else "mixed"
        )

    return df


def main():
    parser = argparse.ArgumentParser(description="Label scraped comments for WHSI pipeline")
    parser.add_argument("--input", required=True, help="Input parquet/csv path")
    parser.add_argument("--output", default=None, help="Output path (default: overwrite input)")
    parser.add_argument("--use-model", action="store_true", help="Use toxic-bert (slower, more accurate)")
    args = parser.parse_args()

    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    labelled = classify_dataframe(df, use_model=args.use_model)
    out = args.output or args.input
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if out.endswith(".parquet"):
        labelled.to_parquet(out, index=False)
    else:
        labelled.to_csv(out, index=False)

    n = int(labelled["is_harmful"].sum())
    print(f"Classified {len(labelled)} rows | harmful: {n} ({100 * n / len(labelled):.1f}%)")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
