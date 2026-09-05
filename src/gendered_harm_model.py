"""
Gendered harm classifier — trained on EDOS (sexist / not sexist).
Fully offline after first train; no HuggingFace required.

Outputs per text:
  gendered_harm_proba, is_gendered_harm, toxicity_score, threat_score

Run:
  python src/gendered_harm_model.py --train
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models/harm_models")
SEXIST_MODEL = MODEL_DIR / "edos_sexist_pipeline.joblib"
THREAT_MODEL = MODEL_DIR / "edos_threat_pipeline.joblib"

# Corpus-tuned on 360 dual-annotated gold labels (validation_sample.csv).
# Old EDOS-proxy point (0.52 / 0.45 / 0.38) gave P=1.00 R=0.72 F1=0.84.
# New operating point maximizes F1 on project gold while keeping P≥0.90:
#   P≈0.91 R≈0.96 F1≈0.93 (Telegram R rises from 0.45 → ~0.93).
PERP_SEXIST_THRESH = 0.40
PERP_SEXIST_WITH_THREAT = 0.34
PERP_THREAT_MIN = 0.28

EDOS_PATHS = [
    "data/labelled/edos_labelled_aggregated.csv",
    "../edos_labelled_aggregated.csv",
    "data/raw/edos_labelled_aggregated.csv",
]

# EDOS category → toxicity / threat priors
HIGH_TOX_CATS = ("derogation", "animosity", "prejudiced", "threatening", "threat")
THREAT_CATS = ("threat", "threatening")

FEMALE_TARGET = re.compile(
    r"\b(she|her|hers|woman|women|girl|girls|female|females|bitch|slut|whore|"
    r"mother|daughter|sister|girlfriend|wife)\b",
    re.IGNORECASE,
)

# Victim-disclosure patterns (discussing abuse suffered — not perpetrator speech)
VICTIM_DISCLOSURE = re.compile(
    r"\b(i was|i've been|happened to me|he did|they did|assaulted me|harassed me|"
    r"my ex|reported him|afraid of him)\b",
    re.IGNORECASE,
)

PERPETRATOR_ATTACK = re.compile(
    r"\b(she is|she's|women are|all women|females are|stupid bitch|dumb whore|"
    r"should be raped|deserves to die|kill her|shut up woman)\b",
    re.IGNORECASE,
)


def find_edos_path() -> str | None:
    for p in EDOS_PATHS:
        if os.path.exists(p):
            return p
    return None


def load_edos_training_frame() -> pd.DataFrame:
    path = find_edos_path()
    if not path:
        raise FileNotFoundError(
            "EDOS not found. Place edos_labelled_aggregated.csv in data/labelled/"
        )
    df = pd.read_csv(path)
    df = df.rename(columns={"text": "comment_text"})
    df["comment_text"] = df["comment_text"].astype(str)
    df = df[df["comment_text"].str.len() > 10]
    df["is_sexist"] = (df["label_sexist"].astype(str).str.lower() == "sexist").astype(int)
    cat = df.get("label_category", pd.Series([""] * len(df))).astype(str).str.lower()
    df["is_threat_cat"] = cat.apply(lambda c: int(any(t in c for t in THREAT_CATS)))
    df["toxicity_prior"] = cat.apply(
        lambda c: 0.90 if any(h in c for h in HIGH_TOX_CATS) else (0.55 if "sexist" in c else 0.08)
    )
    df["threat_prior"] = cat.apply(lambda c: 0.92 if any(t in c for t in THREAT_CATS) else 0.12)
    return df


def train_models(test_size: float = 0.15, random_state: int = 42) -> dict:
    """Train sexist + threat TF-IDF logistic models on EDOS."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_edos_training_frame()

    X = df["comment_text"]
    y_sexist = df["is_sexist"]
    y_threat = df["is_threat_cat"]

    def _fit(X_train, y_train, name: str) -> Pipeline:
        pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=60_000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=400,
                        class_weight="balanced",
                        C=1.5,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        return pipe

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_sexist, test_size=test_size, random_state=random_state, stratify=y_sexist
    )
    sexist_pipe = _fit(X_tr, y_tr, "sexist")
    joblib.dump(sexist_pipe, SEXIST_MODEL)

    _, _, yt_tr, yt_te = train_test_split(
        X, y_threat, test_size=test_size, random_state=random_state, stratify=y_threat
    )
    threat_pipe = _fit(X_tr, yt_tr, "threat")
    joblib.dump(threat_pipe, THREAT_MODEL)

    sexist_pred = sexist_pipe.predict(X_te)
    threat_pred = threat_pipe.predict(X_te)

    report = {
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "sexist_report": classification_report(y_te, sexist_pred, output_dict=True),
        "threat_report": classification_report(yt_te, threat_pred, output_dict=True),
    }
    print(f"[GenderedHarm] Trained on {len(df)} EDOS rows")
    print(classification_report(y_te, sexist_pred, target_names=["not_sexist", "sexist"]))
    print(f"Models saved → {MODEL_DIR}")
    return report


def _load_pipe(path: Path) -> Pipeline:
    if not path.exists():
        train_models()
    return joblib.load(path)


def predict_batch(texts: list[str]) -> pd.DataFrame:
    """Batch predict gendered harm scores for a list of texts."""
    sexist_pipe = _load_pipe(SEXIST_MODEL)
    threat_pipe = _load_pipe(THREAT_MODEL)

    clean = [str(t) if isinstance(t, str) else "" for t in texts]
    sexist_proba = sexist_pipe.predict_proba(clean)[:, 1]
    threat_proba = threat_pipe.predict_proba(clean)[:, 1]

    rows = []
    for text, sp, tp in zip(clean, sexist_proba, threat_proba):
        rows.append(_scores_from_proba(text, float(sp), float(tp)))

    return pd.DataFrame(rows)


def _scores_from_proba(text: str, sexist_proba: float, threat_proba: float) -> dict:
    if len(text.strip()) < 3:
        return {
            "gendered_harm_proba": 0.0,
            "is_gendered_harm": 0,
            "is_harmful": 0,
            "toxicity_score": 0.0,
            "threat_score": 0.0,
            "harm_role": "neutral",
            "targets_women": 0,
            "directed_at_women": 0,
        }

    targets_women = int(bool(FEMALE_TARGET.search(text)))
    victim = bool(VICTIM_DISCLOSURE.search(text))
    attack_kw = bool(PERPETRATOR_ATTACK.search(text))

    # Perpetrator: model sexism OR explicit attack — NOT requiring female-target keywords
    # Thresholds: corpus gold PR curve (see outputs/results/threshold_justification.json)
    is_perpetrator = (
        attack_kw
        or (sexist_proba >= PERP_SEXIST_THRESH and not victim)
        or (sexist_proba >= PERP_SEXIST_WITH_THREAT and threat_proba >= PERP_THREAT_MIN and not victim)
    )

    if victim and not is_perpetrator and sexist_proba < 0.55:
        sexist_proba *= 0.35
        threat_proba *= 0.40
        harm_role = "victim_disclosure"
    elif is_perpetrator:
        harm_role = "perpetrator_attack"
    else:
        harm_role = "neutral_discourse"

    # Directed at women: independent of perpetrator flag (fixes 100% GTR circularity)
    directed_at_women = int(
        targets_women
        and (
            is_perpetrator
            or attack_kw
            or (sexist_proba >= 0.40 and threat_proba >= 0.25)
        )
    )

    tox = float(np.clip(max(sexist_proba, threat_proba * 0.85), 0, 1))
    threat = float(np.clip(max(threat_proba, sexist_proba * 0.5 if is_perpetrator else 0), 0, 1))

    is_gendered = int(sexist_proba >= 0.42 or is_perpetrator or (directed_at_women and sexist_proba >= 0.35))
    is_harmful = int(is_gendered or threat >= 0.50 or tox >= 0.55)

    return {
        "gendered_harm_proba": round(sexist_proba, 4),
        "is_gendered_harm": is_gendered,
        "is_harmful": is_harmful,
        "toxicity_score": round(tox, 4),
        "threat_score": round(threat, 4),
        "harm_role": harm_role,
        "targets_women": targets_women,
        "directed_at_women": directed_at_women,
    }


def label_dataframe(df: pd.DataFrame, text_col: str = "comment_text", batch_size: int = 512) -> pd.DataFrame:
    from tqdm import tqdm

    df = df.copy()
    texts = df[text_col].fillna("").tolist()
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Gendered harm labelling"):
        results.append(predict_batch(texts[i : i + batch_size]))
    scores = pd.concat(results, ignore_index=True)
    for col in scores.columns:
        df[col] = scores[col].values

    df["target_gender"] = np.where(
        df["targets_women"] == 1,
        "female",
        df.get("target_gender", pd.Series(["mixed"] * len(df))),
    )
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        os.makedirs("data/labelled", exist_ok=True)
        src = find_edos_path()
        if src and not os.path.exists("data/labelled/edos_labelled_aggregated.csv"):
            import shutil

            shutil.copy(src, "data/labelled/edos_labelled_aggregated.csv")
        train_models()
