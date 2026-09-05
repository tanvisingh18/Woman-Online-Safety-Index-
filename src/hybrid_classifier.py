"""
Hybrid harm classifier — women-specific gendered harm + optional Detoxify/toxic-bert.

Priority order:
  1. EDOS-trained gendered harm (always — offline, women-specific)
  2. Detoxify multilingual toxicity (if installed)
  3. Keyword fallback

Run: python src/hybrid_classifier.py --sample 500
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

_DETOXIFY = None
_DETOXIFY_OK: bool | None = None


def _load_detoxify():
    global _DETOXIFY, _DETOXIFY_OK
    if _DETOXIFY_OK is False:
        return None
    if _DETOXIFY is not None:
        return _DETOXIFY
    try:
        cache = os.path.join(os.path.dirname(__file__), "..", "models", "torch_cache")
        os.makedirs(cache, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", cache)
        hf_home = os.path.join(cache, "hf")
        os.environ.setdefault("HF_HOME", hf_home)
        # Use local HF cache when weights are already downloaded (avoids 403 offline)
        if os.path.isdir(os.path.join(hf_home, "hub")):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from detoxify import Detoxify

        _DETOXIFY = Detoxify("original", device="cpu")
        _DETOXIFY_OK = True
        print("[HybridClassifier] Detoxify loaded (original)")
        return _DETOXIFY
    except Exception as e:
        _DETOXIFY_OK = False
        print(f"[HybridClassifier] Detoxify unavailable: {e}")
        return None


def label_dataframe(
    df: pd.DataFrame,
    text_col: str = "comment_text",
    batch_size: int = 256,
    use_detoxify: bool = True,
) -> pd.DataFrame:
    """Label with gendered harm (primary) + Detoxify toxicity blend."""
    from gendered_harm_model import label_dataframe as label_gendered

    df = label_gendered(df, text_col=text_col, batch_size=batch_size)

    detox = _load_detoxify() if use_detoxify else None
    if detox is None:
        return df

    texts = df[text_col].fillna("").astype(str).tolist()
    tox_scores = []
    threat_scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Detoxify"):
        batch = texts[i : i + batch_size]
        try:
            out = detox.predict(batch)
            for j in range(len(batch)):
                tox_scores.append(
                    float(
                        max(
                            out["toxicity"][j],
                            out.get("severe_toxicity", [0] * len(batch))[j]
                            if isinstance(out.get("severe_toxicity"), list)
                            else out["toxicity"][j],
                            out["identity_attack"][j],
                            out["insult"][j],
                        )
                    )
                )
                threat_scores.append(float(out["threat"][j]))
        except Exception:
            tox_scores.extend([0.0] * len(batch))
            threat_scores.extend([0.0] * len(batch))

    df["detoxify_toxicity"] = tox_scores
    df["detoxify_threat"] = threat_scores

    # Blend: gendered harm is primary for WHSI; Detoxify refines severity
    df["toxicity_score"] = np.clip(
        0.65 * df["gendered_harm_proba"].astype(float)
        + 0.35 * df["detoxify_toxicity"].astype(float),
        0,
        1,
    )
    df["threat_score"] = np.clip(
        np.maximum(df["threat_score"].astype(float), df["detoxify_threat"].astype(float)),
        0,
        1,
    )

    # Re-affirm gendered flag; boost if Detoxify identity_attack high + female target
    if "targets_women" in df.columns:
        boost = (df["detoxify_toxicity"] > 0.7) & (df["targets_women"] == 1)
        df.loc[boost, "is_gendered_harm"] = 1

    df["is_harmful"] = np.maximum(
        df["is_gendered_harm"].astype(int),
        (df["detoxify_toxicity"] >= 0.55).astype(int),
    )
    df["whsi_harm_flag"] = df["is_gendered_harm"].fillna(0).astype(int)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/master_women_relevant.csv")
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=42)
    out = label_dataframe(df)
    rate = out["is_gendered_harm"].mean()
    print(f"Gendered harm rate: {rate:.2%} | n={len(out)}")
