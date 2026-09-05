"""
SECTION 4 — PREPROCESSING
Women Safety Index | preprocessing.py

Downloads all 5 datasets, cleans/standardizes them, and produces
data/processed/master_dataset.csv  with columns:
    comment_text | platform | subreddit | hashtags | is_harmful
    toxicity_score | threat_score | dataset_source | target_gender

Run: python src/preprocessing.py
"""

import os, re, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip URLs, collapse whitespace, remove control chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "@user", text)                  # anonymise mentions
    text = re.sub(r"[^\x20-\x7E]", " ", text)             # non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_hashtags(text: str) -> str:
    """Return comma-separated hashtags from a text string."""
    return ",".join(re.findall(r"#(\w+)", str(text).lower()))

def safe_binary(series, positive_value) -> pd.Series:
    """Map a column to 0/1 given a positive class value (or list)."""
    if isinstance(positive_value, list):
        return series.isin(positive_value).astype(int)
    return (series == positive_value).astype(int)


# ─────────────────────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────────────────────

def load_edos() -> pd.DataFrame:
    """
    Dataset 1: EDOS SemEval-2023 Task 10
    Source: https://github.com/rewire-online/edos
    Schema expected: rewire_id, text, label_sexist, label_category, label_vector, split
    """
    path = "data/raw/edos_labelled_aggregated.csv"
    if not os.path.exists(path):
        # Try to load from HuggingFace if file missing
        try:
            from datasets import load_dataset
            print("Downloading EDOS from HuggingFace...")
            ds = load_dataset("SemEvalWorkshop/sem_eval_2023_task_10_edos", trust_remote_code=True)
            df = ds["train"].to_pandas()
            df.to_csv(path, index=False)
        except Exception as e:
            print(f"[EDOS] Could not download: {e}")
            print("Place edos_labelled_aggregated.csv in data/raw/")
            return pd.DataFrame()

    df = pd.read_csv(path)

    # Standardise column names — EDOS has different naming conventions
    col_map = {}
    for c in df.columns:
        if "text" in c.lower():
            col_map[c] = "comment_text"
        elif "label_sexist" in c.lower() or "sexist" in c.lower():
            col_map[c] = "label_sexist"
        elif "category" in c.lower():
            col_map[c] = "label_category"
        elif "subreddit" in c.lower() or "source" in c.lower():
            col_map[c] = "subreddit"
    df.rename(columns=col_map, inplace=True)

    if "comment_text" not in df.columns:
        print("[EDOS] Could not find text column. Skipping.")
        return pd.DataFrame()

    df["comment_text"] = df["comment_text"].apply(clean_text)
    df["is_harmful"]   = safe_binary(df.get("label_sexist", pd.Series([])), "sexist")

    # Toxicity from category labels
    high_tox_cats = ["2. derogation", "3. threatening language", "1. threats"]
    if "label_category" in df.columns:
        df["toxicity_score"] = df["label_category"].apply(
            lambda x: 0.85 if any(h in str(x).lower() for h in high_tox_cats) else 0.45
        )
        df["threat_score"] = df["label_category"].apply(
            lambda x: 0.90 if "threat" in str(x).lower() else 0.15
        )
    else:
        df["toxicity_score"] = df["is_harmful"] * 0.65
        df["threat_score"]   = df["is_harmful"] * 0.30

    df["platform"]        = df.get("subreddit", pd.Series(["Reddit"] * len(df))).apply(
        lambda x: "Reddit" if pd.notna(x) else "Gab"
    )
    df["subreddit"]       = df.get("subreddit", pd.Series([None] * len(df)))
    df["hashtags"]        = df["comment_text"].apply(extract_hashtags)
    df["target_gender"]   = "female"
    df["dataset_source"]  = "EDOS"

    print(f"[EDOS] Loaded {len(df)} rows | harmful: {df['is_harmful'].sum()}")
    return df[["comment_text","platform","subreddit","hashtags","is_harmful",
               "toxicity_score","threat_score","dataset_source","target_gender"]]


def load_hatexplain() -> pd.DataFrame:
    """
    Dataset 2: HateXplain
    Source: https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain
    """
    path = "data/raw/hatexplain.parquet"
    if not os.path.exists(path):
        try:
            from datasets import load_dataset
            print("Downloading HateXplain from HuggingFace...")
            ds = load_dataset("Hate-speech-CNERG/hatexplain")
            df = ds["train"].to_pandas()
            df.to_parquet(path, index=False)
        except Exception as e:
            print(f"[HateXplain] Could not download: {e}")
            return pd.DataFrame()
    else:
        df = pd.read_parquet(path)

    # HateXplain tokens are stored as a list — join to string
    if "post_tokens" in df.columns:
        df["comment_text"] = df["post_tokens"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )
    elif "text" in df.columns:
        df["comment_text"] = df["text"]
    else:
        print("[HateXplain] No text column found. Skipping.")
        return pd.DataFrame()

    df["comment_text"] = df["comment_text"].apply(clean_text)

    # Majority vote label from annotators
    if "annotators" in df.columns:
        def majority_label(ann):
            try:
                labels = ann.get("label", []) if isinstance(ann, dict) else []
                if not labels:
                    return "normal"
                from collections import Counter
                return Counter(labels).most_common(1)[0][0]
            except Exception:
                return "normal"
        df["majority_label"] = df["annotators"].apply(majority_label)
    elif "label" in df.columns:
        df["majority_label"] = df["label"]
    else:
        df["majority_label"] = "normal"

    df["is_harmful"]     = safe_binary(df["majority_label"], ["hate", "hatespeech", "offensive"])
    df["toxicity_score"] = df["is_harmful"] * 0.70
    df["threat_score"]   = 0.0

    # Filter for gender-targeted posts
    if "target_communities" in df.columns:
        df["target_gender"] = df["target_communities"].apply(
            lambda x: "female" if any("women" in str(t).lower() or
                                      "female" in str(t).lower()
                                      for t in (x if isinstance(x, list) else [x]))
            else "mixed"
        )
    else:
        df["target_gender"] = "mixed"

    # Platform from post_id prefix
    if "post_id" in df.columns:
        df["platform"] = df["post_id"].apply(
            lambda x: "Twitter" if str(x).startswith("t_") else "Gab"
        )
    else:
        df["platform"] = "Twitter"

    df["subreddit"]      = None
    df["hashtags"]       = df["comment_text"].apply(extract_hashtags)
    df["dataset_source"] = "HateXplain"

    print(f"[HateXplain] Loaded {len(df)} rows | harmful: {df['is_harmful'].sum()}")
    return df[["comment_text","platform","subreddit","hashtags","is_harmful",
               "toxicity_score","threat_score","dataset_source","target_gender"]]


def load_ucberkeley() -> pd.DataFrame:
    """
    Dataset 3: UC Berkeley Measuring Hate Speech
    Source: https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech
    CRITICAL: hate_speech_score is continuous — used directly as fuzzy Toxicity input
    """
    path = "data/raw/ucberkeley.parquet"
    if not os.path.exists(path):
        try:
            from datasets import load_dataset
            print("Downloading UC Berkeley dataset from HuggingFace...")
            ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")
            df = ds["train"].to_pandas()
            df.to_parquet(path, index=False)
        except Exception as e:
            print(f"[UCBerkeley] Could not download: {e}")
            return pd.DataFrame()
    else:
        df = pd.read_parquet(path)

    text_col = next((c for c in df.columns if "text" in c.lower() or "comment" in c.lower()), None)
    if text_col is None:
        print("[UCBerkeley] No text column. Skipping.")
        return pd.DataFrame()

    df["comment_text"] = df[text_col].apply(clean_text)

    # Continuous hate_speech_score: normalise to [0,1]
    score_col = "hate_speech_score" if "hate_speech_score" in df.columns else None
    if score_col:
        mn, mx = df[score_col].min(), df[score_col].max()
        df["toxicity_score"] = ((df[score_col] - mn) / (mx - mn + 1e-9)).clip(0, 1)
    else:
        df["toxicity_score"] = 0.5

    threat_col = "threat" if "threat" in df.columns else None
    if threat_col:
        mn, mx = df[threat_col].min(), df[threat_col].max()
        df["threat_score"] = ((df[threat_col] - mn) / (mx - mn + 1e-9)).clip(0, 1)
    else:
        df["threat_score"] = 0.0

    df["is_harmful"] = (df["toxicity_score"] > 0.5).astype(int)

    # Target gender
    gender_col = next((c for c in df.columns if "gender" in c.lower()), None)
    if gender_col:
        df["target_gender"] = df[gender_col].apply(
            lambda x: "female" if str(x).lower() in ["women","female","girl","girls"] else "mixed"
        )
    else:
        df["target_gender"] = "mixed"

    df["platform"]       = "Twitter"
    df["subreddit"]      = None
    df["hashtags"]       = df["comment_text"].apply(extract_hashtags)
    df["dataset_source"] = "UCBerkeley"

    print(f"[UCBerkeley] Loaded {len(df)} rows | harmful: {df['is_harmful'].sum()}")
    return df[["comment_text","platform","subreddit","hashtags","is_harmful",
               "toxicity_score","threat_score","dataset_source","target_gender"]]


def load_davidson() -> pd.DataFrame:
    """
    Dataset 4: Davidson Twitter Hate Speech
    Source: https://huggingface.co/datasets/tdavidson/hate_speech_offensive
    Labels: 0=hate speech, 1=offensive, 2=neither
    """
    path = "data/raw/davidson.parquet"
    if not os.path.exists(path):
        try:
            from datasets import load_dataset
            print("Downloading Davidson from HuggingFace...")
            ds = load_dataset("tdavidson/hate_speech_offensive")
            df = ds["train"].to_pandas()
            df.to_parquet(path, index=False)
        except Exception as e:
            print(f"[Davidson] Could not download: {e}")
            return pd.DataFrame()
    else:
        df = pd.read_parquet(path)

    text_col = next((c for c in df.columns if "tweet" in c.lower() or "text" in c.lower()), None)
    if text_col is None:
        print("[Davidson] No text column. Skipping.")
        return pd.DataFrame()

    df["comment_text"]   = df[text_col].apply(clean_text)
    class_col            = next((c for c in df.columns if "class" in c.lower()), "class")
    df["is_harmful"]     = safe_binary(df.get(class_col, pd.Series([2]*len(df))), [0, 1])
    df["toxicity_score"] = df.get(class_col, pd.Series([2]*len(df))).map({0: 0.90, 1: 0.60, 2: 0.05}).fillna(0.05)
    df["threat_score"]   = df.get(class_col, pd.Series([2]*len(df))).map({0: 0.70, 1: 0.30, 2: 0.02}).fillna(0.02)
    df["platform"]       = "Twitter"
    df["subreddit"]      = None
    df["hashtags"]       = df["comment_text"].apply(extract_hashtags)
    df["target_gender"]  = "mixed"
    df["dataset_source"] = "Davidson"

    print(f"[Davidson] Loaded {len(df)} rows | harmful: {df['is_harmful'].sum()}")
    return df[["comment_text","platform","subreddit","hashtags","is_harmful",
               "toxicity_score","threat_score","dataset_source","target_gender"]]


def load_kaggle_women() -> pd.DataFrame:
    """
    Dataset 5: Kaggle Women Harassment Dataset
    Place file at data/raw/women_harassment.csv
    """
    path = "data/raw/women_harassment.csv"
    if not os.path.exists(path):
        print("[Kaggle] women_harassment.csv not found. Skipping.")
        print("  → Download from: https://www.kaggle.com/datasets/thevincida/women-harassment-dataset")
        return pd.DataFrame()

    df = pd.read_csv(path)
    text_col  = next((c for c in df.columns if "text" in c.lower() or "comment" in c.lower() or "content" in c.lower()), None)
    label_col = next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), None)

    if text_col is None:
        print("[Kaggle] Could not identify text column.")
        return pd.DataFrame()

    df["comment_text"]   = df[text_col].apply(clean_text)
    df["is_harmful"]     = safe_binary(df.get(label_col, pd.Series([1]*len(df))), [1, "harassment", "Harassment", "1"])
    df["toxicity_score"] = df["is_harmful"] * 0.75
    df["threat_score"]   = df["is_harmful"] * 0.40
    df["platform"]       = "Mixed"
    df["subreddit"]      = None
    df["hashtags"]       = df["comment_text"].apply(extract_hashtags)
    df["target_gender"]  = "female"
    df["dataset_source"] = "KaggleWomen"

    print(f"[Kaggle] Loaded {len(df)} rows | harmful: {df['is_harmful'].sum()}")
    return df[["comment_text","platform","subreddit","hashtags","is_harmful",
               "toxicity_score","threat_score","dataset_source","target_gender"]]


# ─────────────────────────────────────────────────────────────
# MERGE & EXPORT
# ─────────────────────────────────────────────────────────────

def build_master_dataset() -> pd.DataFrame:
    """Load all datasets, merge, deduplicate, and export master CSV."""
    loaders = [load_edos, load_hatexplain, load_ucberkeley, load_davidson, load_kaggle_women]
    frames  = []
    for fn in loaders:
        df = fn()
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No datasets loaded. Check data/raw/ folder.")

    master = pd.concat(frames, ignore_index=True)

    # Remove exact-duplicate text rows (keep first)
    before = len(master)
    master.drop_duplicates(subset=["comment_text", "dataset_source"], inplace=True)
    print(f"\nDeduplication: {before} → {len(master)} rows removed {before - len(master)}")

    # Remove empty texts
    master = master[master["comment_text"].str.len() > 5].reset_index(drop=True)

    # Fill NaN scores with 0
    for col in ["toxicity_score", "threat_score"]:
        master[col] = master[col].fillna(0.0).clip(0.0, 1.0)

    out_path = "data/processed/master_dataset.csv"
    master.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"MASTER DATASET BUILT: {len(master)} total comments")
    print(f"Harmful: {master['is_harmful'].sum()} ({master['is_harmful'].mean()*100:.1f}%)")
    print(f"Platforms: {master['platform'].value_counts().to_dict()}")
    print(f"Datasets: {master['dataset_source'].value_counts().to_dict()}")
    print(f"Saved → {out_path}")
    print(f"{'='*60}\n")
    return master


if __name__ == "__main__":
    master = build_master_dataset()