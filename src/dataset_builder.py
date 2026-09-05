"""
Build thesis-grade datasets with provenance splits.

Outputs:
  data/processed/master_women_relevant.csv   — main WHSI corpus (no Reddit 1M bulk)
  data/processed/master_background.csv       — optional bulk Reddit sample
  data/processed/master_dataset.csv          — symlink/copy of women_relevant for pipeline

Run:
  python src/dataset_builder.py
  python src/dataset_builder.py --label   # apply EDOS-trained gendered harm model
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

WOMEN_SUBREDDITS = {
    "TwoXChromosomes",
    "AskWomen",
    "feminism",
    "WomenInTech",
    "GirlGamers",
    "workingmoms",
    "offmychest",
}

STANDARD_COLS = [
    "comment_text",
    "platform",
    "community",
    "community_type",
    "subreddit",
    "hashtags",
    "is_harmful",
    "is_gendered_harm",
    "gendered_harm_proba",
    "toxicity_score",
    "threat_score",
    "harm_role",
    "targets_women",
    "directed_at_women",
    "is_proxy",
    "dataset_source",
    "dataset_split",
    "target_gender",
    "content_category",
    "thread_depth",
    "parent_comment_id",
    "score",
    "author",
    "created_utc",
    "video_id",
    "comment_id",
]


def _tier_to_toxicity(tier: str) -> float:
    m = {"none": 0.05, "low": 0.35, "medium": 0.65, "high": 0.90}
    return m.get(str(tier).lower().strip(), 0.40)


def _infer_youtube_thread(comment_id: str, parent_from_row=None) -> tuple[int, str]:
    """YouTube encodes replies as parent_id.reply_id in comment_id string."""
    cid = str(comment_id or "").strip()
    if parent_from_row and str(parent_from_row).strip() not in ("", "nan", "None"):
        return 1, str(parent_from_row)
    if "." in cid:
        parts = cid.split(".")
        depth = len(parts) - 1
        parent = ".".join(parts[:-1])
        return depth, parent
    return 0, ""


def load_youtube_women_relevant(scraped_dir: str = "data/scraped") -> pd.DataFrame:
    paths = [
        os.path.join(scraped_dir, "youtube_master_comments.csv"),
        os.path.join(scraped_dir, "youtube_comments_ytdlp.csv"),
        os.path.join(scraped_dir, "youtube_replies_ytdlp.csv"),
    ]
    records = []
    seen_ids = set()

    for path in paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, low_memory=False)
        text_col = "text" if "text" in df.columns else "comment_text"
        for _, row in df.iterrows():
            text = str(row.get(text_col, ""))
            if len(text.strip()) < 5:
                continue
            cid = str(row.get("comment_id", "")) or text[:80]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            tier = row.get("toxicity_tier", "none")
            kw = bool(row.get("harassment_keyword_hit", False))
            channel = str(row.get("channel_title", row.get("channel_id", "unknown")))
            depth, parent = _infer_youtube_thread(
                cid, row.get("parent_comment_id", row.get("parent_id"))
            )
            if depth == 0 and pd.notna(row.get("thread_depth")) and int(row.get("thread_depth", 0) or 0) > 0:
                depth = int(row["thread_depth"])
                parent = str(row.get("parent_comment_id", "") or "")

            records.append(
                {
                    "comment_text": text[:8000],
                    "platform": "YouTube",
                    "community": channel[:120],
                    "community_type": "youtube_channel",
                    "subreddit": channel[:120],
                    "hashtags": str(row.get("hashtags_in_text", "")),
                    "is_harmful": np.nan,
                    "is_gendered_harm": np.nan,
                    "toxicity_score": _tier_to_toxicity(tier) if kw or tier in ("medium", "high") else np.nan,
                    "threat_score": np.nan,
                    "dataset_source": str(row.get("data_source", "youtube_scraped")),
                    "dataset_split": "live_scrape",
                    "target_gender": "female",
                    "content_category": "reply" if depth > 0 else row.get("content_category", "comment"),
                    "thread_depth": depth,
                    "parent_comment_id": parent,
                    "score": pd.to_numeric(row.get("like_count", 0), errors="coerce"),
                    "author": row.get("author_display_name", ""),
                    "created_utc": row.get("published_at", row.get("created_utc", np.nan)),
                    "video_id": row.get("video_id", ""),
                    "comment_id": row.get("comment_id", ""),
                }
            )

    out = pd.DataFrame(records)
    print(f"[Dataset] YouTube women-relevant: {len(out)} rows | replies: {(out.get('thread_depth',0)>0).sum() if not out.empty else 0}")
    return out


def load_reddit_pullpush(path: str = "data/scraped/reddit_women_pullpush.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df = df[df["comment_text"].astype(str).str.len() > 5].copy()
    df["community"] = df.get("subreddit", "reddit").astype(str)
    df["community_type"] = "subreddit"
    df["platform"] = "Reddit"
    df["dataset_split"] = "live_scrape"
    df["women_priority_community"] = df["subreddit"].isin(WOMEN_SUBREDDITS).astype(int)
    print(f"[Dataset] Reddit Pullpush: {len(df)} rows | subs: {df['subreddit'].nunique()}")
    return df


def load_twitter_snscrape(path: str = "data/scraped/twitter_snscrape.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df = df[df["comment_text"].astype(str).str.len() > 5].copy()
    df["community"] = df.get("community", df.get("hashtags", "twitter_search")).astype(str)
    df["community_type"] = "hashtag_search"
    df["platform"] = "Twitter"
    df["dataset_split"] = "live_scrape"
    df["subreddit"] = df["community"]
    if "dataset_source" not in df.columns:
        df["dataset_source"] = "snscrape"
    df["is_proxy"] = 0
    print(f"[Dataset] Twitter snscrape: {len(df)} rows")
    return df


def load_gab_microblog(sample_n: int = 2500) -> pd.DataFrame:
    """Gab posts as Twitter/X stand-in when live X scrape unavailable."""
    paths = ["../gab_1M_unlabelled.csv", "data/raw/gab_1M_unlabelled.csv"]
    path = next((p for p in paths if os.path.exists(p)), None)
    if not path:
        return pd.DataFrame()

    df = pd.read_csv(path, usecols=["text"], nrows=200_000)
    df = df.rename(columns={"text": "comment_text"})
    df = df[df["comment_text"].astype(str).str.len() > 15]

    kw = (
        r"\b(?:woman|women|female|girl|she|her|bitch|slut|feminist|metoo|"
        r"harass|rape|sexist|misogyn)\b"
    )
    df = df[df["comment_text"].str.contains(kw, case=False, na=False)]
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)

    df["platform"] = "Twitter"
    df["community"] = df["comment_text"].str.extract(r"#(\w+)", expand=False).fillna("twitter_microblog")
    df["community_type"] = "hashtag_search"
    df["subreddit"] = df["community"]
    df["hashtags"] = df["comment_text"].str.findall(r"#(\w+)").apply(
        lambda tags: ",".join(tags[:5]) if isinstance(tags, list) else ""
    )
    df["dataset_source"] = "twitter_gab_proxy"
    df["dataset_split"] = "live_scrape"
    df["is_proxy"] = 1
    df["target_gender"] = "female"
    df["is_gendered_harm"] = np.nan
    df["is_harmful"] = np.nan
    print(f"[Dataset] Twitter (Gab proxy): {len(df)} rows — cite as proxy in thesis")
    return df


def load_twitter_tweepy(path: str = "data/scraped/twitter_live.parquet") -> pd.DataFrame:
    if not os.path.exists(path):
        csv_alt = path.replace(".parquet", ".csv")
        if os.path.exists(csv_alt):
            df = pd.read_csv(csv_alt, low_memory=False)
        else:
            return pd.DataFrame()
    else:
        df = pd.read_parquet(path)
    df = df[df["comment_text"].astype(str).str.len() > 5].copy()
    df["platform"] = "Twitter"
    df["dataset_split"] = "live_scrape"
    df["dataset_source"] = df.get("dataset_source", "Tweepy_live")
    df["is_proxy"] = 0
    if "community" not in df.columns:
        df["community"] = df.get("query_hashtag", df.get("hashtags", "twitter_live")).astype(str)
    df["community_type"] = "hashtag_search"
    df["subreddit"] = df["community"]
    print(f"[Dataset] Twitter live API: {len(df)} rows")
    return df


def load_twitter_historical(
    path: str = "data/scraped/twitter_historical.parquet",
    sample_n: int = 2500,
) -> pd.DataFrame:
    """Published Twitter corpus (Davidson 2017) — Path C, no X API."""
    if not os.path.exists(path):
        try:
            from twitter_historical import build_historical_twitter_corpus

            build_historical_twitter_corpus(sample_n=sample_n)
        except Exception as exc:
            print(f"[Dataset] Could not build historical Twitter corpus: {exc}")
            return pd.DataFrame()

    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df = df[df["comment_text"].astype(str).str.len() > 5].copy()
    df["platform"] = "Twitter"
    df["dataset_split"] = "historical_twitter"
    df["dataset_source"] = df.get("dataset_source", "davidson_2017_twitter")
    df["is_proxy"] = 0
    df["is_historical"] = 1
    if "community" not in df.columns:
        df["community"] = "twitter_historical"
    df["community_type"] = df.get("community_type", "historical_hashtag")
    df["subreddit"] = df["community"]
    print(f"[Dataset] Twitter historical (Davidson 2017): {len(df)} rows")
    return df


def load_twitter_corpus(sample_n: int = 2500) -> pd.DataFrame:
    """Priority: Tweepy live > snscrape > Davidson historical > Gab proxy (opt-in)."""
    live = load_twitter_tweepy()
    if not live.empty:
        return live
    sn = load_twitter_snscrape()
    if not sn.empty:
        sn["is_proxy"] = 0
        sn["is_historical"] = 0
        return sn
    historical = load_twitter_historical(sample_n=sample_n)
    if not historical.empty:
        return historical
    if os.environ.get("USE_GAB_PROXY", "").strip() == "1":
        proxy = load_gab_microblog(sample_n=sample_n)
        if not proxy.empty:
            proxy["is_proxy"] = 1
            proxy["is_historical"] = 0
        return proxy
    print("[Dataset] No Twitter corpus (set USE_GAB_PROXY=1 to enable Gab fallback)")
    return pd.DataFrame()


def load_reddit_harassment(path: str = "data/scraped/reddit_harassment_dataset.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[Dataset] Missing {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df = df[df["comment_text"].astype(str).str.len() > 5].copy()

    # Women-centric: priority subreddits + TwoX-heavy weighting
    df["community"] = df["subreddit"].astype(str)
    df["community_type"] = "subreddit"
    df["platform"] = "Reddit"
    df["dataset_split"] = "live_scrape"
    df["dataset_source"] = "reddit_harassment_dataset"
    df["is_harmful"] = np.nan
    df["is_gendered_harm"] = np.nan
    df["target_gender"] = "mixed"

    # Mark women-priority communities
    df["women_priority_community"] = df["subreddit"].isin(WOMEN_SUBREDDITS).astype(int)

    # Parse created_utc
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
        df["created_utc"] = df["created_utc"].astype("int64") // 10**9

    cols = [c for c in STANDARD_COLS if c in df.columns or c in df]
    extra = ["women_priority_community", "post_title", "permalink"]
    keep = list(dict.fromkeys(cols + [c for c in extra if c in df.columns]))
    print(f"[Dataset] Reddit harassment labelled scrape: {len(df)} rows")
    print(f"  Subreddits: {df['subreddit'].value_counts().head(8).to_dict()}")
    return df[keep]


def load_telegram(path: str = "data/scraped/telegram_messages.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    text_col = "text" if "text" in df.columns else "message"
    community_col = "channel" if "channel" in df.columns else "chat_title"

    out = pd.DataFrame(
        {
            "comment_text": df[text_col].astype(str),
            "platform": "Telegram",
            "community": df.get(community_col, "telegram_unknown").astype(str),
            "community_type": "telegram_group",
            "subreddit": df.get(community_col, "telegram_unknown").astype(str),
            "hashtags": df[text_col].astype(str).apply(
                lambda t: ",".join(re.findall(r"#(\w+)", t.lower()))
            ),
            "is_harmful": np.nan,
            "is_gendered_harm": np.nan,
            "toxicity_score": np.nan,
            "threat_score": np.nan,
            "dataset_source": "telegram_messages",
            "dataset_split": "live_scrape",
            "target_gender": "mixed",
            "score": pd.to_numeric(df.get("views", 0), errors="coerce"),
            "author": "",
            "created_utc": pd.to_datetime(df.get("date", np.nan), errors="coerce"),
            "comment_id": df.get("message_id", ""),
        }
    )
    out = out[out["comment_text"].str.len() > 5]
    # Unix timestamp
    out["created_utc"] = pd.to_datetime(out["created_utc"], errors="coerce")
    out["created_utc"] = out["created_utc"].astype("int64") // 10**9

    print(f"[Dataset] Telegram: {len(out)} rows | groups: {out['community'].nunique()}")
    return out


def load_edos_women_relevant(max_rows: int = 8000) -> pd.DataFrame:
    """Labelled EDOS Reddit/Gab sexism — ground-truth anchor for women-targeted harm."""
    paths = [
        "data/labelled/edos_labelled_aggregated.csv",
        "../edos_labelled_aggregated.csv",
    ]
    path = next((p for p in paths if os.path.exists(p)), None)
    if not path:
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df.rename(columns={"text": "comment_text"})
    df = df[df["comment_text"].astype(str).str.len() > 10]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    is_sexist = df["label_sexist"].astype(str).str.lower() == "sexist"
    cat = df.get("label_category", pd.Series([""] * len(df))).astype(str).str.lower()

    df["platform"] = "Reddit"
    df["community"] = "EDOS_corpus"
    df["community_type"] = "subreddit"
    df["subreddit"] = "EDOS_corpus"
    df["is_gendered_harm"] = is_sexist.astype(int)
    df["is_harmful"] = is_sexist.astype(int)
    df["gendered_harm_proba"] = is_sexist.astype(float)
    df["toxicity_score"] = cat.apply(
        lambda c: 0.90 if any(x in c for x in ("derogation", "threat", "animosity")) else 0.45
    )
    df["threat_score"] = cat.apply(lambda c: 0.90 if "threat" in c else 0.15)
    df["harm_role"] = np.where(is_sexist, "perpetrator_attack", "neutral_discourse")
    df["targets_women"] = 1
    df["dataset_source"] = "EDOS"
    df["dataset_split"] = "women_relevant_labelled"
    df["target_gender"] = "female"

    print(f"[Dataset] EDOS labelled anchor: {len(df)} rows | sexist rate {is_sexist.mean():.1%}")
    return df[[c for c in STANDARD_COLS if c in df.columns]]


def load_reddit_background(sample_n: int = 10_000) -> pd.DataFrame:
    """Bulk Reddit 1M — background only, NOT for main WHSI."""
    path = "../reddit_1M_unlabelled.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=["text"], nrows=sample_n * 3)
    df = df.rename(columns={"text": "comment_text"})
    df = df[df["comment_text"].astype(str).str.len() > 5]
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)
    df["platform"] = "Reddit"
    df["community"] = "reddit_bulk_sample"
    df["community_type"] = "subreddit"
    df["subreddit"] = "reddit_bulk_sample"
    df["dataset_split"] = "background"
    df["dataset_source"] = "reddit_1M_unlabelled"
    print(f"[Dataset] Reddit background (excluded from main WHSI): {len(df)} rows")
    return df


# Severity lexicon — tags harsh/sexual/misogynistic language (for filtering & reporting)
SEVERE_PATTERNS = re.compile(
    r"\b(?:fuck|fucking|shit|bitch|whore|slut|cunt|rape|raped|molest|"
    r"kill her|deserve to die|stupid woman|dumb woman|all women|women are|"
    r"feminazi|misogyn|catcall|assaulted|nude|naked|horny|sexual harassment|"
    r"send nudes|suck my|bend over|rape her|bitches)\b",
    re.IGNORECASE,
)


def tag_content_severity(df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
    """Flag rows with explicit/harsh women-targeted language."""
    df = df.copy()
    texts = df[text_col].astype(str)
    df["has_severe_language"] = texts.str.contains(SEVERE_PATTERNS, regex=True, na=False).astype(int)
    df["content_severity"] = np.where(
        df["has_severe_language"] == 1,
        "severe",
        np.where(df.get("is_gendered_harm", 0) == 1, "gendered_moderate", "standard"),
    )
    return df


def load_reddit_live_search(path: str = "data/scraped/reddit_live_search.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df = df[df["comment_text"].astype(str).str.len() > 5].copy()
    df["platform"] = "Reddit"
    df["dataset_split"] = "live_scrape"
    print(f"[Dataset] Reddit live search: {len(df)} rows")
    return df


def build_live_scrapes_only(label: bool = True) -> pd.DataFrame:
    """
    Master built ONLY from genuine platform scrapes — no EDOS, no Gab, no bulk Reddit 1M.
    Sources: your YouTube/Reddit/Telegram uploads + Pullpush live Reddit + Twitter snscrape.
    """
    os.makedirs("data/processed", exist_ok=True)

    twitter_df = load_twitter_corpus()

    frames = [
        load_youtube_women_relevant(),
        load_reddit_harassment(),
        load_reddit_pullpush(),
        load_reddit_live_search(),
        twitter_df,
        load_telegram(),
    ]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError("No live scrape files found in data/scraped/")

    master = pd.concat(frames, ignore_index=True, sort=False)
    master.drop_duplicates(subset=["comment_text", "platform", "dataset_source"], inplace=True)
    master = master[master["comment_text"].astype(str).str.len() > 5]
    hist_mask = master.get("dataset_split", "").astype(str) == "historical_twitter"
    master.loc[~hist_mask, "dataset_split"] = "live_scrape"

    if label:
        from gendered_harm_model import train_models, label_dataframe

        if not os.path.exists("models/harm_models/edos_sexist_pipeline.joblib"):
            train_models()
        labelled = label_dataframe(master.copy())
        master = labelled

    master["whsi_harm_flag"] = master["is_gendered_harm"].fillna(0).astype(int)

    out_path = "data/processed/master_live_scrapes.csv"
    master.to_csv(out_path, index=False)
    master.to_csv("data/processed/master_dataset.csv", index=False)
    master.to_csv("data/processed/master_women_relevant.csv", index=False)

    print(f"\n{'='*60}")
    print(f"LIVE SCRAPES ONLY: {len(master)} rows (NO EDOS / NO Gab proxy)")
    print(master["platform"].value_counts().to_string())
    print(f"By source:\n{master['dataset_source'].value_counts().head(10).to_string()}")
    print(f"Gendered harm rate: {master['is_gendered_harm'].mean():.2%}")
    print(f"Saved → {out_path}")
    print(f"{'='*60}\n")
    return master


def build_women_relevant(
    label: bool = True,
    live_only: bool = False,
    force_detoxify: bool = False,
    use_detoxify: bool = True,
) -> pd.DataFrame:
    if live_only:
        return build_live_scrapes_only(label=label)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/labelled", exist_ok=True)

    # Copy EDOS to labelled folder if needed
    if not os.path.exists("data/labelled/edos_labelled_aggregated.csv"):
        for src in ["../edos_labelled_aggregated.csv", "data/raw/edos_labelled_aggregated.csv"]:
            if os.path.exists(src):
                import shutil
                shutil.copy(src, "data/labelled/edos_labelled_aggregated.csv")
                break

    twitter_df = load_twitter_corpus()

    # MIXED MASTER = live platform scrapes + labelled EDOS anchor (no Reddit 1M bulk)
    frames = [
        # --- LIVE SCRAPES (real apps) ---
        load_youtube_women_relevant(),
        load_reddit_harassment(),
        load_reddit_pullpush(),
        load_reddit_live_search(),
        twitter_df,
        load_telegram(),
        # --- EXISTING LABELLED (classifier training anchor) ---
        load_edos_women_relevant(),
    ]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError("No women-relevant data sources found.")

    master = pd.concat(frames, ignore_index=True, sort=False)
    master.drop_duplicates(subset=["comment_text", "platform", "dataset_source"], inplace=True)
    master = master[master["comment_text"].astype(str).str.len() > 5]

    if label:
        from gendered_harm_model import train_models
        from hybrid_classifier import label_dataframe

        if not os.path.exists("models/harm_models/edos_sexist_pipeline.joblib"):
            train_models()
        needs_label = master["is_gendered_harm"].isna() | master["is_gendered_harm"].isnull()
        from corpus_config import PLATFORM_ANALYSIS_SPLITS

        analysis_mask = master.get("dataset_split", "").astype(str).isin(PLATFORM_ANALYSIS_SPLITS)
        if analysis_mask.any():
            unlabeled_live = analysis_mask & needs_label
        else:
            unlabeled_live = pd.Series(False, index=master.index)

        def _merge_labels(idx, labelled_part):
            for col in labelled_part.columns:
                if col not in master.columns:
                    master[col] = np.nan
                master.loc[idx, col] = labelled_part[col].values

        if force_detoxify and analysis_mask.any():
            print(f"[Label] Force Detoxify hybrid relabel on {analysis_mask.sum()} platform rows…")
            labelled_part = label_dataframe(master.loc[analysis_mask].copy(), use_detoxify=use_detoxify)
            _merge_labels(analysis_mask, labelled_part)
        elif needs_label.any() or unlabeled_live.any():
            edos_mask = master.get("dataset_source", "") == "EDOS"
            live_idx = (needs_label & ~edos_mask) | unlabeled_live
            if live_idx.any():
                print(f"[Label] Hybrid labelling {live_idx.sum()} rows (detoxify={use_detoxify})…")
                labelled_part = label_dataframe(master.loc[live_idx].copy(), use_detoxify=use_detoxify)
                _merge_labels(live_idx, labelled_part)

    master["whsi_harm_flag"] = master["is_gendered_harm"].fillna(master.get("is_harmful")).fillna(0).astype(int)
    master = tag_content_severity(master)

    out_path = "data/processed/master_women_relevant.csv"
    master.to_csv(out_path, index=False)
    master.to_csv("data/processed/master_dataset.csv", index=False)

    bg = load_reddit_background()
    if not bg.empty:
        bg.to_csv("data/processed/master_background.csv", index=False)

    live = master[master.get("dataset_split", "") == "live_scrape"]
    severe = master[master["has_severe_language"] == 1]

    print(f"\n{'='*60}")
    print(f"MIXED MASTER: {len(master)} rows (live scrapes + EDOS anchor)")
    print(master["platform"].value_counts().to_string())
    print(f"\nBy source:\n{master['dataset_source'].value_counts().head(12).to_string()}")
    print(f"\nLive scrapes only: {len(live)} rows")
    print(f"Severe language flagged: {len(severe)} ({100*len(severe)/len(master):.1f}%)")
    print(f"Gendered harm rate: {master['is_gendered_harm'].mean():.2%}")
    print(f"Perpetrator attacks: {(master.get('harm_role','')=='perpetrator_attack').sum()}")
    print(f"Saved → {out_path}")
    print(f"{'='*60}\n")

    stale_parquet = "data/processed/master_with_influence.parquet"
    if os.path.exists(stale_parquet):
        os.remove(stale_parquet)
        print(f"[Dataset] Removed stale {stale_parquet} — re-run influence step to regenerate")

    return master


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--live-only", action="store_true", help="Only genuine scrapes, no EDOS/Gab")
    parser.add_argument("--label", action="store_true", default=True)
    parser.add_argument("--no-label", action="store_true")
    parser.add_argument(
        "--force-detoxify",
        action="store_true",
        help="Re-run EDOS+Detoxify hybrid on all live_scrape rows",
    )
    parser.add_argument(
        "--no-detoxify",
        action="store_true",
        help="EDOS gendered harm only (fast; skip Detoxify pass)",
    )
    args = parser.parse_args()
    build_women_relevant(
        label=not args.no_label,
        live_only=args.live_only,
        force_detoxify=args.force_detoxify,
        use_detoxify=not args.no_detoxify,
    )
