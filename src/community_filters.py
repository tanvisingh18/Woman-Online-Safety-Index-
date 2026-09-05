"""
Community-level filters — exclude news coverage, support spaces, and victim-heavy threads
from harassment hotspot rankings (they discuss abuse; they are not attack hubs).

Run standalone: python src/community_filters.py
"""

from __future__ import annotations

import re

import pandas as pd

# YouTube/news channels that primarily report ON harassment rather than host it
NEWS_MEDIA_PATTERNS = re.compile(
    r"\b(news|guardian|bbc|sky news|trending|feed|reuters|cnn|nbc|abc news|"
    r"morning britain|documentary|report|investigation|journalism)\b",
    re.IGNORECASE,
)

NEWS_CHANNELS = {
    "bbc trending",
    "sky news",
    "the guardian",
    "guardian news",
    "sbs the feed",
    "good morning britain",
    "nate the lawyer",
    "brut india",
    "counterpoint",
    "steve shives",
    "jason schroeder",
    "huffpost",
    "pop trigger",
    "cosmopolitan",
    "tedx talks",
    "economics explained",
    "simple misfits",
    "dr. daf show",
    "bbc news",
    "vice news",
    "nowthis news",
    "the young turks",
}

# Advocacy / support — never rank as harassment hotspots (perpetrator density is not the story)
HOTSPOT_EXCLUDE_SUBREDDITS = {
    "twoxchromosomes",
    "feminism",
    "askwomen",
    "offmychest",
    "relationship_advice",
    "womenintech",
    "girlgamers",
    "workingmoms",
    "bisexual",
}

# Support / education spaces — high victim disclosure, not perpetrator hubs
SUPPORT_SUBREDDITS = {
    "twoxchromosomes",
    "askwomen",
    "offmychest",
    "relationship_advice",
    "feminism",
    "womenintech",
    "girlgamers",
    "workingmoms",
    "bisexual",
    "womenin tech",
}

GENERAL_EDU_SUBREDDITS = {
    "nostupidquestions",
    "askreddit",
    "explainlikeimfive",
    "todayilearned",
}

DISCUSSION_OF_ABUSE = re.compile(
    r"\b(i was (?:raped|assaulted|harassed)|happened to me|my abuser|"
    r"support (?:you|her|women)|i'?m sorry op|rape kit|backlog|law enforcement)\b",
    re.IGNORECASE,
)


def classify_community(
    community: str,
    platform: str,
    group: pd.DataFrame | None = None,
) -> tuple[bool, str]:
    """
    Return (exclude_from_hotspots, reason).
    """
    name = str(community or "").strip()
    low = name.lower()

    if low in ("edos_corpus", "twitter_microblog", "nan", ""):
        return True, "non_community_bucket"

    if platform == "YouTube" and (low in NEWS_CHANNELS or NEWS_MEDIA_PATTERNS.search(name)):
        return True, "news_media_channel"

    if platform == "Reddit":
        if low in GENERAL_EDU_SUBREDDITS:
            return True, "general_education_subreddit"
        if low in HOTSPOT_EXCLUDE_SUBREDDITS:
            return True, "advocacy_or_support_subreddit"
        if low in SUPPORT_SUBREDDITS and group is not None:
            perp = (group.get("harm_role", "") == "perpetrator_attack").mean()
            victim = (group.get("harm_role", "") == "victim_disclosure").mean()
            if victim > perp * 1.5:
                return True, "support_space_victim_discourse"

    if group is not None and len(group) >= 10:
        texts = group.get("comment_text", pd.Series(dtype=str)).astype(str)
        abuse_discussion_share = texts.str.contains(DISCUSSION_OF_ABUSE, regex=True, na=False).mean()
        perp_rate = (group.get("harm_role", "") == "perpetrator_attack").mean()
        if abuse_discussion_share > 0.35 and perp_rate < 0.12:
            return True, "abuse_topic_discussion_not_attacks"

    return False, ""


def filter_hotspot_communities(
    master_df: pd.DataFrame,
    community_col: str = "community",
    min_comments: int = 20,
) -> pd.DataFrame:
    """Score only communities that pass quality filters."""
    if community_col not in master_df.columns:
        return master_df

    kept = []
    for comm, grp in master_df.groupby(community_col):
        if len(grp) < min_comments:
            continue
        plat = grp["platform"].mode()[0] if "platform" in grp.columns else "Unknown"
        exclude, reason = classify_community(comm, plat, grp)
        if not exclude:
            kept.append(comm)

    return master_df[master_df[community_col].isin(kept)].copy()
