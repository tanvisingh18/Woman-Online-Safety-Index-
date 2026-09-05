"""Dataset split constants for platform-level WHSI/MRI analysis."""

from __future__ import annotations

# Live scrapes only — used for all platform comparison tables and rankings.
PLATFORM_ANALYSIS_SPLITS = frozenset({"live_scrape"})

# Davidson 2017 corpus — pipeline validation only, never ranked against live platforms.
VALIDATION_CORPUS_SPLIT = "historical_twitter"


def is_platform_analysis_split(split: str) -> bool:
    return str(split) in PLATFORM_ANALYSIS_SPLITS


def is_validation_corpus_split(split: str) -> bool:
    return str(split) == VALIDATION_CORPUS_SPLIT


def platform_analysis_mask(df) -> "pd.Series":
    import pandas as pd

    split = df.get("dataset_split", pd.Series([""] * len(df))).astype(str)
    src = df.get("dataset_source", pd.Series([""] * len(df))).astype(str)
    return split.isin(PLATFORM_ANALYSIS_SPLITS) & ~src.str.contains("EDOS", case=False, na=False)


def validation_corpus_mask(df) -> "pd.Series":
    import pandas as pd

    split = df.get("dataset_split", pd.Series([""] * len(df))).astype(str)
    return split.eq(VALIDATION_CORPUS_SPLIT)
