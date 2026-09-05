"""Consistent platform display names and proxy/historical flags for thesis outputs."""

from __future__ import annotations

import pandas as pd

PROXY_SOURCES = ("gab_proxy", "twitter_gab_proxy")


def is_gab_proxy_row(row: pd.Series) -> bool:
    if int(row.get("is_proxy", 0) or 0) == 1:
        return True
    src = str(row.get("dataset_source", "")).lower()
    return any(p in src for p in PROXY_SOURCES)


def is_historical_row(row: pd.Series) -> bool:
    if int(row.get("is_historical", 0) or 0) == 1:
        return True
    split = str(row.get("dataset_split", ""))
    src = str(row.get("dataset_source", "")).lower()
    return split == "historical_twitter" or "davidson" in src


def platform_is_proxy(platform: str, ci_row: pd.Series | None = None) -> bool:
    if platform != "Twitter":
        return False
    if ci_row is not None:
        if is_historical_row(ci_row):
            return False
        if int(ci_row.get("is_proxy_data", ci_row.get("is_proxy", 0)) or 0) == 1:
            return True
        src = str(ci_row.get("dataset_source", "")).lower()
        return any(p in src for p in PROXY_SOURCES)
    return False


def platform_is_historical(platform: str, row: pd.Series | None = None) -> bool:
    if platform != "Twitter":
        return False
    if row is not None:
        return is_historical_row(row)
    return False


def display_name(platform: str, is_proxy: bool = False, is_historical: bool = False) -> str:
    if platform == "Twitter" and is_proxy:
        return "Gab (Twitter-proxy)"
    if platform == "Twitter" and is_historical:
        return "Twitter (historical corpus)"
    return str(platform)


def add_display_columns(df: pd.DataFrame, proxy_col: str = "is_proxy_data") -> pd.DataFrame:
    out = df.copy()
    if "is_historical" not in out.columns:
        if "dataset_split" in out.columns:
            out["is_historical"] = (
                out["dataset_split"].astype(str).eq("historical_twitter")
                | out.get("dataset_source", "").astype(str).str.contains("davidson", case=False, na=False)
            ).astype(int)
        else:
            out["is_historical"] = 0

    if proxy_col in out.columns:
        out["is_proxy"] = out[proxy_col].fillna(0).astype(int)
    elif "is_proxy" in out.columns:
        out["is_proxy"] = out["is_proxy"].fillna(0).astype(int)
    elif "dataset_source" in out.columns:
        out["is_proxy"] = out["dataset_source"].astype(str).str.lower().apply(
            lambda s: 1 if any(p in s for p in PROXY_SOURCES) else 0
        )
    else:
        out["is_proxy"] = 0

    out["platform_display"] = out.apply(
        lambda r: display_name(
            r["platform"],
            bool(r.get("is_proxy", 0)),
            bool(r.get("is_historical", 0)),
        ),
        axis=1,
    )
    return out
