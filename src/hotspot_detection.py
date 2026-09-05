"""
SECTION 7 (original) + SECTION 9 EXTENSION — HOTSPOT DETECTION
Women Safety Index | hotspot_detection.py

Computes WHSI at the sub-community level (subreddit / hashtag cluster)
to identify which specific communities within a platform are driving
the platform's overall danger score.

Features:
  - Per-subreddit/community WHSI using same fuzzy engine
  - Time-window WHSI — rolling 7-day WHSI to detect spikes
  - Escalation Score — slope of WHSI trend over time
  - Top-10 Hotspot Leaderboard per platform
  - Emerging hotspot alert when community spikes above threshold

Output:
  outputs/results/community_whsi.csv
  outputs/results/hotspot_leaderboard.csv
  outputs/plots/hotspot_heatmap.png
  outputs/plots/whsi_timeseries.png

Run: python src/hotspot_detection.py
"""

import os, warnings
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.makedirs("outputs/results", exist_ok=True)
os.makedirs("outputs/plots",   exist_ok=True)

# Local imports
import sys; sys.path.insert(0, os.path.dirname(__file__))
from fuzzy_engine        import compute_whsi
from feature_extraction  import (compute_toxicity, compute_threat,
                                  compute_frequency, compute_normalization,
                                  _harm_column)
from community_filters   import classify_community, filter_hotspot_communities


def _community_column(df: pd.DataFrame) -> str:
    if "community" in df.columns and df["community"].notna().any():
        return "community"
    return "subreddit"


def _hotspot_corpus(df: pd.DataFrame, perpetrator_only: bool = True) -> pd.DataFrame:
    """Perpetrator attacks only for hotspot harm scoring."""
    if perpetrator_only and "harm_role" in df.columns:
        return df[df["harm_role"] == "perpetrator_attack"].copy()
    return df.copy()


# ─────────────────────────────────────────────────────────────
# COMMUNITY-LEVEL WHSI
# ─────────────────────────────────────────────────────────────

def compute_community_whsi(master_df: pd.DataFrame,
                            community_col: str | None = None,
                            min_comments: int = 100,
                            perpetrator_only: bool = True,
                            save: bool = True) -> pd.DataFrame:
    """
    Compute WHSI for each sub-community (subreddit/hashtag group).
    Returns DataFrame sorted by WHSI descending.

    Parameters
    ----------
    master_df     : preprocessed DataFrame with all comments
    community_col : column to group by (subreddit / hashtag / cluster)
    min_comments  : minimum comments for scored hotspots (Faculty July Step 7: default 100)
    """
    master_df = _hotspot_corpus(master_df, perpetrator_only=perpetrator_only)
    # Exclude labelled training corpora from live hotspot leaderboard
    if "dataset_source" in master_df.columns:
        master_df = master_df[~master_df["dataset_source"].astype(str).str.contains("EDOS", na=False)]
    if "community" in master_df.columns:
        master_df = master_df[master_df["community"].astype(str) != "EDOS_corpus"]
    if community_col is None:
        community_col = _community_column(master_df)

    master_df = filter_hotspot_communities(master_df, community_col=community_col, min_comments=min_comments)

    if community_col not in master_df.columns:
        print(f"[Hotspot] Column '{community_col}' not found. Cannot compute community WHSI.")
        return pd.DataFrame()

    records = []
    harm_col = _harm_column(master_df)
    groups  = master_df[master_df[community_col].notna()].groupby(community_col)

    for community, group in groups:
        if len(group) < min_comments:
            continue

        T  = compute_toxicity(group)
        Th = compute_threat(group)
        F  = compute_frequency(group)
        N  = compute_normalization(group)

        score, cat = compute_whsi(T, Th, F, N)

        # Platform attribution (majority platform in this community)
        platform = group["platform"].mode()[0] if "platform" in group.columns else "Unknown"

        records.append({
            "community":    community,
            "platform":     platform,
            "WHSI":         score,
            "category":     cat,
            "toxicity":     T,
            "threat":       Th,
            "frequency":    F,
            "normalization":N,
            "n_comments":   len(group),
            "harm_rate":    round(group[harm_col].mean(), 4),
            "gendered_harm_rate": round(
                group.get("is_gendered_harm", group[harm_col]).mean(), 4
            ),
        })

    if not records:
        print("[Hotspot] No communities met minimum comment threshold.")
        return pd.DataFrame()

    df_out = pd.DataFrame(records).sort_values("WHSI", ascending=False)
    if save:
        out_path = "outputs/results/community_whsi.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[Hotspot] {len(df_out)} communities scored. Saved → {out_path}")
    return df_out


# ─────────────────────────────────────────────────────────────
# TOP-10 HOTSPOT LEADERBOARD
# ─────────────────────────────────────────────────────────────

def build_hotspot_leaderboard(community_df: pd.DataFrame,
                               top_n: int = 10) -> pd.DataFrame:
    """
    Return top-N most dangerous communities overall.
    Also outputs per-platform top-3.
    """
    if community_df.empty:
        return pd.DataFrame()

    leaderboard = community_df.head(top_n).copy()
    leaderboard["rank"] = range(1, len(leaderboard) + 1)

    out_path = "outputs/results/hotspot_leaderboard.csv"
    leaderboard.to_csv(out_path, index=False)

    print(f"\n{'='*65}")
    print(f"🔥 TOP {top_n} HARASSMENT HOTSPOTS")
    print(f"{'='*65}")
    print(f"{'Rank':<5} {'Community':<25} {'Platform':<12} {'WHSI':>6} {'Category':<20}")
    print("-"*65)
    for _, r in leaderboard.iterrows():
        print(f"{r['rank']:<5} {str(r['community']):<25} {r['platform']:<12} {r['WHSI']:>6.1f}  {r['category']}")
    print("="*65)

    # Per-platform top-3
    print("\nPer-Platform Top 3:")
    for plat, grp in community_df.groupby("platform"):
        top3 = grp.head(3)
        print(f"\n  {plat}:")
        for _, r in top3.iterrows():
            print(f"    • {r['community']} — WHSI {r['WHSI']:.1f} ({r['category']})")

    return leaderboard


# ─────────────────────────────────────────────────────────────
# TIME-WINDOW WHSI — Rolling spike detection
# ─────────────────────────────────────────────────────────────

def compute_time_window_whsi(master_df:   pd.DataFrame,
                              community:   str,
                              community_col: str = "subreddit",
                              window_days: int = 7,
                              min_comments: int = 10) -> pd.DataFrame:
    """
    Compute rolling WHSI in window_days-day windows for a given community.
    Detects spikes and flags emerging hotspots.

    Requires 'created_utc' column (Unix timestamp).
    """
    if "created_utc" not in master_df.columns:
        print("[Hotspot Time] 'created_utc' column required for time-window analysis.")
        return pd.DataFrame()

    subset = master_df[master_df[community_col] == community].copy()
    if len(subset) < min_comments:
        return pd.DataFrame()

    # Some sources store created_utc as ISO string or have missing timestamps.
    # We support both Unix seconds and parseable datetime strings.
    try:
        subset["created_dt"] = pd.to_datetime(subset["created_utc"], unit="s", utc=True, errors="coerce")
    except Exception:
        subset["created_dt"] = pd.to_datetime(subset["created_utc"], utc=True, errors="coerce")
    subset = subset.sort_values("created_dt")

    start = subset["created_dt"].min()
    end   = subset["created_dt"].max()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()
    windows = pd.date_range(start=start, end=end, freq=f"{window_days}D")

    records = []
    for w_start in windows:
        w_end = w_start + pd.Timedelta(days=window_days)
        window_data = subset[(subset["created_dt"] >= w_start) &
                              (subset["created_dt"] <  w_end)]
        if len(window_data) < min_comments:
            continue
        T  = compute_toxicity(window_data)
        Th = compute_threat(window_data)
        F  = compute_frequency(window_data)
        N  = compute_normalization(window_data)
        score, cat = compute_whsi(T, Th, F, N)
        records.append({
            "community":  community,
            "window_start": w_start.date(),
            "window_end":   w_end.date(),
            "WHSI":         score,
            "category":     cat,
            "n_comments":   len(window_data),
        })

    if not records:
        return pd.DataFrame()

    df_ts = pd.DataFrame(records)
    df_ts["escalation"] = df_ts["WHSI"].diff().fillna(0)

    # Alert threshold: spike if WHSI increases by >15 points in one window
    alert_threshold = 15.0
    df_ts["spike_alert"] = df_ts["escalation"] > alert_threshold
    return df_ts


def compute_escalation_score(time_series_df: pd.DataFrame) -> float:
    """
    Fit linear trend to WHSI over time.
    Positive slope = escalating community danger.
    Returns slope (points per window).
    """
    if time_series_df.empty or len(time_series_df) < 3:
        return 0.0
    x = np.arange(len(time_series_df))
    y = time_series_df["WHSI"].values
    slope = float(np.polyfit(x, y, deg=1)[0])
    return round(slope, 3)


# ─────────────────────────────────────────────────────────────
# EMERGING HOTSPOT ALERT
# ─────────────────────────────────────────────────────────────

def detect_emerging_hotspots_from_df(
    master_df: pd.DataFrame,
    community_df: pd.DataFrame,
    community_col: str = "subreddit",
    top_n: int = 10,
) -> list:
    """Flag spikes among already-scored communities without rewriting community_whsi.csv."""
    if community_df.empty:
        return []
    top_communities = community_df["community"].head(top_n).tolist()
    alerts = []
    for comm in top_communities:
        col = community_col
        if "community" in master_df.columns and comm in set(master_df["community"].astype(str)):
            col = "community"
        elif "subreddit" in master_df.columns:
            col = "subreddit"
        ts = compute_time_window_whsi(master_df, comm, col)
        if ts.empty:
            continue
        esc = compute_escalation_score(ts)
        recent_spike = ts["spike_alert"].iloc[-3:].any() if len(ts) >= 3 else False
        if recent_spike or esc > 2.0:
            alerts.append(
                {
                    "community": comm,
                    "current_WHSI": ts["WHSI"].iloc[-1],
                    "escalation_slope": esc,
                    "spike_detected": recent_spike,
                }
            )
    if alerts:
        print(f"\n⚠️  EMERGING HOTSPOT ALERTS ({len(alerts)} communities)")
        for a in alerts:
            print(
                f"  ⚡ {a['community']:25s}  WHSI={a['current_WHSI']:.1f}  "
                f"slope={a['escalation_slope']:+.2f}  "
                f"spike={'YES' if a['spike_detected'] else 'no'}"
            )
    else:
        print("[Hotspot] No emerging hotspot alerts at this time.")
    return alerts


def detect_emerging_hotspots(master_df: pd.DataFrame,
                               community_col: str = "subreddit",
                               top_n: int = 10,
                               spike_threshold: float = 15.0) -> list:
    """
    For each of the top-N hotspot communities, compute time-series WHSI
    and flag those with recent upward spikes.

    Returns list of dicts with community name, current WHSI, escalation.
    """
    community_df = compute_community_whsi(master_df, community_col, save=False)
    return detect_emerging_hotspots_from_df(master_df, community_df, community_col, top_n)


# ─────────────────────────────────────────────────────────────
# VISUALISATION — WHSI Heatmap
# ─────────────────────────────────────────────────────────────

def plot_hotspot_heatmap(community_df: pd.DataFrame,
                          save_path: str = "outputs/plots/hotspot_heatmap.png",
                          top_n: int = 20):
    """
    Heatmap of [Toxicity, Threat, Frequency, Normalization, WHSI]
    for top-N communities.
    """
    if community_df.empty:
        print("[Hotspot Plot] No data to plot.")
        return

    top = community_df.head(top_n).copy()
    top.index = top["community"].str[:20]

    heat_cols = ["toxicity","threat","frequency","normalization","WHSI"]
    heat_data = top[heat_cols]

    plt.figure(figsize=(11, max(5, top_n // 2)))
    sns.heatmap(
        heat_data, annot=True, fmt=".1f",
        cmap="RdYlGn_r", linewidths=0.4,
        vmin=0, vmax=100,
        cbar_kws={"label": "Score (0–100)"},
        annot_kws={"size": 8}
    )
    plt.title(f"Top {top_n} Harassment Hotspots — Dimension Breakdown",
              fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("")
    plt.ylabel("Community", fontsize=10)
    plt.xticks(fontsize=10, fontweight="bold")
    plt.yticks(fontsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Hotspot heatmap saved → {save_path}")


def plot_whsi_timeseries(ts_df: pd.DataFrame,
                          community: str,
                          save_path: str = None):
    """Plot WHSI over time for a single community."""
    if ts_df.empty:
        return

    save_path = save_path or f"outputs/plots/timeseries_{community[:20]}.png"
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(range(len(ts_df)), ts_df["WHSI"], "o-",
            color="#E53935", linewidth=2, markersize=6)

    # Shade spike windows
    spikes = ts_df[ts_df["spike_alert"]]
    for idx in spikes.index:
        pos = ts_df.index.get_loc(idx)
        ax.axvspan(pos - 0.5, pos + 0.5, color="#FFCDD2", alpha=0.5)

    ax.axhline(75, color="#B71C1C", linestyle="--", alpha=0.6, label="Critical threshold (75)")
    ax.axhline(50, color="#FB8C00", linestyle="--", alpha=0.6, label="Unsafe threshold (50)")

    # Trend line (skip if insufficient / degenerate data)
    x = np.arange(len(ts_df))
    whsi_vals = pd.to_numeric(ts_df["WHSI"], errors="coerce").fillna(0).values
    if len(ts_df) >= 2 and np.nanstd(whsi_vals) > 1e-6:
        try:
            z = np.polyfit(x, whsi_vals, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "--", color="#1565C0", linewidth=1.5,
                    label=f"Trend (slope={z[0]:+.2f})")
        except (np.linalg.LinAlgError, ValueError):
            pass

    ax.set_xticks(range(len(ts_df)))
    ax.set_xticklabels([str(d) for d in ts_df["window_start"]], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("WHSI Score", fontsize=11)
    ax.set_title(f"WHSI Time-Series: r/{community}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Time-series plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_hotspots_for_all_platforms(master: pd.DataFrame):
    """Hotspots per live platform: Reddit→subreddit, YouTube→channel, Telegram→group.
    Twitter/Davidson excluded from scored hotspots (Faculty quarantine).
    """
    comm_col = _community_column(master)
    configs = [
        ("Reddit", comm_col),
        ("YouTube", comm_col),
        ("Telegram", comm_col),
    ]
    all_comm = []
    for plat, col in configs:
        sub = master[master["platform"].str.lower() == plat.lower()] if "platform" in master.columns else master
        if sub.empty:
            continue
        # Live scrape only for scored hotspots
        if "dataset_split" in sub.columns:
            sub = sub[sub["dataset_split"].astype(str) == "live_scrape"]
        cdf = compute_community_whsi(sub, community_col=col, min_comments=100, save=False)
        if not cdf.empty:
            cdf["hotspot_platform"] = plat
            all_comm.append(cdf)
    if all_comm:
        out = pd.concat(all_comm, ignore_index=True).sort_values("WHSI", ascending=False)
        out.to_csv("outputs/results/community_whsi.csv", index=False)
        print(f"[Hotspot] {len(out)} live communities scored (n≥100). Saved → outputs/results/community_whsi.csv")
        return out
    return pd.DataFrame()


if __name__ == "__main__":
    master = pd.read_csv("data/processed/master_dataset.csv")

    # Multi-platform hotspots (live scrape, n≥100)
    community_df = run_hotspots_for_all_platforms(master)
    if community_df.empty:
        community_df = compute_community_whsi(master, community_col="subreddit")
    if not community_df.empty:
        build_hotspot_leaderboard(community_df, top_n=10)
        plot_hotspot_heatmap(community_df, top_n=15)
        # Persist primary table again after any intermediate writes
        community_df.to_csv("outputs/results/community_whsi.csv", index=False)

    # Time-series for top community
    if not community_df.empty and "created_utc" in master.columns:
        top_comm = community_df.iloc[0]["community"]
        ts_df    = compute_time_window_whsi(master, top_comm)
        if not ts_df.empty:
            plot_whsi_timeseries(ts_df, top_comm)

    # Emerging hotspot detection on scored live communities only (do not overwrite CSV)
    if not community_df.empty:
        detect_emerging_hotspots_from_df(master, community_df)