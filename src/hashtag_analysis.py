"""
SECTION 11 — HASHTAG ANALYSIS
Women Safety Index | src/hashtag_analysis.py

Measures:
  1. Hashtag Danger Score  — WHSI of comments containing each hashtag
  2. Prevalence / Harm Ratio
  3. Co-occurrence Network — which hashtags cluster together in harmful content
  4. Velocity             — how fast a hashtag grows in harmful context
  5. Semantic Drift       — embedding shift over time toward harassment

Output:
  outputs/results/hashtag_danger.csv
  outputs/plots/hashtag_cooccurrence.png
  outputs/plots/hashtag_danger_bar.png

Run: python src/hashtag_analysis.py
"""

import os, re, warnings
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings("ignore")
os.makedirs("outputs/results", exist_ok=True)
os.makedirs("outputs/plots",   exist_ok=True)

import sys; sys.path.insert(0, os.path.dirname(__file__))
from fuzzy_engine       import compute_whsi
from feature_extraction import (compute_toxicity, compute_threat,
                                 compute_frequency, compute_normalization)


# ─────────────────────────────────────────────────────────────
# EXTRACT HASHTAGS
# ─────────────────────────────────────────────────────────────

def extract_all_hashtags(df: pd.DataFrame,
                          text_col: str = "comment_text") -> pd.DataFrame:
    """
    Extract all hashtags from text column.
    Returns long-format DataFrame: row per (hashtag, row_index).
    Also parses pre-extracted 'hashtags' column if present.
    """
    records = []

    for idx, row in df.iterrows():
        # 1. Parse from text
        tags_from_text = re.findall(r"#(\w+)", str(row.get(text_col, "")).lower())

        # 2. Also use pre-extracted 'hashtags' column if available
        tags_from_col = []
        if "hashtags" in row and pd.notna(row["hashtags"]) and row["hashtags"]:
            tags_from_col = [t.strip().lower() for t in str(row["hashtags"]).split(",") if t.strip()]

        all_tags = list(set(tags_from_text + tags_from_col))
        for tag in all_tags:
            if tag and len(tag) > 1:   # filter single-char noise
                records.append({"hashtag": tag, "row_idx": idx})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# DANGER SCORE PER HASHTAG
# ─────────────────────────────────────────────────────────────

def compute_hashtag_danger(master_df: pd.DataFrame,
                            min_occurrences: int = 10) -> pd.DataFrame:
    """
    For each hashtag: compute WHSI of comments containing it.

    Parameters
    ----------
    master_df        : enriched master dataset
    min_occurrences  : minimum comments to include a hashtag

    Returns
    -------
    DataFrame sorted by danger_score descending
    """
    tag_map = extract_all_hashtags(master_df)

    if tag_map.empty:
        print("[Hashtag] No hashtags found in dataset.")
        return pd.DataFrame()

    results = []
    grouped = tag_map.groupby("hashtag")

    for tag, group in grouped:
        if len(group) < min_occurrences:
            continue

        subset = master_df.loc[master_df.index.intersection(group["row_idx"])]
        if len(subset) == 0:
            continue

        T  = compute_toxicity(subset)
        Th = compute_threat(subset)
        F  = compute_frequency(subset)
        N  = compute_normalization(subset)

        score, cat = compute_whsi(T, Th, F, N)

        harm_ratio      = subset["is_harmful"].mean()
        n_harmful       = int(subset["is_harmful"].sum())
        platforms       = subset["platform"].value_counts().to_dict()
        dominant_plat   = subset["platform"].mode()[0] if "platform" in subset.columns else "Unknown"

        results.append({
            "hashtag":        tag,
            "danger_score":   score,
            "category":       cat,
            "harm_ratio":     round(float(harm_ratio), 3),
            "n_comments":     len(subset),
            "n_harmful":      n_harmful,
            "occurrences":    len(group),
            "toxicity":       T,
            "threat":         Th,
            "frequency":      F,
            "normalization":  N,
            "dominant_platform": dominant_plat,
            "platforms":      str(platforms),
        })

    if not results:
        print("[Hashtag] No hashtags met minimum occurrence threshold.")
        return pd.DataFrame()

    df_out = pd.DataFrame(results).sort_values("danger_score", ascending=False)
    out_path = "outputs/results/hashtag_danger.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\n[Hashtag Danger] Top 10 most dangerous hashtags:")
    print(df_out.head(10)[["hashtag","danger_score","category","harm_ratio","n_comments"]].to_string(index=False))
    print(f"\nSaved {len(df_out)} hashtags → {out_path}")
    return df_out


# ─────────────────────────────────────────────────────────────
# HASHTAG CO-OCCURRENCE NETWORK
# ─────────────────────────────────────────────────────────────

def build_hashtag_cooccurrence_graph(master_df: pd.DataFrame,
                                      top_n:     int = 50,
                                      save_path: str = "outputs/plots/hashtag_cooccurrence.png") -> nx.Graph:
    """
    Build undirected weighted co-occurrence graph of hashtags in harmful comments.
    Nodes = hashtags. Edges = appear together in same comment. Weight = count.
    """
    harmful = master_df[master_df["is_harmful"] == 1].copy()
    G       = nx.Graph()

    for _, row in harmful.iterrows():
        # Get all hashtags for this comment
        tags_text = re.findall(r"#(\w+)", str(row.get("comment_text","")).lower())
        tags_col  = [t.strip().lower() for t in str(row.get("hashtags","")).split(",") if t.strip()]
        tags      = list(set(tags_text + tags_col))
        tags      = [t for t in tags if len(t) > 1]

        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = sorted([tags[i], tags[j]])
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    if G.number_of_nodes() == 0:
        print("[Hashtag Co-occurrence] No edges found — dataset may have few hashtags.")
        return G

    # Keep top_n nodes by degree
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]
    G_sub     = G.subgraph([n for n, _ in top_nodes]).copy()

    # Assign node danger color (red = high degree/centrality)
    centrality  = nx.degree_centrality(G_sub)
    node_colors = [plt.cm.YlOrRd(centrality[n] * 3) for n in G_sub.nodes()]
    edge_weights = [G_sub[u][v]["weight"] for u, v in G_sub.edges()]
    max_w       = max(edge_weights) if edge_weights else 1

    pos = nx.spring_layout(G_sub, k=0.8, seed=42, iterations=50)

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors,
                            node_size=500, alpha=0.90)
    nx.draw_networkx_edges(G_sub, pos,
                            width=[1 + 4 * w / max_w for w in edge_weights],
                            edge_color="#888888", alpha=0.5)
    nx.draw_networkx_labels(G_sub, pos, font_size=8, font_weight="bold")

    plt.title(f"Hashtag Co-occurrence Network — Top {top_n} Nodes\n(Harmful Comments Only)",
              fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Co-occurrence network saved → {save_path}")
    return G_sub


# ─────────────────────────────────────────────────────────────
# HASHTAG VELOCITY — growth rate in harmful context
# ─────────────────────────────────────────────────────────────

def compute_hashtag_velocity(master_df:   pd.DataFrame,
                               window_days: int = 7) -> pd.DataFrame:
    """
    Compute how quickly each hashtag's harm_ratio is growing.
    Requires 'created_utc' column.

    Returns DataFrame with velocity (harm_ratio change per day).
    """
    if "created_utc" not in master_df.columns:
        print("[Hashtag Velocity] 'created_utc' column required.")
        return pd.DataFrame()

    master_df = master_df.copy()
    master_df["date"] = pd.to_datetime(master_df["created_utc"], unit="s", utc=True).dt.date

    tag_map = extract_all_hashtags(master_df)
    if tag_map.empty:
        return pd.DataFrame()

    results = []
    for tag, group in tag_map.groupby("hashtag"):
        if len(group) < 20:
            continue
        subset = master_df.loc[master_df.index.intersection(group["row_idx"])]
        if "date" not in subset.columns:
            continue

        daily = subset.groupby("date")["is_harmful"].mean().reset_index()
        daily.columns = ["date", "harm_ratio"]
        daily = daily.sort_values("date")

        if len(daily) < 3:
            continue

        x     = np.arange(len(daily))
        slope = float(np.polyfit(x, daily["harm_ratio"].values, 1)[0])
        recent = daily.tail(window_days)["harm_ratio"].mean()

        results.append({
            "hashtag":          tag,
            "velocity_slope":   round(slope, 5),
            "recent_harm_ratio":round(float(recent), 3),
            "trend":            "rising" if slope > 0.005 else ("falling" if slope < -0.005 else "stable"),
        })

    df_out = pd.DataFrame(results).sort_values("velocity_slope", ascending=False)
    out_path = "outputs/results/hashtag_velocity.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Hashtag velocity analysis saved → {out_path}")
    return df_out


# ─────────────────────────────────────────────────────────────
# CROSS-PLATFORM MIGRATION — Section 14
# ─────────────────────────────────────────────────────────────

def compute_cross_platform_hashtag_overlap(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find harmful hashtags present on MULTIPLE platforms.
    High overlap = coordinated cross-platform harassment campaign.
    """
    tag_map = extract_all_hashtags(master_df)
    if tag_map.empty:
        return pd.DataFrame()

    # For each tag: which platforms does it appear in (harmful context)?
    harmful = master_df[master_df["is_harmful"] == 1]
    tag_map_harmful = extract_all_hashtags(harmful)
    if tag_map_harmful.empty:
        return pd.DataFrame()

    tag_map_harmful = tag_map_harmful.merge(
        harmful[["platform"]].reset_index().rename(columns={"index":"row_idx"}),
        on="row_idx", how="left"
    )

    overlap = (
        tag_map_harmful.groupby("hashtag")["platform"]
        .agg(lambda x: sorted(set(x.dropna())))
        .reset_index()
    )
    overlap["n_platforms"] = overlap["platform"].apply(len)
    overlap["platforms"]   = overlap["platform"].apply(lambda x: ",".join(x))
    overlap = overlap[overlap["n_platforms"] > 1].sort_values("n_platforms", ascending=False)
    overlap.drop(columns=["platform"], inplace=True)

    out_path = "outputs/results/hashtag_cross_platform.csv"
    overlap.to_csv(out_path, index=False)
    print(f"Cross-platform overlap: {len(overlap)} hashtags on 2+ platforms → {out_path}")
    return overlap


# ─────────────────────────────────────────────────────────────
# VISUALISE — Danger Bar Chart
# ─────────────────────────────────────────────────────────────

def plot_hashtag_danger_bar(danger_df: pd.DataFrame,
                             top_n:     int = 20,
                             save_path: str = "outputs/plots/hashtag_danger_bar.png"):
    """Horizontal bar chart of top-N most dangerous hashtags."""
    if danger_df.empty:
        return

    top = danger_df.head(top_n)
    colors = top["category"].map({
        "Critically Unsafe": "#B71C1C",
        "Unsafe":            "#E53935",
        "Moderately Unsafe": "#FB8C00",
        "Safe":              "#43A047",
    }).fillna("#888888")

    fig, ax = plt.subplots(figsize=(11, max(5, top_n // 2)))
    bars = ax.barh(top["hashtag"][::-1], top["danger_score"][::-1],
                   color=colors[::-1].values, edgecolor="white", height=0.7)

    # Annotate harm ratio
    for bar, (_, row) in zip(bars, top[::-1].iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"harm {row['harm_ratio']:.0%}", va="center", fontsize=8)

    ax.set_xlabel("WHSI Danger Score", fontsize=11)
    ax.set_title(f"Top {top_n} Most Dangerous Hashtags", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.axvline(75, color="red",    linestyle="--", alpha=0.5, label="Critical (75)")
    ax.axvline(50, color="orange", linestyle="--", alpha=0.5, label="Unsafe (50)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Hashtag danger bar chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    master = pd.read_csv("data/processed/master_dataset.csv")

    danger_df  = compute_hashtag_danger(master, min_occurrences=5)

    if not danger_df.empty:
        plot_hashtag_danger_bar(danger_df, top_n=20)

    build_hashtag_cooccurrence_graph(master, top_n=40)
    compute_cross_platform_hashtag_overlap(master)

    if "created_utc" in master.columns:
        compute_hashtag_velocity(master)