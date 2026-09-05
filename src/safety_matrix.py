"""
SECTION 8 — SAFETY MATRIX
Women Safety Index | safety_matrix.py

Primary matrix: YouTube, Reddit, Telegram (measured live scrapes).
Exploratory panel: Gab (Twitter-proxy) — NOT equivalent to X/Twitter.

Quadrants are visualization aids only; thesis claims use continuous scores + CIs.

Run: python src/safety_matrix.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

import sys

sys.path.insert(0, os.path.dirname(__file__))
from platform_labels import add_display_columns, display_name


def assign_quadrant(
    whsi: float,
    mri: float,
    whsi_threshold: float = 30.0,
    mri_threshold: float = 50.0,
) -> str:
    """Descriptive coordinate label — no safety judgment (WHSI = classifier-estimated harm)."""
    whsi_band = "Higher classifier-estimated harm" if whsi >= whsi_threshold else "Lower classifier-estimated harm"
    mri_band = "Higher moderation response" if mri >= mri_threshold else "Lower moderation response"
    return f"{whsi_band} / {mri_band}"


QUADRANT_COLORS = {
    "Higher classifier-estimated harm / Lower moderation response": "#E53935",
    "Higher classifier-estimated harm / Higher moderation response": "#FB8C00",
    "Lower classifier-estimated harm / Lower moderation response": "#FDD835",
    "Lower classifier-estimated harm / Higher moderation response": "#43A047",
}


def build_safety_matrix(
    whsi_path: str = "data/processed/whsi_scores.csv",
    mri_path: str = "data/processed/mri_scores.csv",
) -> pd.DataFrame:
    from scoring_exports import enrich_whshi_scores, enrich_safety_matrix

    if os.path.exists("outputs/results/harm_rate_confidence_intervals.csv"):
        enrich_whshi_scores(whsi_path=whsi_path)

    whsi = pd.read_csv(whsi_path)
    mri = pd.read_csv(mri_path)[["platform", "MRI_score", "MRI_label", "source"]]
    matrix = whsi.merge(mri, on="platform", how="inner")
    # Primary panel: three live platforms only (Faculty Step 4)
    matrix = matrix[matrix["platform"].isin(["YouTube", "Reddit", "Telegram"])].copy()

    if "is_proxy" not in matrix.columns:
        matrix = add_display_columns(matrix)

    matrix["quadrant"] = matrix.apply(
        lambda r: assign_quadrant(r["WHSI_raw"], r["MRI_score"]), axis=1
    )
    matrix["risk_rank"] = matrix["WHSI_raw"] - matrix["MRI_score"]
    matrix["quadrant_note"] = (
        "Descriptive coordinates only — WHSI_raw is classifier-estimated harm, not validated harm rate"
    )
    matrix.to_csv("outputs/results/safety_matrix_data.csv", index=False)

    enrich_safety_matrix(
        matrix_path="outputs/results/safety_matrix_data.csv",
        whsi_path=whsi_path,
        mri_path=mri_path,
    )
    return pd.read_csv("outputs/results/safety_matrix_data.csv")


def _plot_panel(
    ax,
    matrix_df: pd.DataFrame,
    title: str,
    whsi_threshold: float = 30.0,
    mri_threshold: float = 50.0,
    show_region_hints: bool = False,
):
    ax.axhline(mri_threshold, color="#9E9E9E", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvline(whsi_threshold, color="#9E9E9E", linewidth=1.2, linestyle="--", alpha=0.7)

    shade_alpha = 0.08
    ax.fill_betweenx([0, mri_threshold], whsi_threshold, 100, color="#E53935", alpha=shade_alpha)
    ax.fill_betweenx([mri_threshold, 100], whsi_threshold, 100, color="#FB8C00", alpha=shade_alpha)
    ax.fill_betweenx([0, mri_threshold], 0, whsi_threshold, color="#FDD835", alpha=shade_alpha)
    ax.fill_betweenx([mri_threshold, 100], 0, whsi_threshold, color="#43A047", alpha=shade_alpha)

    if show_region_hints:
        kw = dict(fontsize=7, alpha=0.4, fontstyle="italic", ha="center")
        ax.text(65, 12, "higher WHSI\nlower MRI", color="#666666", **kw)
        ax.text(65, 88, "higher WHSI\nhigher MRI", color="#666666", **kw)
        ax.text(15, 12, "lower WHSI\nlower MRI", color="#666666", **kw)
        ax.text(15, 88, "lower WHSI\nhigher MRI", color="#666666", **kw)

    for _, row in matrix_df.iterrows():
        q_color = QUADRANT_COLORS.get(row["quadrant"], "#888888")
        label = row.get("platform_display", display_name(row["platform"], bool(row.get("is_proxy", 0))))
        x = row["WHSI_raw"]
        y = row["MRI_score"]
        xerr = None
        yerr = None
        if pd.notna(row.get("whsi_raw_ci_low")) and pd.notna(row.get("whsi_raw_ci_high")):
            lo = max(0.0, x - row["whsi_raw_ci_low"])
            hi = max(0.0, row["whsi_raw_ci_high"] - x)
            xerr = [[lo], [hi]]
        if pd.notna(row.get("mri_ci_low")) and pd.notna(row.get("mri_ci_high")):
            ylo = max(0.0, y - row["mri_ci_low"])
            yhi = max(0.0, row["mri_ci_high"] - y)
            yerr = [[ylo], [yhi]]

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=10,
            color=q_color,
            ecolor="#424242",
            elinewidth=1.5,
            capsize=4,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
            alpha=0.9,
        )
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )
        ci_txt = row.get("WHSI_score_display", f"{x:.1f}")
        mri_txt = row.get("MRI_display", f"{y:.1f}")
        ax.annotate(
            f"WHSI {ci_txt}\nMRI {mri_txt}",
            xy=(x, y),
            xytext=(6, -22),
            textcoords="offset points",
            fontsize=7,
            color="#444444",
        )

    ax.set_xlabel(
        "WHSI_raw — classifier-estimated harm (70% WTSHI + 30% fuzzy) + bootstrap 95% CI",
        fontsize=10,
    )
    ax.set_ylabel("MRI — moderation responsiveness + sensitivity band", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)


def plot_safety_matrix(
    matrix_df: pd.DataFrame,
    save_path: str = "outputs/plots/safety_matrix.png",
):
    if "is_historical" not in matrix_df.columns:
        matrix_df = add_display_columns(matrix_df)

    primary = matrix_df[
        (matrix_df.get("is_proxy", 0) != 1) & (matrix_df.get("is_historical", 0) != 1)
    ].copy()
    historical = matrix_df[matrix_df.get("is_historical", 0) == 1].copy()
    proxy = matrix_df[matrix_df.get("is_proxy", 0) == 1].copy()
    secondary = historical if not historical.empty else proxy

    n_panels = 2 if len(secondary) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 8), squeeze=False)

    _plot_panel(
        axes[0, 0],
        primary,
        "Primary Safety Matrix\n(YouTube · Reddit · Telegram — live scrapes)",
    )

    if n_panels == 2:
        if not historical.empty:
            sec_title = (
                "Supplementary — Twitter (historical corpus)\n"
                "Upper-bound anchor: hate-speech-concentrated corpus (Davidson 2017) · not live-comparable"
            )
        else:
            sec_title = (
                "Exploratory — Gab (Twitter-proxy)\n"
                "NOT X/Twitter; do not quote as Twitter"
            )
        _plot_panel(axes[0, 1], secondary, sec_title, show_region_hints=False)

    fig.suptitle(
        "Women Online Safety Matrix — WHSI vs MRI\n"
        "Error bars: WHSI bootstrap 95% CI; MRI min–max from weight/horizon sensitivity",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Safety Matrix plot saved → {save_path}")


def _fmt_score(row, score_col: str, display_col: str) -> str:
    if display_col in row.index and pd.notna(row.get(display_col)):
        return str(row[display_col])
    return f"{row[score_col]:.1f}"


def print_safety_summary(matrix_df: pd.DataFrame):
    print(f"\n{'='*78}")
    print("SAFETY MATRIX — continuous scores with uncertainty (primary platforms first)")
    print(f"{'='*78}")
    primary = matrix_df[
        (matrix_df.get("is_proxy", 0) != 1) & (matrix_df.get("is_historical", 0) != 1)
    ].sort_values("WHSI_raw", ascending=False)
    for _, r in primary.iterrows():
        name = r.get("platform_display", r["platform"])
        whsi_txt = _fmt_score(r, "WHSI_score", "WHSI_score_display")
        mri_txt = _fmt_score(r, "MRI_score", "MRI_display")
        print(f"{name:<28} WHSI {whsi_txt:>18}  MRI {mri_txt:>16}  {r['quadrant']}")
    historical = matrix_df[matrix_df.get("is_historical", 0) == 1]
    for _, r in historical.iterrows():
        name = r.get("platform_display", r["platform"])
        whsi_txt = _fmt_score(r, "WHSI_score", "WHSI_score_display")
        mri_txt = _fmt_score(r, "MRI_score", "MRI_display")
        print(f"{name:<28} WHSI {whsi_txt:>18}  MRI {mri_txt:>16}  [historical supplementary]")
    proxy = matrix_df[matrix_df.get("is_proxy", 0) == 1]
    for _, r in proxy.iterrows():
        name = r.get("platform_display", r["platform"])
        whsi_txt = _fmt_score(r, "WHSI_score", "WHSI_score_display")
        mri_txt = _fmt_score(r, "MRI_score", "MRI_display")
        print(f"{name:<28} WHSI {whsi_txt:>18}  MRI {mri_txt:>16}  [exploratory proxy]")
    print("=" * 78)


if __name__ == "__main__":
    matrix = build_safety_matrix()
    plot_safety_matrix(matrix)
    print_safety_summary(matrix)
