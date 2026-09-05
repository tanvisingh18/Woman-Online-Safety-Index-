"""
WHSI / WTSHI weight sensitivity — robustness of platform ranking to formula choices.

Output: outputs/results/whsi_sensitivity.csv

Run: python src/whsi_sensitivity.py
"""

from __future__ import annotations

import os

import pandas as pd

from fuzzy_engine import compute_whsi
from platform_labels import add_display_columns, display_name


def run_whsi_sensitivity(
    features_path: str = "data/processed/platform_features_integrated.csv",
) -> pd.DataFrame:
    if not os.path.exists(features_path):
        print(f"Missing {features_path} — run integrated scoring first.")
        return pd.DataFrame()

    feat = pd.read_csv(features_path)
    rows = []
    weight_sets = [
        (0.70, 0.30, "primary"),
        (0.60, 0.40, "more_fuzzy"),
        (0.80, 0.20, "more_wtshi"),
        (0.55, 0.45, "balanced"),
    ]

    for w_wtshi, w_fuzzy, label in weight_sets:
        for _, row in feat.iterrows():
            wtshi = float(row.get("wtshi_construct", 0) or 0)
            fuzzy_score, _ = compute_whsi(
                row.get("toxicity_integrated", 0),
                row.get("threat_integrated", 0),
                row.get("frequency_integrated", 0),
                row.get("normalization_integrated", 0),
                gendered_harm_rate=float(row.get("perpetrator_harm_rate", 0)),
                gendered_targeting_ratio=float(row.get("women_targeted_perpetrator_pct", 0)),
            )
            whsi_raw = round(float(w_wtshi * wtshi + w_fuzzy * fuzzy_score), 2)
            rows.append(
                {
                    "platform": row["platform"],
                    "analysis": label,
                    "w_wtshi": w_wtshi,
                    "w_fuzzy": w_fuzzy,
                    "wtshi_construct": wtshi,
                    "WHSI_raw": whsi_raw,
                }
            )

    out = pd.DataFrame(rows)
    proxy_map = {"Twitter": 0}  # historical by default; proxy only if gab source
    out["is_proxy"] = out["platform"].map(lambda p: proxy_map.get(p, 0))
    if "dataset_source" in feat.columns:
        pass
    out = add_display_columns(out, proxy_col="is_proxy")
    # Re-derive display after historical flag from features if present
    if "is_historical" in feat.columns:
        hist = feat.set_index("platform")["is_historical"].to_dict()
        out["is_historical"] = out["platform"].map(lambda p: int(hist.get(p, 0) or 0))
        out["platform_display"] = out.apply(
            lambda r: display_name(r["platform"], bool(r.get("is_proxy", 0)), bool(r.get("is_historical", 0))),
            axis=1,
        )
    os.makedirs("outputs/results", exist_ok=True)
    out.to_csv("outputs/results/whsi_sensitivity.csv", index=False)

    # Rank stability summary for live platforms only
    live = out[(out.get("is_historical", 0) != 1) & (out.get("is_proxy", 0) != 1)].copy()
    stability_rows = []
    live_platforms = [p for p in ["YouTube", "Reddit", "Telegram"] if p in live["platform"].unique()]
    for label, grp in live.groupby("analysis"):
        ranked = grp.sort_values("WHSI_raw", ascending=False)["platform"].tolist()
        stability_rows.append(
            {
                "analysis": label,
                "w_wtshi": grp["w_wtshi"].iloc[0],
                "w_fuzzy": grp["w_fuzzy"].iloc[0],
                "rank_order": " > ".join(ranked),
                "YouTube_WHSI_range": None,
                "Reddit_WHSI_range": None,
                "Telegram_WHSI_range": None,
            }
        )
        for plat in live_platforms:
            sub = grp[grp["platform"] == plat]
            if len(sub):
                stability_rows[-1][f"{plat}_WHSI_range"] = f"{sub['WHSI_raw'].min():.1f}–{sub['WHSI_raw'].max():.1f}"

    rank_orders = [r["rank_order"] for r in stability_rows]
    rank_stable = len(set(rank_orders)) == 1
    summary = {
        "live_platforms": live_platforms,
        "weight_configs_tested": ["0.70/0.30", "0.60/0.40", "0.80/0.20", "0.55/0.45"],
        "live_rank_stable_across_weights": rank_stable,
        "stable_rank_order": rank_orders[0] if rank_stable else rank_orders,
        "interpretation": (
            "Live platform WHSI_raw ordering is invariant across tested WTSHI/fuzzy weights."
            if rank_stable
            else "WARNING: live platform ordering changes under alternate weights — report sensitivity table."
        ),
        "configs": stability_rows,
        "epistemology_note": "WHSI values are classifier-estimated; ranking stability does not imply validated harm measurement.",
    }
    import json

    with open("outputs/results/whsi_sensitivity_rank_stability.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nWHSI WEIGHT SENSITIVITY — platform rank stability:")
    for plat in live_platforms:
        sub = live[live["platform"] == plat]
        disp = sub.iloc[0].get("platform_display", plat)
        print(f"  {disp}: WHSI_raw range {sub['WHSI_raw'].min():.1f} – {sub['WHSI_raw'].max():.1f}")
    print(f"  Live rank stable across weights: {rank_stable} ({rank_orders[0] if rank_orders else 'n/a'})")

    return out


if __name__ == "__main__":
    run_whsi_sensitivity()
