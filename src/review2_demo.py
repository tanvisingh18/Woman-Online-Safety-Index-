#!/usr/bin/env python3
"""Review-2 live demo — prints the locked numbers + one WTSHI example.

Run from project root:
  source ../dataset/.venv/bin/activate
  PYTHONPATH=src python src/review2_demo.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

from corpus_config import platform_analysis_mask  # noqa: E402


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    hr("1) MASTER PANEL — women-relevant live scrape")
    master = pd.read_csv(ROOT / "data/processed/master_women_relevant.csv", low_memory=False)
    panel = master[platform_analysis_mask(master)].copy()
    print(f"Total rows in master CSV: {len(master):,}")
    print(f"Analysis panel (dataset_split == live_scrape): n = {len(panel):,}")
    print(panel["platform"].value_counts().rename("n").to_string())
    assert len(panel) == 19765, f"Expected 19765, got {len(panel)}"

    hr("2) ONE COMMENT → classifier flag → WTSHI role")
    # Prefer a short, high-confidence perpetrator example (text may be harsh).
    perp = panel[
        (panel["harm_role"] == "perpetrator_attack")
        & (panel["is_gendered_harm"] == 1)
        & (panel["comment_text"].astype(str).str.len().between(40, 160))
    ].sort_values("gendered_harm_proba", ascending=False)
    row = perp.iloc[0]
    print(f"platform:            {row['platform']}")
    print(f"classifier flag:     is_gendered_harm={int(row['is_gendered_harm'])}  "
          f"proba={float(row['gendered_harm_proba']):.3f}  whsi_harm_flag={row.get('whsi_harm_flag')}")
    print(f"WTSHI role:          {row['harm_role']}")
    print(f"comment (truncated):  {str(row['comment_text']).replace(chr(10), ' ')[:160]}")
    print("\nRole counts on panel:")
    print(panel["harm_role"].value_counts().to_string())

    hr("3) PLATFORM TABLE — WHSI_raw + MRI")
    whsi = pd.read_csv(ROOT / "data/processed/whsi_scores.csv")
    mri = pd.read_csv(ROOT / "data/processed/mri_scores.csv")
    keep = ["YouTube", "Reddit", "Telegram"]
    w = whsi.set_index("platform").loc[keep]
    m = mri.set_index("platform").loc[keep]
    table = pd.DataFrame({"WHSI_raw": w["WHSI_raw"], "MRI": m["MRI_score"]})
    print(table.round(2).to_string())
    print("\n(Do NOT cite WHSI_literature_adjusted / Telegram 53.88 as primary.)")

    hr("4) SAFETY MATRIX PLOT")
    plot = ROOT / "outputs/plots/safety_matrix.png"
    fig = ROOT / "docs/figures/fig5_safety_matrix.png"
    print(f"Open this image: {plot if plot.exists() else fig}")
    print("Or dashboard tab: streamlit run dashboard/app.py  →  Safety Matrix")

    hr("5) VALIDATION (PRIMARY — not June F1≈0.93)")
    ag = json.loads((ROOT / "data/labelled/v2/agreement_v2.json").read_text())
    ipw = json.loads((ROOT / "outputs/results/heldout_ipw_metrics.json").read_text())
    pa = json.loads((ROOT / "outputs/results/path_a_precision_push.json").read_text())
    wmet = ipw["overall"]["weighted"]
    print(f"Held-out n = {ag['n']}")
    print(f"κ perpetrator (binary) = {ag['cohen_kappa_perpetrator_binary']:.3f}")
    print(f"κ harm_role (3-class)  = {ag['cohen_kappa_harm_role_3class']:.3f}")
    print(
        f"IPW corpus: P={wmet['precision']:.3f}  R={wmet['recall']:.3f}  "
        f"Sp={wmet['specificity']:.3f}  F1={wmet['f1']:.3f}"
    )
    print(
        f"Path A best P_w = {pa['gate']['achieved']:.3f}  "
        f"gate {pa['gate']['target']}  met={pa['gate']['met']}"
    )
    print("\nSay: instrument is NOT a cross-platform danger ranking; "
          "YT≈Reddit; Telegram density not comparable; no 5× / no lit-adj 53.88.")
    print("\nDone.")


if __name__ == "__main__":
    main()
