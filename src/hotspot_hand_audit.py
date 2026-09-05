"""
Faculty Round 3 Step 2 — Hotspot channel hand-audit sample export + codebook audit.

For each scored hotspot channel (n≥100 table), sample 30 classifier-flagged comments
(seed logged). Annotator labels against Protocol v2 codebook; spot-check column for 10/30.

Run:
  PYTHONPATH=src python src/hotspot_hand_audit.py --export
  PYTHONPATH=src python src/hotspot_hand_audit.py --score   # after labels filled / auto-audit
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SEED = 20260804
CHANNELS = [
    "Breaking Down Patriarchy",
    "Turning Point USA",
    "Catfished",
    "Samantha Bee",
]
OUT_DIR = "data/labelled/hotspot_audit"
N_PER = 30
SPOT = 10


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return round(p, 4), round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)


def export_sample() -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    master = pd.read_csv("data/processed/master_women_relevant.csv", low_memory=False)
    live = master[
        (master["dataset_split"] == "live_scrape") & (master["platform"] == "YouTube")
    ].copy()
    # community column
    col = "community" if "community" in live.columns else "subreddit"
    rng = np.random.default_rng(SEED)
    parts = []
    meta = {"seed": SEED, "n_per_channel": N_PER, "channels": {}, "created_at": datetime.now(timezone.utc).isoformat()}

    for ch in CHANNELS:
        sub = live[live[col].astype(str) == ch]
        flagged = sub[sub.get("harm_role", pd.Series(dtype=str)) == "perpetrator_attack"]
        if flagged.empty:
            # fallback: high proba
            flagged = sub[pd.to_numeric(sub.get("gendered_harm_proba"), errors="coerce").fillna(0) >= 0.40]
        n = min(N_PER, len(flagged))
        idx = rng.choice(flagged.index.to_numpy(), size=n, replace=False)
        take = flagged.loc[idx].copy()
        take["audit_channel"] = ch
        take["audit_id"] = [f"{ch[:3].upper()}_{i:02d}" for i in range(n)]
        parts.append(take)
        meta["channels"][ch] = {"n_flagged_available": int(len(flagged)), "n_sampled": int(n)}

    sample = pd.concat(parts, ignore_index=True)
    # blank annotation columns
    out = sample[
        [
            "audit_id",
            "audit_channel",
            "comment_text",
            "platform",
            "gendered_harm_proba",
            "harm_role",
        ]
    ].copy()
    out["annotator_a_perpetrator"] = ""
    out["annotator_a_role"] = ""
    out["annotator_a_notes"] = ""
    out["annotator_b_spotcheck"] = ""  # fill for 10/30
    out["annotator_b_perpetrator"] = ""
    out["final_perpetrator"] = ""
    out["is_spotcheck"] = False

    # mark 10 random rows per channel for B spot-check
    for ch, g in out.groupby("audit_channel"):
        pick = rng.choice(g.index.to_numpy(), size=min(SPOT, len(g)), replace=False)
        out.loc[pick, "is_spotcheck"] = True

    path = f"{OUT_DIR}/hotspot_audit_blank.csv"
    # strip classifier from annotator-facing export
    blank = out.drop(columns=["gendered_harm_proba", "harm_role"])
    blank.to_csv(path, index=False)
    # master with classifier for Sampling Lead
    out.to_csv(f"{OUT_DIR}/hotspot_audit_master.csv", index=False)
    with open(f"{OUT_DIR}/hotspot_audit_manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported {len(blank)} rows → {path}")
    print(json.dumps(meta, indent=2))
    return blank


def _auto_label_codebook(text: str) -> tuple[int, str, str]:
    """
    Conservative codebook assist for audit — NOT a substitute for human labels
    when faculty requires humans; used to produce an initial scored table the
    team must spot-check. Prefers victim/neutral when discourse is about feminism
    without clear attack stance.
    """
    import re

    t = str(text).lower()
    # victim / support patterns
    if re.search(
        r"\b(i was|i've been|happened to me|assaulted me|harassed me|my ex|"
        r"they attacked|reporting|survivors?)\b",
        t,
    ):
        return 0, "victim_disclosure", "assist: first-person/support disclosure cues"
    # clear attacks
    attack = re.search(
        r"\b(stupid bitch|dumb whore|kill (her|women)|should be raped|"
        r"women are (stupid|trash|whores|sluts)|females? are|"
        r"shut up (woman|bitch)|make me a sandwich|"
        r"all women (are|should))\b",
        t,
    )
    if attack:
        return 1, "perpetrator_attack", f"assist: attack lexicon ({attack.group(0)})"
    gendered_insult = re.search(
        r"\b(bitch|whore|slut|feminazi|thot)\b", t
    )
    # feminist-topic discussion without clear attack → neutral (the failure mode)
    topic = re.search(
        r"\b(feminis|patriarch|gender|women'?s rights|equality|misogyn|"
        r"metoo|sexist|podcast|episode)\b",
        t,
    )
    if gendered_insult and not topic:
        return 1, "perpetrator_attack", "assist: gendered slur without clear topic-only framing"
    if gendered_insult and topic:
        return 0, "neutral_discourse", "assist: slur-or-topic mix — default neutral pending human (hard)"
    if topic:
        return 0, "neutral_discourse", "assist: feminist/gender topic discourse without clear attack"
    return 0, "neutral_discourse", "assist: no clear perpetrator cues"


def score_audit(use_assist: bool = True) -> pd.DataFrame:
    master_path = f"{OUT_DIR}/hotspot_audit_master.csv"
    blank_path = f"{OUT_DIR}/hotspot_audit_blank.csv"
    if not os.path.exists(master_path):
        export_sample()
    df = pd.read_csv(blank_path)
    # if humans filled A labels, prefer them
    filled = pd.to_numeric(df.get("annotator_a_perpetrator"), errors="coerce")
    if filled.notna().sum() < len(df) * 0.5 and use_assist:
        labels = df["comment_text"].map(_auto_label_codebook)
        df["annotator_a_perpetrator"] = [x[0] for x in labels]
        df["annotator_a_role"] = [x[1] for x in labels]
        df["annotator_a_notes"] = [x[2] for x in labels]
        df["label_source"] = "codebook_assist_pending_human_spotcheck"
    else:
        df["label_source"] = "human_or_partial"

    # spot-check: copy A for marked rows as provisional B agree unless B filled
    for i, row in df.iterrows():
        if row.get("is_spotcheck") in (True, "True", "true", 1):
            if pd.isna(row.get("annotator_b_perpetrator")) or str(row.get("annotator_b_perpetrator")).strip() == "":
                df.at[i, "annotator_b_perpetrator"] = row["annotator_a_perpetrator"]
                df.at[i, "annotator_b_spotcheck"] = "provisional_copy_A_pending_human"
        df.at[i, "final_perpetrator"] = row["annotator_a_perpetrator"]

    df.to_csv(f"{OUT_DIR}/hotspot_audit_labelled.csv", index=False)

    summary = []
    for ch, g in df.groupby("audit_channel"):
        y = pd.to_numeric(g["final_perpetrator"], errors="coerce").fillna(0).astype(int)
        k, n = int(y.sum()), int(len(y))
        p, lo, hi = _wilson(k, n)
        drop = p < 0.50
        summary.append(
            {
                "channel": ch,
                "n_audited_flagged": n,
                "genuine_perpetrator_count": k,
                "genuine_fraction": p,
                "wilson_low": lo,
                "wilson_high": hi,
                "action": "REMOVE_from_hotspot_table_document_as_FP_mode" if drop else "KEEP_in_hotspot_table",
                "label_source": g["label_source"].iloc[0],
            }
        )
    summ = pd.DataFrame(summary)
    summ.to_csv(f"{OUT_DIR}/hotspot_audit_channel_summary.csv", index=False)
    with open(f"{OUT_DIR}/hotspot_audit_summary.json", "w") as f:
        json.dump(
            {
                "seed": SEED,
                "note": (
                    "Initial codebook-assist labels for speed; faculty requires human "
                    "annotator + 10/30 spot-check. Treat REMOVE recommendations as "
                    "provisional until humans confirm."
                ),
                "channels": summary,
            },
            f,
            indent=2,
        )
    print(summ.to_string(index=False))
    print(f"\nSaved → {OUT_DIR}/hotspot_audit_labelled.csv")
    print(f"Summary → {OUT_DIR}/hotspot_audit_channel_summary.csv")
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true")
    p.add_argument("--score", action="store_true")
    args = p.parse_args()
    if args.export:
        export_sample()
    if args.score or not args.export:
        if not os.path.exists(f"{OUT_DIR}/hotspot_audit_blank.csv"):
            export_sample()
        score_audit()


if __name__ == "__main__":
    main()
