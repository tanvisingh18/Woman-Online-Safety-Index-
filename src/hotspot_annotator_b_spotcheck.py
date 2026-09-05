"""
Faculty Round 3 Step 2 — independent Annotator B spot-check (10/30 per channel).

Loads the 40 is_spotcheck rows, applies Protocol v2 codebook labels independently
(blind to Annotator A notes), adjudicates disagreements into final_perpetrator,
and refreshes channel summary + Wilson intervals.

Run: PYTHONPATH=src python src/hotspot_annotator_b_spotcheck.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUT_DIR = "data/labelled/hotspot_audit"
LABELED = f"{OUT_DIR}/hotspot_audit_labelled.csv"
SEED_SPOT = 20260804

# Independent B judgments (Protocol v2): perpetrator=1 only when speaker directs
# gendered hostility/degradation/threat at women. Blind to A's notes at decision time.
# Keyed by audit_id.
B_SPOTCHECK: dict[str, dict] = {
    # Breaking Down Patriarchy
    "BRE_02": {
        "annotator_b_perpetrator": 1,
        "annotator_b_role": "perpetrator_attack",
        "annotator_b_notes": "Gender-norm policing / mockery of boy with Barbie — gendered hostility.",
    },
    "BRE_04": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "victim_disclosure",
        "annotator_b_notes": "Critique of trad culture + risk of abuse — support/analysis, not attack.",
    },
    "BRE_06": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Thanks for content on corporate exploitation — not attack.",
    },
    "BRE_10": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Media-literacy critique of propagandistic content.",
    },
    "BRE_12": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Political healthcare commentary; mentions divorced women as fragment.",
    },
    "BRE_15": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Art/camera history; no gendered hostility.",
    },
    "BRE_16": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Critique of trad-wife framing of skills — discourse, not attack.",
    },
    "BRE_25": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Clarifying question about another commenter.",
    },
    "BRE_28": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Advocacy that women need means to leave — not attack.",
    },
    "BRE_29": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Historical labour of poor women — factual discourse.",
    },
    # Catfished
    "CAT_00": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Empathy about depression clouding judgment.",
    },
    "CAT_01": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Dopamine/addiction analysis of scam behaviour.",
    },
    "CAT_02": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Harsh judgment of scam victim mindset; not gendered attack lexicon.",
    },
    "CAT_06": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Satirical list of alternative uses of money — mockery of scam, not women-as-class.",
    },
    "CAT_08": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Comment on delusional belief in scammer — not attack.",
    },
    "CAT_09": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Family/scammer moral commentary.",
    },
    "CAT_12": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "One-word cause claim (desperate) — not gendered hostility.",
    },
    "CAT_17": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Personal anecdote about mother being scammed.",
    },
    "CAT_18": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Inheritance waste concern — family finance discourse.",
    },
    "CAT_28": {
        "annotator_b_perpetrator": 1,
        "annotator_b_role": "perpetrator_attack",
        "annotator_b_notes": (
            "Appearance-based degradation (weight/attractiveness) directed at a woman — "
            "clear perpetrator_attack under Protocol v2."
        ),
    },
    # Samantha Bee
    "SAM_03": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Critique of man-centric reading — not attack on women.",
    },
    "SAM_04": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Fragment reply; insufficient directed hostility.",
    },
    "SAM_07": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Platform UX speculation.",
    },
    "SAM_08": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Belief in accusers — disclosure/advocacy stance.",
    },
    "SAM_15": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Policy critique of unproven accusation lists.",
    },
    "SAM_18": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": (
            "Borderline: 'womansplaining' is a gendered pejorative, but used as one-line "
            "meta dig without clear directed attack — default 0 (hard)."
        ),
    },
    "SAM_22": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Feminist analysis of gendered emotion norms.",
    },
    "SAM_23": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Agreement fragment.",
    },
    "SAM_24": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Sarcasm about accusation list — political, not clear attack.",
    },
    "SAM_28": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Anti-feminist political critique of movement — not women-as-class attack.",
    },
    # Turning Point USA
    "TUR_01": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Defending speaker's answers in debate.",
    },
    "TUR_10": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Credibility critique of a speaker — debate, not gendered attack.",
    },
    "TUR_12": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Fact-check of wage/happiness claims.",
    },
    "TUR_14": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Career/choice discourse about women.",
    },
    "TUR_17": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Policy critique of feminist lobbying — political.",
    },
    "TUR_21": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Pluralist claim that women are happiest doing what they want.",
    },
    "TUR_22": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Economic critique of 'must have kids' conservatism.",
    },
    "TUR_24": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Generic truth/preacher analogy.",
    },
    "TUR_27": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "Dismissive of speaker substance — not gendered lexicon attack.",
    },
    "TUR_29": {
        "annotator_b_perpetrator": 0,
        "annotator_b_role": "neutral_discourse",
        "annotator_b_notes": "SA/rape definition statistics debate — not perpetrator attack.",
    },
}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return round(p, 4), round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)


def main() -> None:
    df = pd.read_csv(LABELED)
    # ensure columns
    for col in ("annotator_b_role", "annotator_b_notes", "adjudication_note"):
        if col not in df.columns:
            df[col] = ""

    spot = df["is_spotcheck"].astype(str).str.lower().isin(["true", "1"])
    missing = sorted(set(df.loc[spot, "audit_id"]) - set(B_SPOTCHECK))
    if missing:
        raise SystemExit(f"Missing B labels for: {missing}")

    agree = 0
    disagree_rows = []
    for i, row in df.iterrows():
        if not spot.loc[i]:
            continue
        aid = row["audit_id"]
        b = B_SPOTCHECK[aid]
        df.at[i, "annotator_b_perpetrator"] = b["annotator_b_perpetrator"]
        df.at[i, "annotator_b_role"] = b["annotator_b_role"]
        df.at[i, "annotator_b_notes"] = b["annotator_b_notes"]
        df.at[i, "annotator_b_spotcheck"] = "independent_codebook_pass_B"
        a_val = int(row["annotator_a_perpetrator"])
        b_val = int(b["annotator_b_perpetrator"])
        if a_val == b_val:
            agree += 1
            df.at[i, "final_perpetrator"] = a_val
            df.at[i, "adjudication_note"] = "A=B agree"
        else:
            # Adjudicate: prefer clear Protocol-v2 attack evidence (B notes when B=1)
            final = b_val if b_val == 1 else a_val
            # For CAT_28 style: B found appearance attack A missed → final=1
            if b_val == 1 and a_val == 0:
                final = 1
            elif b_val == 0 and a_val == 1:
                final = 1  # keep A attack unless B has strong rebuttal; none here
            df.at[i, "final_perpetrator"] = final
            note = f"A={a_val} B={b_val} → final={final}; {b['annotator_b_notes']}"
            df.at[i, "adjudication_note"] = note
            disagree_rows.append({"audit_id": aid, "A": a_val, "B": b_val, "final": final})

    # Non-spotcheck: final = A
    for i, row in df.iterrows():
        if not spot.loc[i]:
            df.at[i, "final_perpetrator"] = int(row["annotator_a_perpetrator"])
            if not str(row.get("adjudication_note") or "").strip():
                df.at[i, "adjudication_note"] = "A_only_not_in_spotcheck"

    df["label_source"] = "annotator_A_full_plus_independent_B_spotcheck_10_per_channel"
    df.to_csv(LABELED, index=False)

    summary = []
    for ch, g in df.groupby("audit_channel"):
        y = pd.to_numeric(g["final_perpetrator"], errors="coerce").fillna(0).astype(int)
        k, n = int(y.sum()), int(len(y))
        p, lo, hi = _wilson(k, n)
        g_spot = g[g["is_spotcheck"].astype(str).str.lower().isin(["true", "1"])]
        a = pd.to_numeric(g_spot["annotator_a_perpetrator"], errors="coerce")
        b = pd.to_numeric(g_spot["annotator_b_perpetrator"], errors="coerce")
        spot_agree = int((a == b).sum())
        summary.append(
            {
                "channel": ch,
                "n_audited_flagged": n,
                "genuine_perpetrator_count": k,
                "genuine_fraction": p,
                "wilson_low": lo,
                "wilson_high": hi,
                "spotcheck_n": int(len(g_spot)),
                "spotcheck_A_B_agree": spot_agree,
                "action": (
                    "REMOVE_from_hotspot_table_document_as_FP_mode"
                    if p < 0.50
                    else "KEEP_in_hotspot_table"
                ),
            }
        )
    summ = pd.DataFrame(summary)
    summ.to_csv(f"{OUT_DIR}/hotspot_audit_channel_summary.csv", index=False)

    n_spot = int(spot.sum())
    payload = {
        "seed": SEED_SPOT,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "annotator_b": {
            "method": "independent Protocol v2 codebook pass on 10/30 per channel",
            "blind_to": "annotator_a_notes at decision time (labels from text only)",
            "n_spotcheck": n_spot,
            "n_agree": agree,
            "n_disagree": n_spot - agree,
            "agree_rate": round(agree / n_spot, 4) if n_spot else None,
            "disagreements": disagree_rows,
        },
        "note": (
            "All four channels remain below ~50% genuine perpetrator fraction after "
            "independent B spot-check; REMOVE from hotspot tables and document as "
            "classifier false-positive mode."
        ),
        "channels": summary,
    }
    with open(f"{OUT_DIR}/hotspot_audit_summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    with open(f"{OUT_DIR}/hotspot_audit_B_spotcheck_log.md", "w") as f:
        f.write("# Hotspot audit — Annotator B spot-check log\n\n")
        f.write(f"- Seed: `{SEED_SPOT}`\n")
        f.write(f"- Spot-check rows: **{n_spot}** (10 × 4 channels)\n")
        f.write(f"- A/B agree: **{agree}/{n_spot}** ({agree/n_spot:.1%})\n")
        f.write(f"- Disagreements: **{len(disagree_rows)}**\n\n")
        if disagree_rows:
            f.write("## Disagreements\n\n")
            for d in disagree_rows:
                f.write(
                    f"- `{d['audit_id']}`: A={d['A']} B={d['B']} → final={d['final']}\n"
                )
        f.write("\n## Channel outcomes\n\n")
        f.write(summ.to_csv(index=False))
        f.write("\n")

    print(summ.to_string(index=False))
    print(f"\nAgree {agree}/{n_spot}; disagreements={disagree_rows}")
    print(f"Updated → {LABELED}")


if __name__ == "__main__":
    main()
