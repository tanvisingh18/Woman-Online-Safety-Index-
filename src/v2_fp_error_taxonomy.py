"""
Faculty Round 3 Step 7 — taxonomy of Protocol v2 held-out false positives.

Uses the same frozen-threshold re-score as held-out eval (_scores_from_proba),
so FP count matches the primary confusion matrix (~92).

Run: PYTHONPATH=src python src/v2_fp_error_taxonomy.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pandas as pd

from gendered_harm_model import _scores_from_proba

GOLD = "data/labelled/v2/gold_heldout.csv"
OUT_CSV = "outputs/results/v2_false_positive_taxonomy.csv"
OUT_JSON = "outputs/results/v2_false_positive_taxonomy_summary.json"
OUT_MD = "outputs/results/v2_false_positive_taxonomy.md"


def _classify_fp(text: str, role: str, proba: float) -> tuple[str, str]:
    """Return (error_mode, rationale) for a false positive."""
    t = str(text).lower()
    role = str(role or "")

    if role == "victim_disclosure" or re.search(
        r"\b(i was|i've been|happened to me|my ex|assaulted me|harassed me|"
        r"survivors?|reporting him|he raped)\b",
        t,
    ):
        return (
            "victim_disclosure_overflag",
            "Human gold is non-perpetrator (often disclosure/support); model still flagged.",
        )

    if re.search(
        r"\b(feminis|patriarch|misogyn|metoo|sexis|women'?s rights|gender pay|"
        r"trad.?wife|equality|empowerment)\b",
        t,
    ):
        return (
            "topic_vocabulary_feminist_gender_discourse",
            "Gender/feminist topic lexicon without clear directed perpetrator attack.",
        )

    if re.search(
        r"\b(bitch|whore|slut|thot|feminazi)\b",
        t,
    ) and re.search(
        r"\b(i'?m such a|calling myself|reclaim|quoted|he said|she said|\"bitch\")\b",
        t,
    ):
        return (
            "reclaimed_or_quoted_slur",
            "Slur appears in quote, self-reference, or report without endorsement.",
        )

    if re.search(r"\b(bitch|whore|slut|thot|feminazi|females? are)\b", t):
        return (
            "lexical_slur_without_clear_target_attack",
            "Gendered lexical trigger present but gold judges no directed perpetrator attack.",
        )

    if re.search(
        r"\b(rape|assault|harass|abuse|domestic violence|sexual violence)\b",
        t,
    ):
        return (
            "harm_topic_discussion_not_perpetration",
            "Discusses harm topics (news/meta/support) without speaker as perpetrator.",
        )

    if re.search(
        r"\b(women|woman|female|girl|she|her)\b",
        t,
    ) and proba >= 0.55:
        return (
            "high_score_gender_mention_without_attack",
            "High model score + gender mention; gold: neutral/political/other.",
        )

    if proba >= 0.55:
        return (
            "high_confidence_nonattack",
            "High classifier confidence but gold non-perpetrator — residual hard FP.",
        )

    return (
        "threshold_borderline_context",
        "Near-threshold / mid-score flag; gold non-perpetrator.",
    )


def main() -> None:
    gold = pd.read_csv(GOLD)
    rows = []
    for _, row in gold.iterrows():
        text = str(row.get("comment_text", ""))
        sp = float(row.get("gendered_harm_proba", 0) or 0)
        tp = float(row.get("threat_score", 0) or 0)
        out = _scores_from_proba(text, sp, tp)
        yhat = 1 if out["harm_role"] == "perpetrator_attack" else 0
        y = int(row["human_label_perpetrator"])
        if not (yhat == 1 and y == 0):
            continue
        mode, rationale = _classify_fp(text, row.get("human_label_role"), sp)
        rows.append(
            {
                "row_id": row["row_id"],
                "platform": row["platform"],
                "gendered_harm_proba": sp,
                "threat_score": tp,
                "human_label_role": row.get("human_label_role"),
                "error_mode": mode,
                "rationale": rationale,
                "comment_text": text,
            }
        )

    fp = pd.DataFrame(rows)
    fp.to_csv(OUT_CSV, index=False)

    counts = fp["error_mode"].value_counts().to_dict()
    by_plat = (
        fp.groupby(["platform", "error_mode"]).size().unstack(fill_value=0).to_dict()
        if len(fp)
        else {}
    )
    # prefer readable platform×mode counts
    plat_mode = (
        fp.groupby(["platform", "error_mode"]).size().reset_index(name="n").to_dict("records")
    )

    # Decision aid for Path A
    topic_n = int(counts.get("topic_vocabulary_feminist_gender_discourse", 0))
    victim_n = int(counts.get("victim_disclosure_overflag", 0))
    lexical_n = int(counts.get("lexical_slur_without_clear_target_attack", 0)) + int(
        counts.get("reclaimed_or_quoted_slur", 0)
    )
    harm_topic_n = int(counts.get("harm_topic_discussion_not_perpetration", 0))
    gender_hi_n = int(counts.get("high_score_gender_mention_without_attack", 0))
    structural = topic_n + victim_n + lexical_n + harm_topic_n + gender_hi_n
    share_structural = structural / len(fp) if len(fp) else 0.0
    borderline_n = int(counts.get("threshold_borderline_context", 0)) + int(
        counts.get("high_confidence_nonattack", 0)
    )

    if share_structural >= 0.45:
        path_a_hint = (
            f"~{share_structural:.0%} of FPs are disclosure/topic/lexical/gender-mention modes "
            f"(n={structural}); ~{borderline_n} are borderline/high-confidence residual. "
            "Recommended Path A sequence: (1) error-informed re-threshold + negative-class "
            "filters on June set only; (2) evaluate on IPW-weighted v2 only; (3) if precision "
            "still <0.5, fine-tune DistilBERT/HateBERT on EDOS+gold — lexical TF-IDF alone "
            "is unlikely to clear the TwoX/topic failure mode."
        )
    else:
        path_a_hint = (
            "Substantial borderline/threshold FPs → try re-threshold on June set first, "
            "evaluate on reweighted v2 only; escalate to transformer if precision still <0.5."
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_false_positives": int(len(fp)),
        "error_mode_counts": counts,
        "platform_error_mode_counts": plat_mode,
        "share_structural_lexical_topic_disclosure": round(share_structural, 4),
        "path_a_recommendation": path_a_hint,
        "files": {"taxonomy_csv": OUT_CSV, "summary_json": OUT_JSON},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# v2 held-out false-positive taxonomy (Step 7)",
        "",
        f"**n_FP = {len(fp)}** (frozen thresholds; same scorer as held-out eval).",
        "",
        "## Error-mode counts",
        "",
        "| Error mode | n |",
        "|------------|---|",
    ]
    for mode, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {mode} | {n} |")
    lines += [
        "",
        f"**Structural lexical/topic/disclosure share:** {share_structural:.1%}",
        "",
        "## Path A implication",
        "",
        path_a_hint,
        "",
        f"Full rows: `{OUT_CSV}`",
        "",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"FP n={len(fp)}")
    print(pd.Series(counts).sort_values(ascending=False).to_string())
    print(path_a_hint)


if __name__ == "__main__":
    main()
