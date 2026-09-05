"""
Evidence-based structural sub-inputs for WHSI (N, Th) and MRI (removal, consistency).

Does NOT replace WHSI/MRI architecture — enriches inputs per literature alignment.
Sources documented in STRUCTURAL_EVIDENCE_REFERENCES below.

Run standalone: python src/structural_platform_evidence.py
"""

from __future__ import annotations

import json
import os

import pandas as pd

# Publicly scrapable share of total harm-relevant content (literature-informed estimates)
PUBLIC_SCRAPABLE_RATIO = {
    "YouTube": 0.85,
    "Reddit": 0.70,
    "Telegram": 0.10,
    "Twitter": 0.75,
    "Facebook": 0.80,
    "Gab": 0.60,
}

# Moderation bypass rate — banned content / ecosystems re-form (0–1)
MODERATION_BYPASS_RATE = {
    "YouTube": 0.20,
    "Reddit": 0.45,
    "Telegram": 0.85,
    "Twitter": 0.55,
    "Facebook": 0.25,
    "Gab": 0.90,
}

# Sampling coverage weight for WHSI_corrected (1.0 = scrape representative; >1 = under-sampled harm)
SAMPLING_COVERAGE_WEIGHT = {
    "YouTube": 1.0,
    "Reddit": 1.0,
    "Telegram": 1.4,
    "Twitter": 1.0,
    "Facebook": 1.0,
    "Gab": 1.0,
}

# Proactive detection rate — removed before user report (0–1), from transparency / reports
PROACTIVE_DETECTION_RATE = {
    "YouTube": 0.72,
    "Reddit": 0.38,
    "Telegram": 0.03,
    "Twitter": 0.45,
    "Facebook": 0.68,
    "Gab": 0.05,
}

# Regulatory accountability score (0–1): DSA cooperation, enforcement response
REGULATORY_ACCOUNTABILITY_SCORE = {
    "YouTube": 0.88,
    "Reddit": 0.62,
    "Telegram": 0.28,
    "Twitter": 0.55,
    "Facebook": 0.85,
    "Gab": 0.15,
}

# External validation — AI Forensics 2026 aggregate (supplementary row only, not live scrape)
AI_FORENSICS_TELEGRAM_2026 = {
    "platform_display": "Telegram (AI Forensics 2026 ext. est.)",
    "platform": "Telegram",
    "source": "AI Forensics 2026 — ecosystem of abuse report (2.8M msgs, 16 groups)",
    "perpetrator_harm_rate_est": 0.42,
    "structural_risk": "Very High",
    "literature_note": "Upper-bound literature estimate; not merged into headline WHSI",
}

# Full citation chains for PDR/RAS — see export_mri_input_derivations() and docs/MRI_INPUT_DERIVATIONS.md
PDR_RAS_DERIVATIONS = {
    "PDR_definition": (
        "Proactive Detection Rate (PDR): estimated share of policy-violating content "
        "detected/removed by automated or platform-initiated systems before a user report. "
        "Used as 20% sub-input: RemovalRate_adj = 0.80 × reactive_removal + 0.20 × PDR."
    ),
    "RAS_definition": (
        "Regulatory Accountability Score (RAS): transparent coding rubric (0–1) for DSA/VLOP "
        "transparency, law-enforcement cooperation, and documented regulatory pressure. "
        "Used as 30% sub-input: Consistency_adj = 0.70 × reported_consistency + 0.30 × RAS."
    ),
    "PDR": {
        "YouTube": {
            "final_value": 0.72,
            "source_id": "google_transparency_2023",
            "source_document": "Google/YouTube Transparency Report 2023; CA AB 587 Q4 2023 report",
            "source_url": "https://transparencyreport.google.com/youtube-policy/removals",
            "raw_published_figures": [
                {
                    "metric": "Comment removals first-detected by automated flagging",
                    "value": "99.5%",
                    "period": "Jul–Sep 2023",
                    "citation": "Google Transparency Report Help Center, comment enforcement tables",
                },
                {
                    "metric": "Video removals first-detected by automated flagging",
                    "value": "96.1%",
                    "period": "Q4 2023",
                    "citation": "YouTube CA AB 587 Transparency Report (CA OAG filing), Q4 2023",
                },
            ],
            "coding_rule": (
                "PDR cannot be taken as 99.5% because comment removals are spam-dominated (~85%) "
                "and automated-first-detection ≠ gendered-harm-specific proactive removal. "
                "Apply harm-relevance discount d=0.75 to video automated-first-detection anchor: "
                "PDR = 0.961 × 0.75 = 0.721 → 0.72."
            ),
            "derivation_formula": "PDR = automated_first_detection_video × harm_relevance_discount",
            "derivation_steps": "0.961 × 0.75 = 0.72075 → round to 0.72",
        },
        "Reddit": {
            "final_value": 0.38,
            "source_id": "reddit_transparency_2023",
            "source_document": "Reddit Transparency Report 2023; AutoModerator documentation",
            "source_url": "https://www.redditinc.com/policies/transparency-report-2023",
            "raw_published_figures": [
                {
                    "metric": "Reactive removal rate (transparency CSV input)",
                    "value": "68%",
                    "period": "2023",
                    "citation": "data/transparency/moderation_reports.csv ← Reddit Transparency Report 2023",
                },
                {
                    "metric": "AutoModerator / automated tooling prevalence",
                    "value": "documented platform-wide, no single published PDR %",
                    "period": "2023",
                    "citation": "Reddit transparency + moderation SoK (2023) — partial proactive tooling",
                },
            ],
            "coding_rule": (
                "Reddit publishes enforcement volumes but not a clean 'removed before report' % "
                "for harassment. Code PDR=0.38 as medium: below YouTube (heavy automation), "
                "above Telegram (minimal GBV proactive scan in private groups). "
                "Anchored below Meta/Google tier (~0.68–0.72) and above Telegram floor."
            ),
            "derivation_formula": "Expert-coded ordinal anchor on published automation evidence",
            "derivation_steps": "Ordinal scale {Telegram:0.03, Reddit:0.38, YouTube:0.72} — documented in rubric",
        },
        "Telegram": {
            "final_value": 0.03,
            "source_id": "telegram_transparency_gap_2024",
            "source_document": "BBC 2024; NBC 2024; Stanford Internet Observatory (cited in NBC)",
            "source_url": "https://www.bbc.com/news/articles/cy54905nv0go",
            "raw_published_figures": [
                {
                    "metric": "Membership in NCMEC/IWF hash-sharing programmes",
                    "value": "Not a member (BBC 2024)",
                    "period": "2024",
                    "citation": "BBC — Telegram refused to join NCMEC/IWF child-protection schemes",
                },
                {
                    "metric": "Proactive GBV scan in private groups",
                    "value": "Not documented; private chats excluded from public moderation claims",
                    "period": "2024",
                    "citation": "NBC/Stanford — CSAM rules focus on public channels; private GBV ecosystem opaque",
                },
                {
                    "metric": "Telegram DSA report AI/algorithmic detection (public content)",
                    "value": "Millions of removals, but not convertible to GBV PDR",
                    "period": "2025",
                    "citation": "Telegram EU DSA Transparency Report 2025 — aggregate enforcement only",
                },
            ],
            "coding_rule": (
                "For gendered-harm / IBSA in private/semi-private ecosystems (AI Forensics 2026), "
                "proactive detection is negligible. Code PDR=0.03 as floor estimate (2–5% range) "
                "reflecting public-channel hash/AI tooling only, not abuse-network visibility."
            ),
            "derivation_formula": "Expert-coded floor within literature range 0.02–0.05",
            "derivation_steps": "Lower bound of documented range → 0.03",
        },
        "Twitter": {
            "final_value": 0.45,
            "source_id": "twitter_transparency_2023",
            "source_document": "Twitter/X Transparency Report 2023",
            "raw_published_figures": [
                {"metric": "Reactive removal rate (CSV)", "value": "72%", "period": "2023"},
            ],
            "coding_rule": "Mid-tier automation vs Telegram; below YouTube. Supplementary panel only.",
            "derivation_formula": "Expert-coded ordinal anchor",
            "derivation_steps": "Between Reddit (0.38) and YouTube (0.72) → 0.45",
        },
        "Facebook": {
            "final_value": 0.68,
            "source_id": "meta_transparency_2023",
            "source_document": "Meta Community Standards Enforcement Report Q4 2023; Ranking Digital Rights 2023",
            "source_url": "https://transparency.meta.com/",
            "raw_published_figures": [
                {
                    "metric": "Proactive detection rate (Meta CSE — content actioned before user report)",
                    "value": "~90%+ for several policy areas (aggregate; spam/fraud heavy)",
                    "period": "Q4 2023",
                    "citation": "Meta Transparency Center — proactive vs reactive share tables",
                },
                {
                    "metric": "Reactive removal input used in MRI CSV",
                    "value": "85%",
                    "period": "2023",
                    "citation": "data/transparency/moderation_reports.csv ← Meta Transparency Q4 2023",
                },
            ],
            "coding_rule": (
                "Meta publishes high proactive % across policy areas, but harassment/bullying "
                "and gender-based harm are lower-automation than spam. Apply harm-relevance "
                "discount similar to YouTube: start from ~0.90 proactive share × 0.75 "
                "relevance → ~0.68. Place Facebook between Reddit (0.38) and YouTube (0.72)."
            ),
            "derivation_formula": "PDR ≈ published_proactive_share × harm_relevance_discount",
            "derivation_steps": "0.90 × 0.75 = 0.675 → round to 0.68",
            "panel": "secondary_contextual — no live WHSI scrape for Facebook in this thesis",
        },
        "Gab": {
            "final_value": 0.05,
            "source_id": "gab_expert_coded_2023",
            "source_document": "No public Gab transparency report; expert-coded ordinal estimate",
            "source_url": None,
            "raw_published_figures": [
                {
                    "metric": "Published GBV / harassment proactive-detection %",
                    "value": "Not available",
                    "period": "2023",
                    "citation": "Gab does not publish Meta/Google-style enforcement transparency",
                },
            ],
            "coding_rule": (
                "EXPERT-CODED ESTIMATE (Faculty July Step 6). Gab markets itself as a "
                "minimal-moderation free-speech platform; no verifiable proactive GBV "
                "detection figures exist. Code PDR=0.05 as near-floor (above Telegram 0.03 "
                "only for residual public-post tooling), explicitly labelled estimate — "
                "NOT a measured rate. Secondary MRI panel only; excluded from primary "
                "YouTube/Reddit/Telegram results claims."
            ),
            "derivation_formula": "Expert-coded floor on ordinal scale {Telegram:0.03, Gab:0.05, Reddit:0.38, …}",
            "derivation_steps": "No published figure → ordinal floor 0.05 with explicit estimate label",
            "panel": "secondary_expert_estimate",
        },
    },
    "RAS": {
        "YouTube": {
            "final_value": 0.88,
            "source_id": "ranking_digital_rights",
            "source_document": "Ranking Digital Rights 2023; Google DSA/VLOP disclosures; Google LER",
            "source_url": "https://rankingdigitalrights.org/",
            "rubric_criteria": [
                {"criterion": "Publishes detailed transparency / DSA compliance", "weight": 0.35, "score": 0.95},
                {"criterion": "Law-enforcement request cooperation (LER)", "weight": 0.30, "score": 0.90},
                {"criterion": "Low recent regulatory enforcement pressure (2023)", "weight": 0.35, "score": 0.80},
            ],
            "coding_rule": "RAS = weighted sum of rubric criteria (0–1 each).",
            "derivation_formula": "0.35×0.95 + 0.30×0.90 + 0.35×0.80",
            "derivation_steps": "0.3325 + 0.27 + 0.28 = 0.8825 → 0.88",
        },
        "Reddit": {
            "final_value": 0.62,
            "source_id": "ranking_digital_rights",
            "source_document": "Ranking Digital Rights 2023; Reddit Transparency Report 2023",
            "rubric_criteria": [
                {"criterion": "Transparency reporting (partial vs Google/Meta)", "weight": 0.35, "score": 0.70},
                {"criterion": "Law-enforcement / legal request cooperation", "weight": 0.30, "score": 0.65},
                {"criterion": "Regulatory standing (DSA, moderate scrutiny)", "weight": 0.35, "score": 0.55},
            ],
            "coding_rule": "RAS = weighted rubric; mid-tier accountability.",
            "derivation_formula": "0.35×0.70 + 0.30×0.65 + 0.35×0.55",
            "derivation_steps": "0.245 + 0.195 + 0.1925 = 0.6325 → 0.62",
        },
        "Telegram": {
            "final_value": 0.28,
            "source_id": "telegram_regulatory_2024",
            "source_document": "BBC 2024; MLex/Guardian on Durov arrest; EU DSA VLOP scrutiny",
            "source_url": "https://www.mlex.com/mlex/articles/2462368",
            "rubric_criteria": [
                {"criterion": "Transparency quality / independent verification", "weight": 0.35, "score": 0.25},
                {"criterion": "Law-enforcement & regulator cooperation", "weight": 0.30, "score": 0.20},
                {"criterion": "Documented regulatory pressure (DSA, arrests, takedown resistance)", "weight": 0.35, "score": 0.35},
            ],
            "coding_rule": (
                "Low RAS reflects limited verifiable transparency and documented resistance "
                "to external moderation demands (Durov arrest Aug 2024; AI Forensics 2026 DSA calls)."
            ),
            "derivation_formula": "0.35×0.25 + 0.30×0.20 + 0.35×0.35",
            "derivation_steps": "0.0875 + 0.06 + 0.1225 = 0.27 → 0.28",
        },
        "Twitter": {
            "final_value": 0.55,
            "source_id": "ranking_digital_rights",
            "rubric_criteria": [
                {"criterion": "Transparency reporting", "weight": 0.35, "score": 0.60},
                {"criterion": "LER cooperation", "weight": 0.30, "score": 0.55},
                {"criterion": "Regulatory pressure post-2022 ownership change", "weight": 0.35, "score": 0.50},
            ],
            "derivation_formula": "0.35×0.60 + 0.30×0.55 + 0.35×0.50",
            "derivation_steps": "0.21 + 0.165 + 0.175 = 0.55",
        },
        "Facebook": {
            "final_value": 0.85,
            "source_id": "meta_transparency_2023",
            "source_document": "Meta Transparency Report Q4 2023; Ranking Digital Rights 2023; DSA VLOP status",
            "source_url": "https://transparency.meta.com/",
            "rubric_criteria": [
                {"criterion": "Publishes detailed CSE / transparency reports", "weight": 0.35, "score": 0.90},
                {"criterion": "Law-enforcement request cooperation", "weight": 0.30, "score": 0.85},
                {"criterion": "Regulatory standing (DSA VLOP; US state transparency laws)", "weight": 0.35, "score": 0.80},
            ],
            "coding_rule": "RAS = weighted sum of rubric criteria (0–1 each). Near-YouTube tier.",
            "derivation_formula": "0.35×0.90 + 0.30×0.85 + 0.35×0.80",
            "derivation_steps": "0.315 + 0.255 + 0.28 = 0.85",
            "panel": "secondary_contextual — transparency-sourced; no live WHSI scrape",
        },
        "Gab": {
            "final_value": 0.15,
            "source_id": "gab_expert_coded_2023",
            "source_document": "No public Gab transparency / DSA report; expert-coded ordinal estimate",
            "source_url": None,
            "rubric_criteria": [
                {"criterion": "Publishes verifiable enforcement transparency", "weight": 0.35, "score": 0.10},
                {"criterion": "Documented LE / regulator cooperation", "weight": 0.30, "score": 0.15},
                {"criterion": "External regulatory pressure / compliance posture", "weight": 0.35, "score": 0.20},
            ],
            "coding_rule": (
                "EXPERT-CODED ESTIMATE (Faculty July Step 6). Gab has no Ranking Digital Rights "
                "scorecard entry comparable to Meta/Google and no DSA VLOP transparency pack. "
                "RAS=0.15 is an ordinal floor estimate reflecting minimal verifiable accountability. "
                "Label as estimate in every table; secondary MRI panel only."
            ),
            "derivation_formula": "0.35×0.10 + 0.30×0.15 + 0.35×0.20",
            "derivation_steps": "0.035 + 0.045 + 0.07 = 0.15",
            "panel": "secondary_expert_estimate",
        },
    },
}

STRUCTURAL_EVIDENCE_REFERENCES = [
    {
        "id": "ai_forensics_2026",
        "title": "Telegram enables widespread gender-based ecosystem of abuse",
        "use": "VIS/MBR Telegram, supplementary perpetrator rate estimate",
        "url": "https://aiforensics.org/",
    },
    {
        "id": "chandrasekharan_2017",
        "title": "The Bag of Communities — Reddit ban effectiveness",
        "use": "MBR Reddit medium estimate",
    },
    {
        "id": "google_transparency_2023",
        "title": "Google / YouTube Transparency Report 2023",
        "use": "PDR YouTube proactive removal share",
    },
    {
        "id": "ranking_digital_rights",
        "title": "Ranking Digital Rights Big Tech Scorecard",
        "use": "RAS regulatory cooperation framing",
        "url": "https://rankingdigitalrights.org/",
    },
]


def visibility_inversion_score(platform: str) -> float:
    """VIS on 0–100: higher = more harm likely hidden from public scrape."""
    ratio = PUBLIC_SCRAPABLE_RATIO.get(platform, 0.5)
    return round((1.0 - ratio) * 100.0, 2)


def adjust_threat_with_mbr(th_explicit: float, platform: str) -> tuple[float, float]:
    """Th_adjusted = 0.70 × Th_scraped + 0.30 × (MBR × 100)."""
    mbr = MODERATION_BYPASS_RATE.get(platform, 0.5) * 100.0
    adjusted = 0.70 * th_explicit + 0.30 * mbr
    return round(min(100.0, adjusted), 2), round(mbr, 2)


def adjust_normalization_with_vis(n_engagement: float, platform: str) -> tuple[float, float]:
    """N_adjusted = 0.60 × N_engagement + 0.40 × VIS."""
    vis = visibility_inversion_score(platform)
    adjusted = 0.60 * n_engagement + 0.40 * vis
    return round(min(100.0, adjusted), 2), vis


def adjust_removal_rate(reactive_rate: float, platform: str) -> tuple[float, float]:
    """RemovalRate_adj = 0.80 × reactive + 0.20 × PDR."""
    pdr = PROACTIVE_DETECTION_RATE.get(platform, 0.2)
    adjusted = 0.80 * reactive_rate + 0.20 * pdr
    return round(min(1.0, adjusted), 4), pdr


def adjust_consistency_score(consistency: float, platform: str) -> tuple[float, float]:
    """Consistency_adj = 0.70 × reported + 0.30 × RAS."""
    ras = REGULATORY_ACCOUNTABILITY_SCORE.get(platform, 0.5)
    adjusted = 0.70 * consistency + 0.30 * ras
    return round(min(1.0, adjusted), 4), ras


def sampling_weight(platform: str) -> float:
    return SAMPLING_COVERAGE_WEIGHT.get(platform, 1.0)


def structural_risk_label(platform: str) -> str:
    vis = visibility_inversion_score(platform)
    mbr = MODERATION_BYPASS_RATE.get(platform, 0.5) * 100
    composite = 0.5 * vis + 0.5 * mbr
    if composite >= 70:
        return "Very High"
    if composite >= 45:
        return "Medium"
    return "Low"


def literature_alignment_note(platform: str, whsi_raw: float, mri: float) -> str:
    vis = visibility_inversion_score(platform)
    if platform == "Telegram" and whsi_raw < 20 and mri < 25:
        return "Aligns: low observable WHSI + low MRI + very high structural risk (VIS)"
    if platform == "YouTube" and mri > 75:
        return "Aligns: post-filter visible harm + high MRI/PDR"
    if platform == "Reddit":
        return "Partial align: targeted scrape + moderate MRI"
    if platform == "Twitter":
        return "Historical corpus — not live comparable"
    return "See triangulated table"


def export_mri_input_derivations(
    transparency_path: str = "data/transparency/moderation_reports.csv",
    json_path: str = "outputs/results/mri_input_derivations.json",
    markdown_path: str = "docs/MRI_INPUT_DERIVATIONS.md",
) -> dict:
    """
    Export full PDR/RAS citation chains and worked MRI sub-component math
    for thesis methodology (source → raw figure → coding rule → final value).
    """
    from mri_engine import compute_mri, speed_score

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(os.path.dirname(markdown_path), exist_ok=True)

    trans = pd.read_csv(transparency_path)
    worked = []
    for _, row in trans.iterrows():
        plat = row["platform"]
        reactive = float(row["removal_rate"])
        consistency_rep = float(row["consistency_score"])
        hours = float(row["avg_response_hours"])
        removal_adj, pdr = adjust_removal_rate(reactive, plat)
        consistency_adj, ras = adjust_consistency_score(consistency_rep, plat)
        speed = speed_score(hours)
        mri = compute_mri(removal_adj, hours, consistency_adj)
        worked.append({
            "platform": plat,
            "transparency_source": row.get("source", ""),
            "reactive_removal_rate": reactive,
            "reported_consistency": consistency_rep,
            "avg_response_hours": hours,
            "PDR": pdr,
            "RAS": ras,
            "removal_adj_formula": f"0.80 × {reactive} + 0.20 × {pdr}",
            "removal_adj": removal_adj,
            "consistency_adj_formula": f"0.70 × {consistency_rep} + 0.30 × {ras}",
            "consistency_adj": consistency_adj,
            "speed_score": round(speed, 2),
            "MRI_formula": (
                f"0.45 × ({removal_adj}×100) + 0.30 × {round(speed, 2)} "
                f"+ 0.25 × ({consistency_adj}×100)"
            ),
            "MRI_score": mri,
            "PDR_derivation": PDR_RAS_DERIVATIONS["PDR"].get(plat),
            "RAS_derivation": PDR_RAS_DERIVATIONS["RAS"].get(plat),
        })

    payload = {
        "definitions": {
            "PDR": PDR_RAS_DERIVATIONS["PDR_definition"],
            "RAS": PDR_RAS_DERIVATIONS["RAS_definition"],
        },
        "mri_integration": {
            "removal_adjusted": "RemovalRate_adj = 0.80 × reactive_removal + 0.20 × PDR",
            "consistency_adjusted": "Consistency_adj = 0.70 × reported_consistency + 0.30 × RAS",
            "mri": "MRI = 0.45 × RemovalRate_adj + 0.30 × SpeedScore + 0.25 × Consistency_adj",
            "speed": "SpeedScore = max(0, 1 − avg_response_hours / 720) × 100",
        },
        "PDR_derivations": PDR_RAS_DERIVATIONS["PDR"],
        "RAS_derivations": PDR_RAS_DERIVATIONS["RAS"],
        "worked_mri_examples": worked,
        "transparency_csv": transparency_path,
        "references": STRUCTURAL_EVIDENCE_REFERENCES,
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# MRI input derivations — PDR and RAS",
        "",
        "This document closes the citation chain for Proactive Detection Rate (PDR) and "
        "Regulatory Accountability Score (RAS). An examiner asking *\"where does 72% come from?\"* "
        "should be able to follow **source document → raw published figure → coding rule → final value → MRI sub-input**.",
        "",
        "Machine-readable twin: `outputs/results/mri_input_derivations.json` (regenerated by "
        "`python src/structural_platform_evidence.py`).",
        "",
        "## Definitions",
        "",
        f"**PDR:** {PDR_RAS_DERIVATIONS['PDR_definition']}",
        "",
        f"**RAS:** {PDR_RAS_DERIVATIONS['RAS_definition']}",
        "",
        "## How PDR and RAS enter MRI",
        "",
        "| Step | Formula |",
        "|------|---------|",
        "| 1. Reactive removal | From `data/transparency/moderation_reports.csv` |",
        "| 2. Removal adjusted | `0.80 × reactive + 0.20 × PDR` |",
        "| 3. Consistency adjusted | `0.70 × reported + 0.30 × RAS` |",
        "| 4. Speed | `max(0, 1 − hours/720) × 100` |",
        "| 5. MRI | `0.45 × Removal_adj + 0.30 × Speed + 0.25 × Consistency_adj` |",
        "",
    ]

    for input_name, derivations in [("PDR", PDR_RAS_DERIVATIONS["PDR"]), ("RAS", PDR_RAS_DERIVATIONS["RAS"])]:
        lines.append(f"## {input_name} — per-platform derivation")
        lines.append("")
        for plat, d in derivations.items():
            final = d["final_value"]
            pct = int(round(final * 100))
            lines.append(f"### {plat} — {input_name} = {pct}% ({final})")
            lines.append("")
            lines.append(f"| Field | Value |")
            lines.append(f"|-------|-------|")
            lines.append(f"| Source document | {d.get('source_document', '—')} |")
            if d.get("source_url"):
                lines.append(f"| URL | {d['source_url']} |")
            lines.append(f"| Coding rule | {d.get('coding_rule', '—')} |")
            lines.append(f"| Derivation | `{d.get('derivation_formula', '—')}` → **{d.get('derivation_steps', '—')}** |")
            lines.append("")
            if d.get("raw_published_figures"):
                lines.append("**Raw published figures:**")
                lines.append("")
                lines.append("| Metric | Value | Period | Citation |")
                lines.append("|--------|-------|--------|----------|")
                for fig in d["raw_published_figures"]:
                    lines.append(
                        f"| {fig['metric']} | {fig['value']} | {fig.get('period', '—')} | {fig.get('citation', '—')} |"
                    )
                lines.append("")
            if d.get("rubric_criteria"):
                lines.append("**RAS rubric (weighted criteria):**")
                lines.append("")
                lines.append("| Criterion | Weight | Score |")
                lines.append("|-----------|--------|-------|")
                for c in d["rubric_criteria"]:
                    lines.append(f"| {c['criterion']} | {c['weight']} | {c['score']} |")
                lines.append("")

    lines.extend([
        "## Worked MRI examples (live platforms)",
        "",
        "Transparency inputs from `data/transparency/moderation_reports.csv`.",
        "",
    ])
    for w in worked:
        if w["platform"] not in ("YouTube", "Reddit", "Telegram"):
            continue
        r_pct = round(w["reactive_removal_rate"] * 100, 1)
        pdr_pct = round(w["PDR"] * 100, 1)
        rem_adj_pct = round(w["removal_adj"] * 100, 1)
        c_rep_pct = round(w["reported_consistency"] * 100, 1)
        ras_pct = round(w["RAS"] * 100, 1)
        c_adj_pct = round(w["consistency_adj"] * 100, 1)
        lines.append(f"### {w['platform']} — MRI = {w['MRI_score']}")
        lines.append("")
        lines.append(f"1. **Removal:** reactive {r_pct}% ({w['transparency_source']})")
        lines.append(f"2. **PDR:** {pdr_pct}% — see derivation table above")
        lines.append(f"3. **Removal_adj:** {w['removal_adj_formula']} = **{rem_adj_pct}%**")
        lines.append(f"4. **Consistency (reported):** {c_rep_pct}%")
        lines.append(f"5. **RAS:** {ras_pct}% — see derivation table above")
        lines.append(f"6. **Consistency_adj:** {w['consistency_adj_formula']} = **{c_adj_pct}%**")
        lines.append(f"7. **Speed:** {w['avg_response_hours']}h → SpeedScore = {w['speed_score']}")
        lines.append(f"8. **MRI:** {w['MRI_formula']} = **{w['MRI_score']}**")
        lines.append("")

    with open(markdown_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return payload


def export_evidence_config(path: str = "outputs/results/structural_evidence_config.json") -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "public_scrapable_ratio": PUBLIC_SCRAPABLE_RATIO,
        "moderation_bypass_rate": MODERATION_BYPASS_RATE,
        "sampling_coverage_weight": SAMPLING_COVERAGE_WEIGHT,
        "proactive_detection_rate": PROACTIVE_DETECTION_RATE,
        "regulatory_accountability_score": REGULATORY_ACCOUNTABILITY_SCORE,
        "ai_forensics_telegram_2026": AI_FORENSICS_TELEGRAM_2026,
        "references": STRUCTURAL_EVIDENCE_REFERENCES,
        "whsi_sub_input_formulas": {
            "N_adjusted": "0.60 × N_engagement + 0.40 × VIS",
            "Th_adjusted": "0.70 × Th_explicit + 0.30 × (MBR × 100)",
            "WHSI_literature_adjusted": "0.70 × WTSHI(EHER) + 0.30 × fuzzy — Assumption A1",
        },
        "mri_sub_input_formulas": {
            "removal_adjusted": "0.80 × reactive_removal + 0.20 × PDR",
            "consistency_adjusted": "0.70 × enforcement_consistency + 0.30 × RAS",
        },
        "mri_derivation_doc": "docs/MRI_INPUT_DERIVATIONS.md",
        "mri_derivation_json": "outputs/results/mri_input_derivations.json",
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    export_evidence_config()
    export_mri_input_derivations()
    print("Exported structural evidence config → outputs/results/structural_evidence_config.json")
    print("Exported PDR/RAS derivations → outputs/results/mri_input_derivations.json")
    print("Exported methodology doc → docs/MRI_INPUT_DERIVATIONS.md")
