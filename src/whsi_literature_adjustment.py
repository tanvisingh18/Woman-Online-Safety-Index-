"""
WHSI_literature_adjusted — ecosystem harm exposure (literature-grounded).

Assumption A1 (see docs/ASSUMPTION_A1_VISIBILITY.md):
  Harm density in non-scrapable channels is AT LEAST as high as in publicly observable channels.
  EHER = observed_rate / public_scrapable_ratio caps at 1.0.

Run: python src/whsi_literature_adjustment.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from structural_platform_evidence import PUBLIC_SCRAPABLE_RATIO

# Women-targeting share among harm (0–1) — only where a cited source states it.
# Otherwise falls back to observed scrape estimate (labeled in export).
LITERATURE_WOMEN_TARGETING_SHARE: dict[str, dict[str, Any]] = {
    "Telegram": {
        "value": 1.0,
        "source_id": "ai_forensics_2026",
        "note": "AI Forensics 2026: victims in studied GBV networks are largely women.",
    },
    "YouTube": {
        "value": 0.868,
        "source_id": "almgren_2023_youtube",
        "note": "Almgren & Strandberg 2023: 86.8% of toxic YouTube comments target individuals.",
    },
}

# Published point-estimate anchors (validation / sensitivity — not merged into headline EHER)
LITERATURE_ANCHOR_RATES: dict[str, dict[str, Any]] = {
    "YouTube": {
        "rate": 0.00643,
        "source_id": "almgren_2023_youtube",
        "note": "512/79,577 toxic comments (0.643%) on general Swedish YouTube channels post-moderation.",
    },
    "Reddit": {
        "rate": 0.0625,
        "source_id": "mattingly_2022_reddit",
        "note": "6.25% norm-violating comments in top-97 subreddits (2016), Mattingly et al. CSCW 2022.",
    },
    "Telegram": {
        "rate": None,
        "source_id": "ai_forensics_2026",
        "note": "No single site-wide perpetrator % published; EHER uses visibility-adjusted scrape (below).",
    },
}

LITERATURE_ECOSYSTEM_SOURCES = [
    {
        "id": "ai_forensics_2026",
        "title": "Harassment as Infrastructure: How Telegram's design enables TFGBV",
        "authors": "AI Forensics",
        "year": 2026,
        "url": "https://aiforensics.org/work/telegram-harassment-infrastructure",
        "finding": "2.8M messages across 16 coordinated GBV groups; abuse in private/semi-private channels that reopen after removal.",
        "used_for": "Telegram PUBLIC_SCRAPABLE_RATIO=0.10; women-targeting=1.0 in studied ecosystem.",
    },
    {
        "id": "almgren_2023_youtube",
        "title": "Toxic language in the comment section of Swedish YouTube channels",
        "authors": "Almgren, Strandberg",
        "year": 2023,
        "url": "https://www.diva-portal.org/smash/get/diva2:1784332/FULLTEXT01.pdf",
        "finding": "0.643% of comments toxic post-moderation; 86.8% of toxic comments individually targeted.",
        "used_for": "YouTube anchor rate + women-targeting; PUBLIC_SCRAPABLE_RATIO=0.85 (public surviving comments).",
    },
    {
        "id": "mattingly_2022_reddit",
        "title": "Measuring the Prevalence of Anti-Social Behavior in Online Communities",
        "authors": "Mattingly, Hajli, Halevy",
        "year": 2022,
        "url": "https://dl.acm.org/doi/10.1145/3555552",
        "finding": "6.25% (2016) / 4.28% (2020) of comments violate platform norms in top-97 subreddits.",
        "used_for": "Reddit anchor rate; PUBLIC_SCRAPABLE_RATIO=0.70 (majority public subreddits).",
    },
    {
        "id": "google_transparency_2023",
        "title": "Google / YouTube Transparency Report 2023",
        "year": 2023,
        "url": "https://transparencyreport.google.com/",
        "finding": "Large-scale proactive comment removal before user visibility.",
        "used_for": "YouTube high PUBLIC_SCRAPABLE_RATIO — surviving comments are post-filter public layer.",
    },
    {
        "id": "chandrasekharan_2017_reddit",
        "title": "You Can't Stay Here: The Efficacy of Reddit's 2015 Ban Examined Through Hate Speech",
        "authors": "Chandrasekharan et al.",
        "year": 2017,
        "url": "https://dl.acm.org/doi/10.1145/3134666",
        "finding": "Hate communities persist in long-tail subreddits; partial ban evasion documented.",
        "used_for": "Reddit PUBLIC_SCRAPABLE_RATIO — expert-coded; mix of public and quarantined communities.",
    },
    {
        "id": "eu_dsa_telegram_2024",
        "title": "EU Digital Services Act — Telegram designated VLOP; transparency obligations",
        "year": 2024,
        "url": "https://digital-strategy.ec.europa.eu/en/policies/dsa-vlop",
        "finding": "Telegram designated very large platform; limited verifiable moderation transparency vs Meta/Google.",
        "used_for": "Independent second source for low Telegram visibility/accountability (separate from AI Forensics harm density).",
    },
    {
        "id": "un_women_cyber_violence_2021",
        "title": "UN Women — Online and ICT-facilitated violence against women and girls",
        "year": 2021,
        "url": "https://www.unwomen.org/en/digital-library/publications/2021/05/online-violence-against-women",
        "finding": "Private messaging and closed groups are primary vectors for technology-facilitated GBV.",
        "used_for": "Independent support for Assumption A1 (hidden-channel harm) — not a numeric ratio source.",
    },
]


def _women_targeting_lit(platform: str, observed_pct: float) -> tuple[float, str]:
    """Return (share 0–1, provenance label)."""
    if platform in LITERATURE_WOMEN_TARGETING_SHARE:
        entry = LITERATURE_WOMEN_TARGETING_SHARE[platform]
        return float(entry["value"]), f"literature:{entry['source_id']}"
    share = float(np.clip(observed_pct / 100.0, 0.0, 1.0))
    return share, "observed_scrape:classifier_estimate"


def ecosystem_harm_exposure_rate(platform: str, observed_perpetrator_rate: float) -> tuple[float, dict[str, Any]]:
    """
    EHER = min(1, observed_rate / public_scrapable_ratio).

    Rationale: if only fraction R of ecosystem harm is publicly observable, then
    ecosystem exposure ≈ observed / R (capped at 100% for conservative thesis bound).
    """
    ratio = float(PUBLIC_SCRAPABLE_RATIO.get(platform, 0.5))
    if ratio <= 0:
        ratio = 0.5
    eher = float(np.clip(observed_perpetrator_rate / ratio, 0.0, 1.0))
    anchor = LITERATURE_ANCHOR_RATES.get(platform, {})
    return round(eher, 4), {
        "public_scrapable_ratio": ratio,
        "observed_perpetrator_rate": round(float(observed_perpetrator_rate), 4),
        "formula": "EHER = min(1, observed_perpetrator_rate / public_scrapable_ratio)",
        "literature_anchor_rate": anchor.get("rate"),
        "literature_anchor_source": anchor.get("source_id"),
    }


def ecosystem_harm_exposure_rate(
    platform: str,
    observed_perpetrator_rate: float,
    scrapable_ratio: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    EHER = min(1, observed_rate / public_scrapable_ratio).

    Assumption A1: dividing by scrapable_ratio assumes unobserved harm density
    is at least as high as observed public-layer density (see docs/ASSUMPTION_A1_VISIBILITY.md).
    """
    ratio = float(scrapable_ratio if scrapable_ratio is not None else PUBLIC_SCRAPABLE_RATIO.get(platform, 0.5))
    if ratio <= 0:
        ratio = 0.5
    eher = float(np.clip(observed_perpetrator_rate / ratio, 0.0, 1.0))
    anchor = LITERATURE_ANCHOR_RATES.get(platform, {})
    return round(eher, 4), {
        "assumption_A1": "Harm in invisible channels >= harm density in visible channels",
        "public_scrapable_ratio": ratio,
        "ratio_provenance": "expert-coded estimate informed by cited sources — not a direct published percentage",
        "observed_perpetrator_rate": round(float(observed_perpetrator_rate), 4),
        "formula": "EHER = min(1, observed_perpetrator_rate / public_scrapable_ratio)",
        "literature_anchor_rate": anchor.get("rate"),
        "literature_anchor_source": anchor.get("source_id"),
    }


def scrapable_ratio_sensitivity_band(
    platform: str,
    observed_perpetrator_rate: float,
    observed_women_targeted_pct: float,
    fuzzy_score: float,
    ratio_band: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Recompute WHSI_lit across scrapable-ratio sensitivity band (Faculty Step 5)."""
    if platform == "Telegram":
        band = ratio_band or [0.05, 0.10, 0.20]
    elif platform in ("YouTube", "Reddit"):
        base = PUBLIC_SCRAPABLE_RATIO.get(platform, 0.5)
        band = ratio_band or [max(0.05, base - 0.10), base, min(0.95, base + 0.10)]
    else:
        base = PUBLIC_SCRAPABLE_RATIO.get(platform, 0.5)
        band = ratio_band or [base]

    rows = []
    for ratio in band:
        wtshi_lit, eher, meta = wtshi_literature_construct(
            platform, observed_perpetrator_rate, observed_women_targeted_pct, scrapable_ratio=ratio
        )
        whsi_lit = round(float(np.clip(0.70 * wtshi_lit + 0.30 * fuzzy_score, 0, 100)), 2)
        rows.append(
            {
                "platform": platform,
                "scrapable_ratio": ratio,
                "EHER": eher,
                "WHSI_literature_adjusted": whsi_lit,
            }
        )
    return rows


def wtshi_literature_construct(
    platform: str,
    observed_perpetrator_rate: float,
    observed_women_targeted_pct: float,
    scrapable_ratio: float | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """WTSHI using literature ecosystem exposure + cited women-targeting."""
    eher, eher_meta = ecosystem_harm_exposure_rate(platform, observed_perpetrator_rate, scrapable_ratio)
    w_share, w_prov = _women_targeting_lit(platform, observed_women_targeted_pct)
    wtshi = round(float(np.clip(100.0 * eher * (0.50 + 0.50 * w_share), 0.0, 100.0)), 2)
    meta = {
        **eher_meta,
        "ecosystem_harm_exposure_rate": eher,
        "women_targeting_share": w_share,
        "women_targeting_provenance": w_prov,
        "wtshi_formula": "100 × EHER × (0.50 + 0.50 × women_targeting_share)",
        "wtshi_literature": wtshi,
    }
    return wtshi, eher, meta


def whsi_literature_adjusted(
    platform: str,
    observed_perpetrator_rate: float,
    observed_women_targeted_pct: float,
    fuzzy_score: float,
) -> tuple[float, dict[str, Any]]:
    """Same WHSI blend as WHSI_raw, with literature WTSHI input."""
    wtshi_lit, eher, meta = wtshi_literature_construct(
        platform, observed_perpetrator_rate, observed_women_targeted_pct
    )
    score = round(float(np.clip(0.70 * wtshi_lit + 0.30 * float(fuzzy_score), 0.0, 100.0)), 2)
    meta["fuzzy_component"] = round(float(fuzzy_score), 2)
    meta["whsi_formula"] = "0.70 × WTSHI_literature + 0.30 × fuzzy"
    meta["WHSI_literature_adjusted"] = score
    return score, meta


def export_literature_sources(path: str = "outputs/results/whsi_literature_sources.json") -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Sensitivity band for live platforms (Faculty Step 5)
    sensitivity_examples = []
    for plat, perp, w_pct, fuzzy in [
        ("Telegram", 0.0345, 75.61, 31.58),
        ("YouTube", 0.1926, 85.23, 25.23),
        ("Reddit", 0.1741, 88.09, 25.26),
    ]:
        band = scrapable_ratio_sensitivity_band(plat, perp, w_pct, fuzzy)
        whsi_vals = [b["WHSI_literature_adjusted"] for b in band]
        sensitivity_examples.append(
            {
                "platform": plat,
                "band": band,
                "WHSI_lit_range": [min(whsi_vals), max(whsi_vals)],
                "telegram_highest_across_band": (
                    plat == "Telegram"
                    or max(whsi_vals) > 30
                ),
            }
        )

    payload = {
        "purpose": "Ecosystem harm exposure for WHSI_literature_adjusted column",
        "assumption_A1": {
            "name": "A1 — visibility equivalence",
            "statement": (
                "Harm density in non-publicly-scrapable channels is at least as high as in "
                "publicly observable channels for the same platform."
            ),
            "doc": "docs/ASSUMPTION_A1_VISIBILITY.md",
            "convergence_claim": (
                "NOT four independent signals — VIS, MBR, and EHER for Telegram derive largely "
                "from AI Forensics 2026 visibility estimate. EU DSA and UN Women provide "
                "independent support for hidden-channel harm (Assumption A1), not numeric convergence. "
                "Finding is literature-consistent, not literature-confirmed."
            ),
        },
        "primary_formula": {
            "EHER": "min(1, observed_perpetrator_rate / public_scrapable_ratio)",
            "WTSHI_literature": "100 × EHER × (0.50 + 0.50 × women_targeting_share)",
            "WHSI_literature_adjusted": "0.70 × WTSHI_literature + 0.30 × fuzzy",
        },
        "public_scrapable_ratio": PUBLIC_SCRAPABLE_RATIO,
        "ratio_label": "expert-coded estimates informed by cited sources — not direct published percentages",
        "scrapable_ratio_sensitivity": sensitivity_examples,
        "women_targeting_literature": LITERATURE_WOMEN_TARGETING_SHARE,
        "literature_anchor_rates": LITERATURE_ANCHOR_RATES,
        "sources": LITERATURE_ECOSYSTEM_SOURCES,
        "removed_metric": "WHSI_corrected — deprecated; use WHSI_literature_adjusted only",
        "disclaimer": (
            "WHSI_raw uses observed scrape rates. WHSI_literature_adjusted is conditional on "
            "Assumption A1 and scrapable-ratio estimates. Present sensitivity band, not point value alone."
        ),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    export_literature_sources()
    print("Exported → outputs/results/whsi_literature_sources.json")

    # Illustrative ranking check with current scrape rates (May 2026 pipeline)
    examples = [
        ("Telegram", 0.0345, 75.61, 28.0),
        ("Reddit", 0.1741, 88.09, 32.0),
        ("YouTube", 0.1926, 85.23, 35.0),
    ]
    print("\nWHSI_literature_adjusted (illustrative fuzzy placeholders):")
    for plat, perp, w_pct, fuzzy in examples:
        score, meta = whsi_literature_adjusted(plat, perp, w_pct, fuzzy)
        print(f"  {plat}: EHER={meta['ecosystem_harm_exposure_rate']:.3f}  WHSI_lit={score:.1f}")
