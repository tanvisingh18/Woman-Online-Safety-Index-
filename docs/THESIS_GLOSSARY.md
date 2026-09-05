# Thesis Glossary — Terminology Freeze (Faculty Step 10)

Use these spellings and definitions **identically** everywhere.

| Acronym | Full name | Definition |
|---------|-----------|------------|
| **WHSI** | Women Harassment Severity Index | Platform-level harm score 0–100 |
| **WTSHI** | Women-Targeted Severity Harm Index | Perpetrator/victim construct input to WHSI |
| **WHSI_raw** | — | `0.70 × WTSHI + 0.30 × fuzzy` — primary measurement score |
| **WHSI_score** | — | `WHSI_raw + hotspot_bonus + pileon_bonus` — headline with tail-risk |
| **WHSI_lit_adj** | WHSI_literature_adjusted | Ecosystem harm under Assumption A1 |
| **MRI** | Moderation Responsiveness Index | Transparency-based accountability 0–100 |
| **EHER** | Ecosystem Harm Exposure Rate | `min(1, observed_rate / scrapable_ratio)` |
| **VIS** | Visibility Inversion Score | `(1 − scrapable_ratio) × 100` |
| **MBR** | Moderation Bypass Rate | Literature-estimated re-formation of banned content |
| **PDR** | Proactive Detection Rate | Estimated pre-report removal share |
| **RAS** | Regulatory Accountability Score | Weighted DSA/transparency rubric |
| **A1** | Assumption A1 | Hidden-channel harm density ≥ visible-channel density |

## Removed terms

| Term | Status |
|------|--------|
| WHSI_corrected | **Removed** — use WHSI_literature_adjusted |
| "Four platforms" | Use **three live platforms** + validation corpus |
| "Convergence of four signals" | Use **literature-consistent** (not literature-confirmed) |

## Which number for what (boxed panel)

| Question | Use this column |
|----------|-----------------|
| What did we observe in our scrape? | **WHSI_raw** + **classifier-flagged rate** |
| What is the Se/Sp-corrected true-rate estimate? | **prevalence-corrected rate** (Rogan–Gladen) — see `rate_reconciliation.csv` |
| What might ecosystem harm be under A1? | **WHSI_literature_adjusted** + **sensitivity band** (Telegram: always show band) |
| Headline score with community tail risk? | **WHSI_score** |
| How accountable is the platform? | **MRI** |
| Pipeline sanity check on toxic corpus? | **validation_corpus_davidson2017.json** (not ranked) |

## Two κ metrics

| Term | Meaning |
|------|---------|
| Inter-annotator κ | Human A vs B on gold labels (**1.00**) |
| Model–Detoxify κ | Classifier vs Detoxify (**≈0.29**) — not gold |

See `docs/KAPPA_METRICS_DISTINCTION.md`.

