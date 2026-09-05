# Real vs Estimated — Women Safety Index

Every thesis metric tagged by evidence type.

## WHSI (Women Harassment Severity Index)

| Component | Evidence | Notes |
|-----------|----------|-------|
| Raw comments | **REAL** | yt-dlp, Pullpush, Telegram export, Reddit harassment scrape |
| Harm labels | **MODEL** | EDOS + Detoxify hybrid; F1≈0.58 EDOS holdout |
| WHSI_raw | **COMPUTED** | 70% WTSHI + 30% fuzzy — observable scrape harm |
| WHSI_literature_adjusted | **LITERATURE+SCRAPE** | EHER = observed_rate / public_scrapable_ratio; see whsi_literature_sources.json |
| Bootstrap CI | **STATISTICAL** | harm_rate_confidence_intervals.csv |

## MRI (Moderation Responsiveness Index)

| Platform | Primary source | Empirical layer |
|----------|----------------|-----------------|
| YouTube | Google transparency 2023 | Two-wave yt-dlp (scheduled ≥48h) |
| Reddit | Reddit transparency 2023 | Removal flags + scheduled two-wave Pullpush |
| Twitter | X transparency 2023 | **Historical Davidson 2017 corpus (Path C)** — supplementary |
| Telegram | **ESTIMATED** | No public report |

## Twitter corpus (no API)

- **Path C active:** Davidson et al. 2017 via HuggingFace — `historical_twitter` split
- **Not comparable** to live YouTube/Reddit/Telegram — supplementary panel only
- Live X API optional if credentials added later

## Telegram sampling

- Current scrape: general public channels
- Better sampling: `data/scraped/TELEGRAM_SAMPLING.md` + `telegram_public_scraper.py` (free, public only)
- Literature ecosystem harm: `WHSI_literature_adjusted` column

## Human validation

- Fill `data/labelled/validation_sample.csv` then run validate_classifier --import-annotations

## Current scores

**WHSI_raw (live):** YouTube=20.05, Reddit=19.04, Telegram=11.59
**WHSI_literature_adjusted:** YouTube=22.38, Reddit=23.95, Telegram=33.62
**MRI:** Facebook=84.34, YouTube=82.28, Twitter=72.77, Reddit=60.64, Gab=34.38, Telegram=11.27

**Empirical MRI wave 2 earliest:** 2026-06-25T07:01:32.055947+00:00
