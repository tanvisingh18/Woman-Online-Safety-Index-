# Women Online Safety Index (WHSI × MRI)

Dual-index framework for **platform-level women-targeted harm** (WHSI) and **moderation accountability** (MRI), with **WTSHI** role separation (perpetrator / victim / neutral).

**Team:** Naina Jose · Tanvi Singh · Tejashree Elayaraja  
**Guide:** Prof. Ilanthenral K P S K  

## What this repo contains

| Path | Purpose |
|------|---------|
| `src/` | Pipeline code (classifier, WTSHI, fuzzy WHSI, MRI, validation, demo) |
| `dashboard/` | Streamlit UI |
| `data/processed/` | Scored panel + WHSI/MRI CSVs (`master_women_relevant.csv`, etc.) |
| `data/labelled/v2/` | Protocol v2 held-out gold + agreement |
| `data/transparency/` | MRI transparency inputs |
| `data/scraped/` | Live scrape sources used for the panel |
| `outputs/plots/` | Safety matrix, MRI breakdown, etc. |
| `outputs/results/` | IPW metrics, Path A, safety matrix CSVs |
| `models/harm_models/` | Frozen TF-IDF+LR gendered-harm model |
| `docs/` | Review-2 report, lit chapter, glossary, figures |
| `run_pipeline.py` | End-to-end orchestrator |
| `requirements.txt` | Python deps |

## Locked demo numbers (do not “improve” for primary claims)

- Panel **n = 19,765** (`dataset_split == live_scrape`)
- WHSI_raw: YouTube **28.41** · Reddit **24.95** · Telegram **12.69**
- MRI: YouTube **82.28** · Reddit **69.18** · Telegram **11.27**
- IPW F1 **0.440** · Path A P_w **0.402** (precision gate not met)
- Primary claim uses **WHSI_raw + MRI** only — not literature-adjusted Telegram 53.88

## Quick start

```bash
cd women_safety_index_upload
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Safe Review-2 demo (uses saved panel; no scrape)
python src/mri_engine.py
python src/safety_matrix.py
PYTHONPATH=src python src/review2_demo.py

# Optional UI
streamlit run dashboard/app.py
```

**Do not run alone before a demo:** `python src/fuzzy_engine.py` (overwrites locked WHSI exports). Prefer `integrated_scoring` / existing `whsi_scores_integrated.csv` path.

**Do not need for demo:** full `run_pipeline.py`, DistilBERT, fresh scrapes.

## Live 3-sentence classifier check

```bash
python - <<'PY'
import sys; sys.path.insert(0, "src")
from classifier import classify_dataframe
import pandas as pd
df = pd.DataFrame({
    "comment_text": [
        "Women belong in the kitchen, not in STEM.",
        "I was harassed at work and need advice from other women.",
        "Great video, thanks for posting."
    ],
    "platform": ["YouTube", "Reddit", "YouTube"],
})
print(classify_dataframe(df)[
    ["comment_text", "platform", "is_gendered_harm", "gendered_harm_proba", "harm_role"]
].to_string(index=False))
PY
```

## What was intentionally left out of this upload

- Faculty update packs / closure diffs / sign-off DOCX clutter  
- College PPT/report templates (`reference doc/`)  
- `.pydeps/`, HuggingFace `torch_cache/`, DistilBERT Path A folders (~900MB+)  
- Intermediate lit-review card drafts and scraped wave logs  

Full working tree (if you have it): sibling folder `women_safety_index/`.

## License / data note

Scraped social text is for academic demonstration only. Do not commit live API credentials — use `config/*.example` only.
