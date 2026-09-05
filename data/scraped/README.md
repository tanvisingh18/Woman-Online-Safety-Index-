# Where to put your scraped datasets

Upload **any** scraped platform data into this folder (`data/scraped/`).

## Your 4 thesis platforms

| Platform | What to upload | Notes |
|----------|----------------|-------|
| **YouTube** | `youtube_master_comments.csv` (+ videos/channels optional) | Comments **and** subcomments — same file if `content_category` / `parent_comment_id` present |
| **Reddit** | `reddit_live.parquet` OR symlink `reddit_1M_unlabelled.csv` | Raw `text` only → classifier labels it |
| **Telegram** | `telegram_messages.csv` | Need: message text + chat/group name column |
| **Twitter** | `twitter_live.parquet` | When ready; until then Davidson/UCBerkeley cover Twitter in master set |

## Already on your machine (no move required)

```
../dataset/data/youtube_master_comments.csv   ← YouTube (8.7k comments)
../reddit_1M_unlabelled.csv                   ← Reddit raw text (1M — pipeline samples 50k)
../edos_labelled_aggregated.csv               ← Labelled Reddit/Gab for training & calibration
```

**Telegram:** not found yet — export your scrape to `data/scraped/telegram_messages.csv`

## Recommended file names

| Platform | Put file here | Format |
|----------|---------------|--------|
| Reddit | `data/scraped/reddit_live.parquet` or `.csv` | From `reddit_scraper.py` or your own scrape |
| Twitter/X | `data/scraped/twitter_live.parquet` | From `twitter_scraper.py` |
| YouTube | `data/scraped/youtube_live.parquet` **OR** symlink/copy your master CSV | See below |
| Gab / other | `data/scraped/gab_live.csv` | Any CSV with a text column |
| Wave 2 (MRI test) | `data/scraped/reddit_wave1.parquet`, `reddit_wave2.parquet` | Same subreddits, 48h apart |

## Your existing YouTube data (already on disk)

You can **copy or symlink** (do not move originals):

```bash
cp "../dataset/data/youtube_master_comments.csv" "data/scraped/youtube_master_comments.csv"
```

Or point the pipeline at the full path:

```bash
python run_pipeline.py --youtube "../dataset/data/youtube_master_comments.csv"
```

## Required columns (pipeline will map aliases)

**Minimum:** a text column (`comment_text`, `text`, `body`, `tweet`, …)

**Strongly recommended for accurate WHSI/MRI:**

- `platform` — Reddit / Twitter / YouTube / Gab
- `is_harmful` or labels — otherwise classifier runs automatically
- `toxicity_score`, `threat_score` — 0–1 floats (optional; derived if missing)
- `score` or `like_count` — for Normalization dimension
- `author` — for Frequency (repeat offenders)
- `subreddit` or `channel_title` — for hotspot detection
- `created_utc` or `published_at` — for time-window spikes
- `hashtags` — for Section 11 hashtag analysis

## Check if your data is good enough

```bash
python src/dataset_audit.py --path data/scraped/your_file.csv
python src/dataset_audit.py --all-scraped
```

## Ingest into the master dataset

```bash
python src/load_scraped.py --all
python run_pipeline.py --skip-influence   # or full pipeline after ingest
```
