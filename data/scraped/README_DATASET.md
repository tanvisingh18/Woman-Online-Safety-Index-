# YouTube Hybrid Dataset (ready for WHSI / MRI)

Generated: hybrid bulk run (yt-dlp Phase A). Add `YOUTUBE_API_KEY` and re-run `run-hybrid` to append API channel data + moderation fields.

## Main files (use these)

| File | Rows | Contents |
|------|------|----------|
| `youtube_master_comments.csv` | 8,744 | Comments + subcomments, joined with video description & tags |
| `youtube_master_videos.csv` | 36 | Titles, **description boxes**, tags, hashtags, stats |
| `youtube_master_channels.csv` | 33 | Channel IDs and titles (inferred from videos) |
| `youtube_hotspots.csv` | per-video | Toxicity density for hotspot detection |

## Category columns (in master comments)

- `content_category`: `comment` | `subcomment`
- `surface_type`: `search_discourse` (from harassment-related searches)
- `toxicity_tier`: `none` | `low` | `medium` | `high` (keyword-based pre-label)
- `harassment_research_flag`: True if harsh-language keyword matched
- `thread_id`: groups thread under same video
- `video_description`, `video_tags`, `video_hashtags`: from video metadata

## Scale up before deadline

```bash
cd youtube_collector
# Optional: add YOUTUBE_API_KEY to .env then:
../.venv/bin/python main.py run-hybrid --max-videos 45 --max-searches 12

# Or append more videos without API:
../.venv/bin/python main.py run-ytdlp --max-videos 20 --max-searches 12 --append
../.venv/bin/python main.py merge
```
