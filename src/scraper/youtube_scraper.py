"""
SECTION 9.3 — YOUTUBE SCRAPER
Women Safety Index | src/scraper/youtube_scraper.py

Uses YouTube Data API v3 to collect comments on videos
related to women's issues, harassment cases, and social topics.

Setup:
  1. Go to https://console.developers.google.com
  2. Create project → Enable YouTube Data API v3 → Create API Key
  3. Set env variable: export YOUTUBE_API_KEY="your_key"
     OR place in config/youtube_credentials.json

Output: data/scraped/youtube_live.parquet

Run: python src/scraper/youtube_scraper.py
"""

import os, re, json, time, warnings
from datetime import datetime
import pandas as pd

warnings.filterwarnings("ignore")
os.makedirs("data/scraped", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SEARCH TERMS — used to find relevant YouTube videos
# ─────────────────────────────────────────────────────────────
YOUTUBE_SEARCH_TERMS = [
    "women harassment online news",
    "sexism women workplace",
    "metoo movement documentary",
    "online abuse women social media",
    "gender based violence report",
]


# ─────────────────────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────────────────────

def get_youtube_service():
    """Build YouTube API service object."""
    try:
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError("Install: pip install google-api-python-client")

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        creds_path = "config/youtube_credentials.json"
        if os.path.exists(creds_path):
            with open(creds_path) as f:
                api_key = json.load(f).get("api_key")

    if not api_key:
        raise EnvironmentError("Set YOUTUBE_API_KEY env variable or config/youtube_credentials.json")

    return build("youtube", "v3", developerKey=api_key)


def extract_hashtags(text: str) -> str:
    return ",".join(re.findall(r"#(\w+)", str(text).lower()))


# ─────────────────────────────────────────────────────────────
# SEARCH VIDEOS
# ─────────────────────────────────────────────────────────────

def search_videos(query: str, youtube_service, max_results: int = 10) -> list:
    """Return list of video IDs for a search query."""
    try:
        response = youtube_service.search().list(
            q=query,
            part="id,snippet",
            maxResults=max_results,
            type="video",
            relevanceLanguage="en",
            order="relevance",
        ).execute()
        return [item["id"]["videoId"] for item in response.get("items", [])
                if item["id"]["kind"] == "youtube#video"]
    except Exception as e:
        print(f"  YouTube search error for '{query}': {e}")
        return []


# ─────────────────────────────────────────────────────────────
# SCRAPE COMMENTS FROM VIDEO
# ─────────────────────────────────────────────────────────────

def scrape_video_comments(video_id: str,
                           youtube_service,
                           max_comments: int = 100) -> pd.DataFrame:
    """Fetch top-level comments for a YouTube video."""
    records = []
    page_token = None

    while len(records) < max_comments:
        try:
            params = dict(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(records)),
                textFormat="plainText",
                order="relevance",
            )
            if page_token:
                params["pageToken"] = page_token

            response = youtube_service.commentThreads().list(**params).execute()

            for item in response.get("items", []):
                snippet  = item["snippet"]["topLevelComment"]["snippet"]
                text     = snippet.get("textDisplay", "")
                if not text.strip():
                    continue
                records.append({
                    "comment_text":   text,
                    "platform":       "YouTube",
                    "subreddit":      None,
                    "hashtags":       extract_hashtags(text),
                    "video_id":       video_id,
                    "comment_id":     item["snippet"]["topLevelComment"]["id"],
                    "author":         snippet.get("authorDisplayName", "[anonymous]"),
                    "created_utc":    None,   # parsed below
                    "likes":          snippet.get("likeCount", 0),
                    "reply_count":    item["snippet"].get("totalReplyCount", 0),
                    "is_harmful":     None,
                    "toxicity_score": None,
                    "threat_score":   None,
                    "dataset_source": "YouTube_live",
                    "target_gender":  None,
                    "scraped_at":     datetime.utcnow().isoformat(),
                })

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        except Exception as e:
            print(f"  Video {video_id}: {e}")
            break

    print(f"  Video {video_id}: {len(records)} comments")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def scrape_youtube(search_terms:  list = None,
                   max_videos:    int  = 5,
                   max_comments:  int  = 100,
                   output_path:   str  = "data/scraped/youtube_live.parquet") -> pd.DataFrame:
    """Search YouTube for relevant videos and collect their comments."""
    yt = get_youtube_service()
    terms   = search_terms or YOUTUBE_SEARCH_TERMS
    all_dfs = []
    seen_videos = set()

    for term in terms:
        video_ids = search_videos(term, yt, max_results=max_videos)
        for vid in video_ids:
            if vid in seen_videos:
                continue
            seen_videos.add(vid)
            df = scrape_video_comments(vid, yt, max_comments=max_comments)
            if not df.empty:
                all_dfs.append(df)
            time.sleep(0.5)

    if not all_dfs:
        print("No YouTube data collected.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.drop_duplicates(subset=["comment_id"], inplace=True)
    combined.to_parquet(output_path, index=False)
    print(f"\nSaved {len(combined)} YouTube comments → {output_path}")
    return combined


if __name__ == "__main__":
    scrape_youtube()