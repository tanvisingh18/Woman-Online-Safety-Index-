"""
SECTION 9.2 — TWITTER/X SCRAPER
Women Safety Index | src/scraper/twitter_scraper.py

Uses Tweepy with Twitter API v2 (free tier: 500k tweets/month).
Collects tweets from harassment-related hashtags and timelines.

Setup:
  1. Apply at https://developer.twitter.com
  2. Create project + app → get Bearer Token
  3. Set env variable: export TWITTER_BEARER_TOKEN="your_token"
     OR place token in config/twitter_credentials.json

Output: data/scraped/twitter_live.parquet

Run: python src/scraper/twitter_scraper.py
"""

import os, re, time, json, warnings
from datetime import datetime
import pandas as pd

warnings.filterwarnings("ignore")
os.makedirs("data/scraped", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# HARASSMENT HASHTAG LIST (women-safety focused)
# ─────────────────────────────────────────────────────────────
HARASSMENT_HASHTAGS = [
    "MeToo",
    "WomenSafety",
    "OnlineAbuse",
    "CyberHarassment",
    "DigitalViolence",
    "EndGenderBasedViolence",
    "WomenInSTEM",
    "WomenInTech",
    "misogyny",
    "sexualharassment",
    "rapeculture",
    "YesAllWomen",
]

SEARCH_QUERIES = [
    "women harassed online lang:en -is:retweet",
    "sexist comment women lang:en -is:retweet",
    "threaten woman online lang:en -is:retweet",
    "rape threat woman lang:en -is:retweet",
    "misogyny lang:en -is:retweet",
]


# ─────────────────────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────────────────────

def get_twitter_client():
    """Get Tweepy client from env variable or config file."""
    try:
        import tweepy
    except ImportError:
        raise ImportError("Install tweepy: pip install tweepy==4.14.0")

    # Try environment variable first
    bearer = os.environ.get("TWITTER_BEARER_TOKEN")

    if not bearer:
        creds_path = "config/twitter_credentials.json"
        if os.path.exists(creds_path):
            with open(creds_path) as f:
                creds = json.load(f)
            bearer = creds.get("bearer_token")

    if not bearer:
        raise EnvironmentError(
            "Twitter bearer token not found.\n"
            "Set TWITTER_BEARER_TOKEN env variable or create config/twitter_credentials.json"
        )

    return tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)


# ─────────────────────────────────────────────────────────────
# SCRAPE BY HASHTAG
# ─────────────────────────────────────────────────────────────

def extract_hashtags_from_entities(entities: dict) -> str:
    """Extract hashtag strings from tweet entities object."""
    if not entities or "hashtags" not in entities:
        return ""
    return ",".join(h.get("tag", "") for h in entities["hashtags"])


def scrape_hashtag_tweets(hashtag:     str,
                           client,
                           max_results: int = 100) -> pd.DataFrame:
    """
    Scrape recent tweets for a given hashtag.

    Parameters
    ----------
    hashtag     : without # prefix, e.g. "MeToo"
    client      : authenticated Tweepy client
    max_results : 10–100 per request (free tier limit)

    Returns
    -------
    DataFrame with schema compatible with master_dataset
    """
    query = f"#{hashtag} -is:retweet lang:en"
    records = []

    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),   # API cap per request
            tweet_fields=["created_at","public_metrics","author_id","text","entities","lang"],
        )
        if not response.data:
            print(f"  #{hashtag}: 0 tweets")
            return pd.DataFrame()

        for tweet in response.data:
            metrics = tweet.public_metrics or {}
            records.append({
                # Core schema
                "comment_text":   tweet.text,
                "platform":       "Twitter",
                "subreddit":      None,
                "hashtags":       hashtag + "," + extract_hashtags_from_entities(getattr(tweet, "entities", {})),
                # Twitter-specific
                "tweet_id":       tweet.id,
                "author_id":      tweet.author_id,
                "created_utc":    tweet.created_at.timestamp() if tweet.created_at else None,
                "likes":          metrics.get("like_count", 0),
                "retweets":       metrics.get("retweet_count", 0),
                "replies":        metrics.get("reply_count", 0),
                "impressions":    metrics.get("impression_count", 0),
                # Pipeline placeholders
                "is_harmful":     None,
                "toxicity_score": None,
                "threat_score":   None,
                "dataset_source": "Tweepy_live",
                "target_gender":  None,
                "scraped_at":     datetime.utcnow().isoformat(),
                "query_hashtag":  hashtag,
            })

    except Exception as e:
        print(f"  #{hashtag}: Error — {e}")
        return pd.DataFrame()

    print(f"  #{hashtag}: {len(records)} tweets")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# SCRAPE BY QUERY STRING
# ─────────────────────────────────────────────────────────────

def scrape_query_tweets(query:        str,
                         client,
                         max_results:  int = 100) -> pd.DataFrame:
    """Scrape tweets matching an arbitrary query string."""
    records = []
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),
            tweet_fields=["created_at","public_metrics","author_id","text"],
        )
        if not response.data:
            return pd.DataFrame()

        for tweet in response.data:
            metrics = tweet.public_metrics or {}
            records.append({
                "comment_text":   tweet.text,
                "platform":       "Twitter",
                "subreddit":      None,
                "hashtags":       re.findall(r"#(\w+)", tweet.text.lower()),
                "tweet_id":       tweet.id,
                "author_id":      tweet.author_id,
                "created_utc":    tweet.created_at.timestamp() if tweet.created_at else None,
                "likes":          metrics.get("like_count", 0),
                "retweets":       metrics.get("retweet_count", 0),
                "is_harmful":     None,
                "toxicity_score": None,
                "threat_score":   None,
                "dataset_source": "Tweepy_query_live",
                "target_gender":  None,
                "scraped_at":     datetime.utcnow().isoformat(),
                "query_hashtag":  query[:50],
            })
    except Exception as e:
        print(f"  Query '{query[:40]}': Error — {e}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# MULTI-HASHTAG RUNNER
# ─────────────────────────────────────────────────────────────

def scrape_all_hashtags(hashtag_list: list = None,
                         max_results:  int  = 100,
                         output_path:  str  = "data/scraped/twitter_live.parquet") -> pd.DataFrame:
    """Scrape all target hashtags and save to parquet."""
    client = get_twitter_client()
    targets = hashtag_list or HARASSMENT_HASHTAGS
    all_dfs = []

    print(f"Scraping {len(targets)} hashtags on Twitter...")
    for tag in targets:
        df = scrape_hashtag_tweets(tag, client, max_results)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1)   # courtesy delay

    # Also run direct queries
    for query in SEARCH_QUERIES:
        df = scrape_query_tweets(query, client, max_results)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1)

    if not all_dfs:
        print("No Twitter data collected.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.drop_duplicates(subset=["tweet_id"], inplace=True)
    combined.to_parquet(output_path, index=False)
    print(f"\nSaved {len(combined)} tweets → {output_path}")
    return combined


if __name__ == "__main__":
    scrape_all_hashtags()