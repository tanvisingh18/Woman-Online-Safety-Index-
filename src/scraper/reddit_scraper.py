"""
SECTION 9.1 — REDDIT SCRAPER
Women Safety Index | src/scraper/reddit_scraper.py

Uses PRAW (Python Reddit API Wrapper) — official, rate-limit-aware.
Scrapes comments from targeted subreddits and stores as parquet.

Setup:
  1. Go to https://www.reddit.com/prefs/apps → Create App (script type)
  2. Fill config/reddit_credentials.json with client_id, client_secret, user_agent
  3. pip install praw pyarrow

Output: data/scraped/reddit_live.parquet

Run: python src/scraper/reddit_scraper.py
"""

import os, re, json, time, warnings
import pandas as pd
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.makedirs("data/scraped", exist_ok=True)

TARGET_SUBREDDITS = [
    # Women-centric (likely to surface harassment in discussions)
    "TwoXChromosomes",
    "AskWomen",
    "feminism",
    "WomenInTech",
    # Potentially hostile (documented in literature)
    "MensRights",
    "TheRedPill",
    # High-traffic general (baseline / comparison)
    "relationship_advice",
    "AITA",
    "news",
    "worldnews",
]

def extract_hashtags(text: str) -> str:
    return ",".join(re.findall(r"#(\w+)", str(text).lower()))


def get_reddit_client(config_path: str = "config/reddit_credentials.json"):
    """Initialise PRAW client from credentials JSON."""
    try:
        import praw
    except ImportError:
        raise ImportError("Install praw: pip install praw==7.7.1")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Credentials not found at {config_path}. "
            "Create it with client_id, client_secret, user_agent."
        )

    with open(config_path) as f:
        creds = json.load(f)

    reddit = praw.Reddit(**creds)
    reddit.read_only = True   # safety — we never post
    return reddit

def scrape_subreddit_comments(subreddit_name: str,
                               reddit,
                               n_posts:           int = 200,
                               comments_per_post: int = 50,
                               sort_by:           str = "hot") -> pd.DataFrame:
    """
    Scrape comments from a subreddit.

    Parameters
    ----------
    subreddit_name    : e.g. "TwoXChromosomes"
    reddit            : authenticated PRAW client
    n_posts           : number of posts to fetch
    comments_per_post : max comments per post (top-level only)
    sort_by           : "hot" | "new" | "top" | "controversial"

    Returns
    -------
    DataFrame with schema matching master_dataset + extra scrape fields
    """
    subreddit = reddit.subreddit(subreddit_name)
    feed = {
        "hot":           subreddit.hot,
        "new":           subreddit.new,
        "top":           subreddit.top,
        "controversial": subreddit.controversial,
    }.get(sort_by, subreddit.hot)

    records = []
    posts_fetched = 0

    try:
        for post in tqdm(feed(limit=n_posts), desc=f"r/{subreddit_name}", unit="post"):
            posts_fetched += 1
            try:
                post.comments.replace_more(limit=0)   # flatten MoreComments objects
            except Exception:
                pass

            for comment in post.comments.list()[:comments_per_post]:
                body = getattr(comment, "body", "")
                if body in ("[deleted]", "[removed]", ""):
                    continue

                records.append({
                    # ── Core schema fields ──────────────
                    "comment_text":     body,
                    "platform":         "Reddit",
                    "subreddit":        subreddit_name,
                    "hashtags":         extract_hashtags(body),
                    # ── Reddit-specific metadata ─────────
                    "post_id":          post.id,
                    "comment_id":       comment.id,
                    "post_title":       post.title,
                    "author":           str(comment.author) if comment.author else "[deleted]",
                    "created_utc":      comment.created_utc,
                    "score":            comment.score,
                    "controversiality": getattr(comment, "controversiality", 0),
                    "is_gilded":        bool(comment.gilded),
                    # ── Post-level metadata ──────────────
                    "post_score":       post.score,
                    "post_upvote_ratio":post.upvote_ratio,
                    "post_num_comments":post.num_comments,
                    # ── Pipeline placeholders ────────────
                    "is_harmful":       None,   # filled by classifier
                    "toxicity_score":   None,
                    "threat_score":     None,
                    "dataset_source":   "PRAW_live",
                    "target_gender":    None,
                    # ── Scrape metadata ──────────────────
                    "scraped_at":       datetime.utcnow().isoformat(),
                    "sort_by":          sort_by,
                })

    except Exception as e:
        print(f"  [r/{subreddit_name}] Error: {e}")

    df = pd.DataFrame(records)
    print(f"  r/{subreddit_name}: {len(df)} comments from {posts_fetched} posts")
    return df


def scrape_multiple_subreddits(subreddit_list: list,
                                n_posts:        int = 200,
                                output_path:    str = "data/scraped/reddit_live.parquet",
                                sort_by:        str = "hot") -> pd.DataFrame:
    """
    Scrape all subreddits in list, deduplicate, and save.
    Adds 1-second sleep between subreddits to respect rate limits.
    """
    reddit  = get_reddit_client()
    all_dfs = []

    for sub in subreddit_list:
        try:
            df = scrape_subreddit_comments(sub, reddit, n_posts, sort_by=sort_by)
            all_dfs.append(df)
        except Exception as e:
            print(f"  r/{sub}: FAILED — {e}")
        time.sleep(1)  # Reddit rate limit courtesy delay

    if not all_dfs:
        print("No data collected.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate by comment_id
    before = len(combined)
    combined.drop_duplicates(subset=["comment_id"], inplace=True)
    print(f"\nDeduplication: {before} → {len(combined)} rows")

    combined.to_parquet(output_path, index=False)
    print(f"Saved {len(combined)} comments → {output_path}")
    return combined

def scrape_incremental(subreddit_list: list,
                        checkpoint_path: str = "data/scraped/scrape_checkpoint.json",
                        n_posts: int = 200) -> pd.DataFrame:
    """
    Load last scrape timestamp per subreddit from checkpoint.
    Only fetch posts/comments newer than that timestamp.
    Saves updated checkpoint after each successful scrape.
    """
    checkpoint = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

    reddit  = get_reddit_client()
    all_dfs = []

    for sub in subreddit_list:
        last_ts = checkpoint.get(sub, 0)
        subreddit_obj = reddit.subreddit(sub)
        records = []
        max_ts  = last_ts

        try:
            for post in tqdm(subreddit_obj.new(limit=n_posts), desc=f"r/{sub} (new)"):
                if post.created_utc <= last_ts:
                    break   # already have older posts
                post.comments.replace_more(limit=0)
                for comment in post.comments.list()[:30]:
                    body = getattr(comment, "body", "")
                    if body in ("[deleted]", "[removed]", ""):
                        continue
                    max_ts = max(max_ts, comment.created_utc)
                    records.append({
                        "comment_text": body,
                        "platform":     "Reddit",
                        "subreddit":    sub,
                        "hashtags":     extract_hashtags(body),
                        "post_id":      post.id,
                        "comment_id":   comment.id,
                        "author":       str(comment.author) if comment.author else "[deleted]",
                        "created_utc":  comment.created_utc,
                        "score":        comment.score,
                        "controversiality": getattr(comment, "controversiality", 0),
                        "is_harmful":   None,
                        "toxicity_score": None,
                        "threat_score": None,
                        "dataset_source": "PRAW_incremental",
                        "target_gender": None,
                        "scraped_at":   datetime.utcnow().isoformat(),
                    })
        except Exception as e:
            print(f"  r/{sub}: {e}")
        finally:
            checkpoint[sub] = max_ts
            time.sleep(1)

        df = pd.DataFrame(records)
        if not df.empty:
            all_dfs.append(df)
            print(f"  r/{sub}: {len(df)} NEW comments")

    # Save updated checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    if not all_dfs:
        print("No new data since last scrape.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = "data/scraped/reddit_incremental.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"Saved {len(combined)} new comments → {out_path}")
    return combined

def scrape_wave(wave_num: int, subreddit_list: list = None, n_posts: int = 100):
    """
    Save a labeled wave for empirical reaction-speed measurement.
    Call once now, wait 48h, call again with wave_num=2.
    """
    subreddits = subreddit_list or TARGET_SUBREDDITS[:5]
    out_path   = f"data/scraped/reddit_wave{wave_num}.parquet"
    df = scrape_multiple_subreddits(subreddits, n_posts=n_posts, output_path=out_path)
    print(f"\nWave {wave_num} saved → {out_path} ({len(df)} comments)")
    print(f"Now wait 48 hours and run: python src/scraper/reddit_scraper.py --wave 2")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reddit Scraper for Women Safety Index")
    parser.add_argument("--mode",    default="full",  choices=["full","incremental","wave"])
    parser.add_argument("--wave",    default=1, type=int, help="Wave number (1 or 2) for MRI measurement")
    parser.add_argument("--n_posts", default=200, type=int)
    parser.add_argument("--sort",    default="hot", choices=["hot","new","top","controversial"])
    args = parser.parse_args()

    if args.mode == "full":
        scrape_multiple_subreddits(TARGET_SUBREDDITS, n_posts=args.n_posts, sort_by=args.sort)
    elif args.mode == "incremental":
        scrape_incremental(TARGET_SUBREDDITS, n_posts=args.n_posts)
    elif args.mode == "wave":
        scrape_wave(args.wave, n_posts=args.n_posts)