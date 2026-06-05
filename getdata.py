import argparse
import json
import os
import time
from datetime import datetime, timedelta



def get_reddit_client():
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv:
        load_dotenv()
    import praw
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Missing REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET")
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=os.getenv("REDDIT_USER_AGENT", "wsbanalyst scraper"),
    )


def save_to_json(posts, filename="wsb_posts.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(posts)} posts to {filename}")


def get_all_comments(submission):
    try:
        submission.comments.replace_more(limit=0)
    except Exception:
        pass

    comments = []
    for comment in submission.comments.list():
        if hasattr(comment, "body") and comment.body not in ("[deleted]", "[removed]"):
            comments.append({
                "body": comment.body,
                "score": comment.score,
                "created_at": datetime.fromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                "comment_id": comment.id,
                "parent_id": comment.parent_id,
                "awards": comment.total_awards_received,
            })

    comments.sort(key=lambda x: x["score"], reverse=True)
    return comments


def get_wsb_posts(subreddit="wallstreetbets", days=30.0, request_delay=0.6):
    reddit = get_reddit_client()
    posts = []
    cutoff_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
    current_day = None
    progress_total = max(1, int(days + 0.999))

    try:
        from tqdm import tqdm
        pbar = tqdm(total=progress_total, desc="Scraping WSB", unit="day")
    except ImportError:
        pbar = None

    for submission in reddit.subreddit(subreddit).new(limit=None):
        if submission.created_utc < cutoff_timestamp:
            break

        post_date = datetime.fromtimestamp(submission.created_utc).strftime("%Y-%m-%d")

        if post_date != current_day:
            if current_day is not None:
                if pbar:
                    pbar.update(1)
            current_day = post_date
            if pbar:
                pbar.set_postfix(day=post_date, posts=len(posts), refresh=False)

        if submission.selftext in ("[deleted]", "[removed]"):
            continue

        comments = get_all_comments(submission)

        posts.append({
            "id": submission.id,
            "title": submission.title,
            "text": submission.selftext,
            "flair": submission.link_flair_text,
            "upvotes": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "awards": submission.total_awards_received,
            "comments": comments,
            "created_at": datetime.fromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
            "permalink": f"https://www.reddit.com{submission.permalink}",
        })

        if request_delay > 0:
            time.sleep(request_delay)

    if pbar:
        pbar.update(max(0, progress_total - pbar.n))
        pbar.close()
    return posts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape recent r/wallstreetbets posts and comments.")
    parser.add_argument("--subreddit", default="wallstreetbets")
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--output", default="wsb_posts.json")
    parser.add_argument("--request-delay", type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wsb_posts = get_wsb_posts(
        subreddit=args.subreddit,
        days=args.days,
        request_delay=args.request_delay,
    )
    save_to_json(wsb_posts, args.output)
    print(f"Done. {len(wsb_posts)} posts, {sum(len(p['comments']) for p in wsb_posts)} comments")
