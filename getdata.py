import argparse
import json
import os
import time
from datetime import datetime, timedelta

BOT_AUTHORS = {"automoderator", "visualmod", "wsbmod", "market_sentiment"}
BOT_TEXT_PATTERNS = (
    "i am a bot",
    "this action was performed automatically",
    "contact the moderators",
    "ban bet created",
    "ban bet lost",
)


def _author_name(obj) -> str:
    author = getattr(obj, "author", None)
    return str(author).lower() if author else ""


def _looks_like_bot(author: str, text: str) -> bool:
    lowered = (text or "").lower()
    return author.lower() in BOT_AUTHORS or any(pattern in lowered for pattern in BOT_TEXT_PATTERNS)



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
        author = _author_name(comment)
        if (
            hasattr(comment, "body")
            and comment.body not in ("[deleted]", "[removed]")
            and not _looks_like_bot(author, comment.body)
        ):
            comments.append({
                "body": comment.body,
                "author": author,
                "score": comment.score,
                "created_at": datetime.fromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                "comment_id": comment.id,
                "parent_id": comment.parent_id,
                "awards": comment.total_awards_received,
            })

    comments.sort(key=lambda x: x["score"], reverse=True)
    return comments


def _submission_to_post(submission, include_comments=True):
    author = _author_name(submission)
    post_text = f"{submission.title}\n{submission.selftext or ''}"
    if submission.selftext in ("[deleted]", "[removed]") or _looks_like_bot(author, post_text):
        return None

    return {
        "id": submission.id,
        "author": author,
        "title": submission.title,
        "text": submission.selftext,
        "flair": submission.link_flair_text,
        "upvotes": submission.score,
        "upvote_ratio": submission.upvote_ratio,
        "num_comments": submission.num_comments,
        "awards": submission.total_awards_received,
        "comments": get_all_comments(submission) if include_comments else [],
        "created_at": datetime.fromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
        "permalink": f"https://www.reddit.com{submission.permalink}",
    }


def _comment_to_dict(comment):
    author = _author_name(comment)
    body = getattr(comment, "body", "") or ""
    if body in ("[deleted]", "[removed]") or _looks_like_bot(author, body):
        return None
    return {
        "body": body,
        "author": author,
        "score": comment.score,
        "created_at": datetime.fromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
        "comment_id": comment.id,
        "parent_id": comment.parent_id,
        "awards": getattr(comment, "total_awards_received", 0),
    }


def get_recent_wsb_posts(subreddit="wallstreetbets", since_ts=None, lookback_minutes=30, request_delay=0.0):
    reddit = get_reddit_client()
    cutoff = since_ts if since_ts is not None else (datetime.now() - timedelta(minutes=lookback_minutes)).timestamp()
    posts = []
    for submission in reddit.subreddit(subreddit).new(limit=None):
        if submission.created_utc < cutoff:
            break
        post = _submission_to_post(submission, include_comments=True)
        if post:
            posts.append(post)
        if request_delay > 0:
            time.sleep(request_delay)
    return posts


def get_recent_wsb_comments(subreddit="wallstreetbets", since_ts=None, lookback_minutes=30, request_delay=0.0):
    reddit = get_reddit_client()
    cutoff = since_ts if since_ts is not None else (datetime.now() - timedelta(minutes=lookback_minutes)).timestamp()
    comments_by_post = {}
    parent_posts = {}

    for comment in reddit.subreddit(subreddit).comments(limit=None):
        if comment.created_utc < cutoff:
            break
        comment_data = _comment_to_dict(comment)
        if not comment_data:
            continue
        try:
            submission = comment.submission
            post_id = submission.id
            comments_by_post.setdefault(post_id, []).append(comment_data)
            if post_id not in parent_posts:
                parent = _submission_to_post(submission, include_comments=False)
                if parent:
                    parent_posts[post_id] = parent
        except Exception:
            continue
        if request_delay > 0:
            time.sleep(request_delay)

    posts = []
    for post_id, comments in comments_by_post.items():
        post = parent_posts.get(post_id)
        if not post:
            continue
        post["comments"] = comments
        posts.append(post)
    return posts


def refresh_post_scores(posts, subreddit="wallstreetbets", active_days=3.0, max_posts=150, request_delay=0.0):
    reddit = get_reddit_client()
    cutoff = datetime.now() - timedelta(days=active_days)
    refreshed = 0
    candidates = []
    for post in posts:
        created_text = post.get("created_at") or post.get("created_utc")
        try:
            created = datetime.strptime(str(created_text), "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            continue
        if created >= cutoff and post.get("id"):
            candidates.append((created, post))
    candidates.sort(key=lambda item: item[0], reverse=True)
    refresh_candidates = candidates if max_posts <= 0 else candidates[:max_posts]
    for _, post in refresh_candidates:
        post_id = post.get("id")
        try:
            submission = reddit.submission(id=post_id)
            post["upvotes"] = submission.score
            post["upvote_ratio"] = submission.upvote_ratio
            post["num_comments"] = submission.num_comments
            post["awards"] = submission.total_awards_received
            refreshed += 1
        except Exception:
            continue
        if request_delay > 0:
            time.sleep(request_delay)
    return refreshed


def refresh_active_post_comments(posts, subreddit="wallstreetbets", active_days=3.0, max_posts=75, request_delay=0.0):
    reddit = get_reddit_client()
    cutoff = datetime.now() - timedelta(days=active_days)
    candidates = []
    seen = set()
    for post in posts:
        post_id = str(post.get("id") or "").strip()
        if not post_id or post_id in seen:
            continue
        created_text = post.get("created_at") or post.get("created_utc")
        try:
            created = datetime.strptime(str(created_text), "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            continue
        if created >= cutoff:
            seen.add(post_id)
            activity = int(post.get("num_comments") or 0) + abs(int(post.get("upvotes") or 0))
            candidates.append((created, activity, post_id))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    refresh_candidates = candidates if max_posts <= 0 else candidates[:max_posts]
    refreshed_posts = []
    for _, _, post_id in refresh_candidates:
        try:
            submission = reddit.submission(id=post_id)
            post = _submission_to_post(submission, include_comments=True)
            if post:
                refreshed_posts.append(post)
        except Exception:
            continue
        if request_delay > 0:
            time.sleep(request_delay)
    return refreshed_posts


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

        post = _submission_to_post(submission, include_comments=True)
        if not post:
            continue
        posts.append(post)

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
