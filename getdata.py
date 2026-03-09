import praw
import time
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent='wsbscraper (by u/Appropriate_Still445)'
)

def save_to_json(posts, filename="wsb_posts.json"):
    with open(filename, 'w') as f:
        json.dump(posts, f, indent=4)
    print(f"Saved {len(posts)} posts to {filename}")

def get_all_comments(submission):
    """Get every comment, sorted by score"""
    try:
        submission.comments.replace_more(limit=0)
    except Exception:
        pass

    comments = []
    for comment in submission.comments.list():
        if hasattr(comment, 'body') and comment.body not in ('[deleted]', '[removed]'):
            comments.append({
                'body': comment.body,
                'score': comment.score,
                'created_at': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'comment_id': comment.id,
                'parent_id': comment.parent_id,
                'awards': comment.total_awards_received,
            })

    comments.sort(key=lambda x: x['score'], reverse=True)
    return comments

def get_wsb_posts(subreddit='wallstreetbets', days=30):
    posts = []
    cutoff_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
    current_day = None

    pbar = tqdm(total=days, desc="Scraping WSB", unit="day")

    for submission in reddit.subreddit(subreddit).new(limit=None):
        if submission.created_utc < cutoff_timestamp:
            break

        post_date = datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d')

        # Advance day counter when we move to a new day
        if post_date != current_day:
            if current_day is not None:
                pbar.update(1)
            current_day = post_date
            pbar.set_postfix(day=post_date, posts=len(posts), refresh=False)

        if submission.selftext in ('[deleted]', '[removed]'):
            continue

        comments = get_all_comments(submission)

        posts.append({
            'id': submission.id,
            'title': submission.title,
            'text': submission.selftext,
            'flair': submission.link_flair_text,
            'upvotes': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'num_comments': submission.num_comments,
            'awards': submission.total_awards_received,
            'comments': comments,
            'created_at': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            'permalink': f"https://www.reddit.com{submission.permalink}"
        })

        time.sleep(0.6)

    pbar.update(1)  # final day
    pbar.close()
    return posts

if __name__ == "__main__":
    wsb_posts = get_wsb_posts(days=30)
    save_to_json(wsb_posts, "wsb_posts.json")
    print(f"\n✅ Done! {len(wsb_posts)} posts, {sum(len(p['comments']) for p in wsb_posts)} comments")