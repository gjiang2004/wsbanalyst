import praw
import time
from datetime import datetime, timedelta


reddit = praw.Reddit(client_id='arfcXl0wxWzmPnMBXnhPmA',
                     client_secret='9U6IMEUynxZ2hQhx8CbT1-K3vrxVGw',
                     user_agent='wsbscraper (by u/Appropriate_Still445)')


import json

# json format
def save_to_json(posts, filename="wsb_posts.json"):
    with open(filename, 'w') as f:
        json.dump(posts, f, indent=4)
    print(f"Saved {len(posts)} posts to {filename}")


# get post from _ days ago
def get_wsb_posts(subreddit='wallstreetbets', days=1):
    posts = []
    
    days_ago = datetime.now() - timedelta(days=days)
    days_ago_timestamp = int(days_ago.timestamp())
    
    for submission in reddit.subreddit(subreddit).new(limit=None):

        if submission.created_utc < days_ago_timestamp:
            break

        post_url = submission.url
        
        # url broken for some reason still
        if submission.is_self:
            post_url = f"https://www.reddit.com{submission.permalink}"

        # comments by best

        posts.append({
            'title': submission.title,
            'text': submission.selftext,
            'id': submission.id,
            'upvotes': submission.score,
            'comments': [comment.body for comment in submission.comments.list()[1:11] if hasattr(comment, 'body')],
            'created_at': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            'url': post_url
        })
        print("currently processing date:" + datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'))
    
    return posts

wsb_posts = get_wsb_posts(days=30)
save_to_json(wsb_posts)