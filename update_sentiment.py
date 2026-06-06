import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        env_path = Path(".env")
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
        return
    load_dotenv()


def _require_reddit_env() -> None:
    missing = [
        name
        for name in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET")
        if not os.getenv(name)
    ]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required Reddit environment variables: {joined}")


def _parse_created_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value))
    text = str(value).strip()
    for fmt in (TS_FORMAT, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _load_posts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected {path} to contain a JSON list of posts")
    return data


def _merge_comments(existing: list[dict], incoming: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for comment in existing + incoming:
        key = str(comment.get("comment_id") or comment.get("id") or comment.get("body") or len(by_id))
        by_id[key] = comment
    return sorted(
        by_id.values(),
        key=lambda c: (int(c.get("score") or 0), str(c.get("created_at") or "")),
        reverse=True,
    )


def _merge_posts(existing: list[dict], incoming: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for post in existing:
        post_id = str(post.get("id") or post.get("permalink") or len(by_id))
        by_id[post_id] = post

    for post in incoming:
        post_id = str(post.get("id") or post.get("permalink") or len(by_id))
        if post_id in by_id:
            current = by_id[post_id]
            merged = {**current, **post}
            merged["comments"] = _merge_comments(current.get("comments") or [], post.get("comments") or [])
            by_id[post_id] = merged
        else:
            by_id[post_id] = post

    return list(by_id.values())


def _prune_posts(posts: list[dict], window_days: float) -> list[dict]:
    cutoff = datetime.now() - timedelta(days=window_days)
    kept = []
    for post in posts:
        created = _parse_created_at(post.get("created_at") or post.get("created_utc"))
        recent_post = created is None or created >= cutoff
        recent_comments = []
        for comment in post.get("comments") or []:
            comment_created = _parse_created_at(comment.get("created_at") or comment.get("created_utc"))
            if comment_created is None or comment_created >= cutoff:
                recent_comments.append(comment)
        if recent_post or recent_comments:
            post = {**post, "comments": recent_comments if not recent_post else post.get("comments") or []}
            kept.append(post)
    return sorted(
        kept,
        key=lambda p: _parse_created_at(p.get("created_at") or p.get("created_utc")) or datetime.min,
        reverse=True,
    )


def _load_state(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(path: Path, state: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _latest_post_ts(posts: list[dict]) -> float | None:
    latest: float | None = None
    for post in posts:
        created = _parse_created_at(post.get("created_at") or post.get("created_utc"))
        if created is None:
            continue
        ts = created.timestamp()
        latest = ts if latest is None else max(latest, ts)
    return latest


def _latest_comment_ts(posts: list[dict]) -> float | None:
    latest: float | None = None
    for post in posts:
        for comment in post.get("comments") or []:
            created = _parse_created_at(comment.get("created_at") or comment.get("created_utc"))
            if created is None:
                continue
            ts = created.timestamp()
            latest = ts if latest is None else max(latest, ts)
    return latest


def _write_posts(path: Path, posts: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh a rolling WSB post store and regenerate ticker sentiment JSON."
    )
    parser.add_argument("--scrape-days", type=float, default=float(os.getenv("WSB_SCRAPE_DAYS", "1")))
    parser.add_argument("--window-days", type=float, default=float(os.getenv("WSB_SENTIMENT_WINDOW_DAYS", "28")))
    parser.add_argument("--subreddit", default=os.getenv("WSB_SUBREDDIT", "wallstreetbets"))
    parser.add_argument("--store-file", default=os.getenv("WSB_POSTS_FILE", "wsb_posts.json"))
    parser.add_argument("--output", default=os.getenv("TICKER_SENTIMENT_FILE", "ticker_sentiment.json"))
    parser.add_argument("--daily-output", default=os.getenv("DAILY_SENTIMENT_FILE", "backend/agg_sentiment.json"))
    parser.add_argument("--aggregate-days", type=float, default=float(os.getenv("SENTIMENT_AGGREGATE_WINDOW_DAYS", "14")))
    parser.add_argument("--finbert-model", default=os.getenv("FINBERT_MODEL", "ProsusAI/finbert"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("FINBERT_BATCH_SIZE", "4")))
    parser.add_argument("--min-mentions", type=int, default=int(os.getenv("MIN_TICKER_MENTIONS", "3")))
    parser.add_argument("--min-confidence", type=float, default=float(os.getenv("MIN_TICKER_CONFIDENCE", "0.65")))
    parser.add_argument("--request-delay", type=float, default=float(os.getenv("REDDIT_REQUEST_DELAY", "0.6")))
    parser.add_argument("--state-file", default=os.getenv("WSB_SCRAPE_STATE_FILE", "wsb_scrape_state.json"))
    parser.add_argument("--storage", choices=("db", "json"), default=os.getenv("WSB_STORAGE", "db"))
    parser.add_argument("--overlap-minutes", type=float, default=float(os.getenv("WSB_SCRAPE_OVERLAP_MINUTES", "30")))
    parser.add_argument("--score-refresh-days", type=float, default=float(os.getenv("WSB_SCORE_REFRESH_DAYS", "3")))
    parser.add_argument("--max-score-refresh", type=int, default=int(os.getenv("WSB_MAX_SCORE_REFRESH", "150")))
    parser.add_argument("--comment-refresh-days", type=float, default=float(os.getenv("WSB_COMMENT_REFRESH_DAYS", "3")))
    parser.add_argument("--max-comment-refresh-posts", type=int, default=int(os.getenv("WSB_MAX_COMMENT_REFRESH_POSTS", "75")))
    parser.add_argument("--sentiment-cache", default=os.getenv("FINBERT_SENTIMENT_CACHE", "finbert_sentiment_cache.json"))
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing store and scrape the full window.")
    parser.add_argument("--skip-analysis", action="store_true", help="Only update the rolling Reddit post store/state; skip FinBERT sentiment output.")
    return parser.parse_args()


def main() -> None:
    _load_env()
    _require_reddit_env()

    args = parse_args()

    from getdata import (
        get_recent_wsb_comments,
        get_recent_wsb_posts,
        get_wsb_posts,
        refresh_active_post_comments,
        refresh_post_scores,
    )
    import analyze_wsb

    store_path = Path(args.store_file)
    output_path = Path(args.output)
    state_path = Path(args.state_file)
    use_database = args.storage == "db"
    db_store = None

    if use_database:
        import db_store as db_module
        db_store = db_module
        db_store.init_db()
        if not args.rebuild and db_store.count_posts() == 0 and store_path.exists():
            seed_posts = _load_posts(store_path)
            if seed_posts:
                print(f"Seeding database from {store_path} ({len(seed_posts)} posts)...")
                db_store.save_posts(seed_posts, subreddit=args.subreddit)
        existing_posts = [] if args.rebuild else db_store.load_posts(window_days=args.window_days)
    else:
        existing_posts = [] if args.rebuild else _load_posts(store_path)

    if args.rebuild:
        scrape_days = args.window_days
        print(f"Scraping r/{args.subreddit} for the last {scrape_days:g} day(s)...")
        scraped_posts = get_wsb_posts(
            subreddit=args.subreddit,
            days=scrape_days,
            request_delay=args.request_delay,
        )
        incoming_posts = scraped_posts
    else:
        state = db_store.load_state() if use_database and db_store else _load_state(state_path)
        fallback_since = (datetime.now() - timedelta(days=args.scrape_days)).timestamp()
        overlap_seconds = max(args.overlap_minutes, 0) * 60
        post_since = float(state.get("last_post_seen_at", fallback_since)) - overlap_seconds
        comment_since = float(state.get("last_comment_seen_at", fallback_since)) - overlap_seconds

        print(
            f"Incremental scrape r/{args.subreddit}: posts since "
            f"{datetime.fromtimestamp(post_since).strftime(TS_FORMAT)}, comments since "
            f"{datetime.fromtimestamp(comment_since).strftime(TS_FORMAT)}."
        )
        new_posts = get_recent_wsb_posts(
            subreddit=args.subreddit,
            since_ts=post_since,
            request_delay=args.request_delay,
        )
        comment_posts = get_recent_wsb_comments(
            subreddit=args.subreddit,
            since_ts=comment_since,
            request_delay=args.request_delay,
        )
        active_comment_posts = refresh_active_post_comments(
            existing_posts,
            subreddit=args.subreddit,
            active_days=args.comment_refresh_days,
            max_posts=args.max_comment_refresh_posts,
            request_delay=args.request_delay,
        )
        refreshed_scores = refresh_post_scores(
            existing_posts,
            subreddit=args.subreddit,
            active_days=args.score_refresh_days,
            max_posts=args.max_score_refresh,
            request_delay=args.request_delay,
        )
        incoming_posts = _merge_posts(_merge_posts(new_posts, comment_posts), active_comment_posts)
        print(
            f"Incremental scrape found {len(new_posts)} new/recent posts, "
            f"{sum(len(p.get('comments') or []) for p in comment_posts)} new/recent comments, "
            f"refreshed comment trees for {len(active_comment_posts)} active posts, "
            f"and refreshed scores for {refreshed_scores} active posts."
        )

    merged_posts = _merge_posts(existing_posts, incoming_posts)
    rolling_posts = _prune_posts(merged_posts, args.window_days)

    if use_database and db_store:
        db_store.save_posts(rolling_posts, subreddit=args.subreddit)
        db_store.prune_old(args.window_days)
        rolling_posts = db_store.load_posts(window_days=args.window_days)

    _write_posts(store_path, rolling_posts)

    latest_post = _latest_post_ts(rolling_posts)
    latest_comment = _latest_comment_ts(rolling_posts)
    if latest_post or latest_comment:
        next_state = {
            "last_successful_run_at": datetime.now().timestamp(),
            "last_post_seen_at": latest_post or datetime.now().timestamp(),
            "last_comment_seen_at": latest_comment or latest_post or datetime.now().timestamp(),
            "window_days": args.window_days,
        }
        if use_database and db_store:
            db_store.write_state(next_state)
        else:
            _write_state(state_path, next_state)

    print(
        f"Post store ({'database' if use_database else 'json'}): {len(existing_posts)} existing + {len(incoming_posts)} incoming "
        f"-> {len(rolling_posts)} kept over {args.window_days:g} day(s)."
    )

    if args.skip_analysis:
        print("Skipping FinBERT sentiment analysis; rolling Reddit store/state updated only.")
        return

    print(f"Analyzing sentiment into {output_path}...")
    analyze_wsb.DECAY_WINDOW_DAYS = args.aggregate_days
    analyze_wsb.run(
        input_file=str(store_path),
        output_file=str(output_path),
        daily_sentiment_file=args.daily_output,
        aggregate_window_days=args.aggregate_days,
        finbert_model=args.finbert_model,
        batch_size=args.batch_size,
        min_mentions=args.min_mentions,
        min_confidence=args.min_confidence,
        sentiment_cache_file=args.sentiment_cache,
    )

    print(f"Done. Updated {output_path} and {args.daily_output} from rolling store {store_path}.")


if __name__ == "__main__":
    main()
