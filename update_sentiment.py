import argparse
import os
from pathlib import Path



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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape recent r/wallstreetbets posts and refresh ticker sentiment JSON."
    )
    parser.add_argument("--days", type=float, default=1.0, help="Lookback window in days.")
    parser.add_argument("--subreddit", default="wallstreetbets")
    parser.add_argument("--posts-file", default="wsb_posts_24h.json")
    parser.add_argument("--output", default=os.getenv("TICKER_SENTIMENT_FILE", "ticker_sentiment.json"))
    parser.add_argument("--finbert-model", default=os.getenv("FINBERT_MODEL", "ProsusAI/finbert"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("FINBERT_BATCH_SIZE", "4")))
    parser.add_argument("--min-mentions", type=int, default=int(os.getenv("MIN_TICKER_MENTIONS", "3")))
    parser.add_argument("--min-confidence", type=float, default=float(os.getenv("MIN_TICKER_CONFIDENCE", "0.65")))
    return parser.parse_args()


def main() -> None:
    _load_env()
    _require_reddit_env()

    args = parse_args()

    from getdata import get_wsb_posts, save_to_json
    from analyze_wsb import run as run_sentiment

    posts_path = Path(args.posts_file)
    output_path = Path(args.output)

    print(f"Scraping r/{args.subreddit} for the last {args.days:g} day(s)...")
    posts = get_wsb_posts(subreddit=args.subreddit, days=args.days)
    save_to_json(posts, str(posts_path))

    print(f"Analyzing sentiment into {output_path}...")
    run_sentiment(
        input_file=str(posts_path),
        output_file=str(output_path),
        finbert_model=args.finbert_model,
        batch_size=args.batch_size,
        min_mentions=args.min_mentions,
        min_confidence=args.min_confidence,
    )

    print(f"Done. Updated {output_path} from {len(posts)} scraped posts.")


if __name__ == "__main__":
    main()
