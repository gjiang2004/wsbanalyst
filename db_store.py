import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_SQLITE_PATH = "wsb_data.sqlite3"


def database_url() -> str:
    return os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")


def using_postgres(url: str | None = None) -> bool:
    value = url or database_url()
    return value.startswith("postgres://") or value.startswith("postgresql://")


def _sqlite_path(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "sqlite":
        return DEFAULT_SQLITE_PATH
    if parsed.path in ("", "/"):
        return DEFAULT_SQLITE_PATH
    if parsed.netloc:
        return f"/{parsed.netloc}{parsed.path}"
    return parsed.path.lstrip("/") or DEFAULT_SQLITE_PATH


@contextmanager
def connect(url: str | None = None) -> Iterator[Any]:
    value = url or database_url()
    if using_postgres(value):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("Postgres DATABASE_URL requires psycopg. Install requirements.txt first.") from exc
        conn = psycopg.connect(value, row_factory=dict_row)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return

    path = Path(_sqlite_path(value))
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _placeholder(url: str | None = None) -> str:
    return "%s" if using_postgres(url) else "?"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def parse_created_at(value: Any) -> datetime | None:
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


def _post_created(post: dict) -> str | None:
    created = parse_created_at(post.get("created_at") or post.get("created_utc"))
    return created.strftime(TS_FORMAT) if created else None


def _comment_created(comment: dict) -> str | None:
    created = parse_created_at(comment.get("created_at") or comment.get("created_utc"))
    return created.strftime(TS_FORMAT) if created else None


def init_db(url: str | None = None) -> None:
    value = url or database_url()
    postgres = using_postgres(value)
    post_raw_type = "JSONB" if postgres else "TEXT"
    comment_raw_type = "JSONB" if postgres else "TEXT"
    payload_type = "JSONB" if postgres else "TEXT"
    now_expr = "CURRENT_TIMESTAMP"

    with connect(value) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS reddit_posts (
                id TEXT PRIMARY KEY,
                subreddit TEXT NOT NULL DEFAULT 'wallstreetbets',
                author TEXT,
                title TEXT,
                text TEXT,
                flair TEXT,
                upvotes INTEGER DEFAULT 0,
                upvote_ratio REAL,
                num_comments INTEGER DEFAULT 0,
                awards INTEGER DEFAULT 0,
                created_at TIMESTAMP,
                permalink TEXT,
                raw_json {post_raw_type},
                updated_at TIMESTAMP DEFAULT {now_expr}
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS reddit_comments (
                comment_id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                parent_id TEXT,
                author TEXT,
                body TEXT,
                score INTEGER DEFAULT 0,
                awards INTEGER DEFAULT 0,
                created_at TIMESTAMP,
                raw_json {comment_raw_type},
                updated_at TIMESTAMP DEFAULT {now_expr}
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reddit_posts_created ON reddit_posts(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reddit_comments_created ON reddit_comments(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reddit_comments_post ON reddit_comments(post_id)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scrape_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS sentiment_snapshots (
                window_days INTEGER PRIMARY KEY,
                payload {payload_type} NOT NULL,
                generated_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT {now_expr}
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_snapshots_updated ON sentiment_snapshots(updated_at)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_ticker_sentiment (
                day TEXT NOT NULL,
                ticker TEXT NOT NULL,
                refined_sentiment REAL NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (day, ticker)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_ticker_sentiment_day ON daily_ticker_sentiment(day)")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS portfolio_runs (
                id TEXT PRIMARY KEY,
                payload {payload_type} NOT NULL,
                generated_at TIMESTAMP DEFAULT {now_expr},
                updated_at TIMESTAMP DEFAULT {now_expr}
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS portfolio_daily_values (
                run_id TEXT NOT NULL,
                day TEXT NOT NULL,
                investment REAL,
                today_profit REAL,
                total_profit REAL,
                payload {payload_type} NOT NULL,
                updated_at TIMESTAMP DEFAULT {now_expr},
                PRIMARY KEY (run_id, day)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_daily_values_day ON portfolio_daily_values(day)")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS portfolio_trades (
                run_id TEXT NOT NULL,
                day TEXT NOT NULL,
                trade_index INTEGER NOT NULL,
                trade_type TEXT NOT NULL,
                ticker TEXT,
                side TEXT,
                payload {payload_type} NOT NULL,
                updated_at TIMESTAMP DEFAULT {now_expr},
                PRIMARY KEY (run_id, day, trade_type, trade_index)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_trades_ticker ON portfolio_trades(ticker)")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS finbert_cache (
                cache_key TEXT PRIMARY KEY,
                payload {payload_type} NOT NULL,
                updated_at TIMESTAMP DEFAULT {now_expr}
            )
        """)


def count_posts(url: str | None = None) -> int:
    with connect(url) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM reddit_posts")
        return int(_row_get(cur.fetchone(), "n", 0) or 0)


def save_posts(posts: list[dict], subreddit: str = "wallstreetbets", url: str | None = None) -> None:
    value = url or database_url()
    p = _placeholder(value)
    postgres = using_postgres(value)
    post_raw = "CAST(%s AS JSONB)" if postgres else p
    comment_raw = "CAST(%s AS JSONB)" if postgres else p
    post_sql = f"""
        INSERT INTO reddit_posts (
            id, subreddit, author, title, text, flair, upvotes, upvote_ratio,
            num_comments, awards, created_at, permalink, raw_json, updated_at
        ) VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{post_raw},CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            subreddit=excluded.subreddit,
            author=excluded.author,
            title=excluded.title,
            text=excluded.text,
            flair=excluded.flair,
            upvotes=excluded.upvotes,
            upvote_ratio=excluded.upvote_ratio,
            num_comments=excluded.num_comments,
            awards=excluded.awards,
            created_at=excluded.created_at,
            permalink=excluded.permalink,
            raw_json=excluded.raw_json,
            updated_at=CURRENT_TIMESTAMP
    """
    comment_sql = f"""
        INSERT INTO reddit_comments (
            comment_id, post_id, parent_id, author, body, score, awards,
            created_at, raw_json, updated_at
        ) VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{comment_raw},CURRENT_TIMESTAMP)
        ON CONFLICT(comment_id) DO UPDATE SET
            post_id=excluded.post_id,
            parent_id=excluded.parent_id,
            author=excluded.author,
            body=excluded.body,
            score=excluded.score,
            awards=excluded.awards,
            created_at=excluded.created_at,
            raw_json=excluded.raw_json,
            updated_at=CURRENT_TIMESTAMP
    """
    with connect(value) as conn:
        cur = conn.cursor()
        for post in posts:
            post_id = str(post.get("id") or "").strip()
            if not post_id:
                continue
            comments = post.get("comments") or []
            raw_post = {**post, "comments": []}
            cur.execute(post_sql, (
                post_id,
                subreddit,
                post.get("author"),
                post.get("title"),
                post.get("text"),
                post.get("flair"),
                int(post.get("upvotes") or 0),
                post.get("upvote_ratio"),
                int(post.get("num_comments") or len(comments) or 0),
                int(post.get("awards") or 0),
                _post_created(post),
                post.get("permalink"),
                _json(raw_post),
            ))
            for comment in comments:
                comment_id = str(comment.get("comment_id") or comment.get("id") or "").strip()
                if not comment_id:
                    continue
                cur.execute(comment_sql, (
                    comment_id,
                    post_id,
                    comment.get("parent_id"),
                    comment.get("author"),
                    comment.get("body"),
                    int(comment.get("score") or 0),
                    int(comment.get("awards") or 0),
                    _comment_created(comment),
                    _json(comment),
                ))


def load_posts(window_days: float | None = None, url: str | None = None) -> list[dict]:
    value = url or database_url()
    p = _placeholder(value)
    cutoff = None
    if window_days is not None:
        cutoff = (datetime.now() - timedelta(days=window_days)).strftime(TS_FORMAT)

    with connect(value) as conn:
        cur = conn.cursor()
        if cutoff:
            cur.execute(
                f"""
                SELECT * FROM reddit_posts
                WHERE created_at IS NULL OR created_at >= {p}
                   OR id IN (SELECT post_id FROM reddit_comments WHERE created_at IS NULL OR created_at >= {p})
                ORDER BY created_at DESC
                """,
                (cutoff, cutoff),
            )
        else:
            cur.execute("SELECT * FROM reddit_posts ORDER BY created_at DESC")
        post_rows = cur.fetchall()
        post_ids = [row["id"] for row in post_rows]
        comments_by_post: dict[str, list[dict]] = {post_id: [] for post_id in post_ids}
        if post_ids:
            placeholders = ",".join([p] * len(post_ids))
            params: tuple[Any, ...]
            if cutoff:
                params = (*post_ids, cutoff)
                cur.execute(
                    f"""
                    SELECT * FROM reddit_comments
                    WHERE post_id IN ({placeholders})
                      AND (created_at IS NULL OR created_at >= {p})
                    ORDER BY post_id, score DESC, created_at DESC
                    """,
                    params,
                )
            else:
                params = tuple(post_ids)
                cur.execute(
                    f"""
                    SELECT * FROM reddit_comments
                    WHERE post_id IN ({placeholders})
                    ORDER BY post_id, score DESC, created_at DESC
                    """,
                    params,
                )
            for row in cur.fetchall():
                comments_by_post[row["post_id"]].append(_comment_from_row(row))

    posts = []
    for row in post_rows:
        post = _post_from_row(row)
        post["comments"] = comments_by_post.get(post["id"], [])
        posts.append(post)
    return posts


def prune_old(window_days: float, url: str | None = None) -> None:
    value = url or database_url()
    p = _placeholder(value)
    cutoff = (datetime.now() - timedelta(days=window_days)).strftime(TS_FORMAT)
    with connect(value) as conn:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM reddit_comments WHERE created_at IS NOT NULL AND created_at < {p}", (cutoff,))
        cur.execute(
            f"""
            DELETE FROM reddit_posts
            WHERE created_at IS NOT NULL AND created_at < {p}
              AND id NOT IN (SELECT DISTINCT post_id FROM reddit_comments)
            """,
            (cutoff,),
        )


def load_state(url: str | None = None) -> dict[str, float]:
    with connect(url) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM scrape_state")
        state = {}
        for row in cur.fetchall():
            try:
                state[str(row["key"])] = float(row["value"])
            except (TypeError, ValueError):
                continue
        return state


def write_state(state: dict[str, float], url: str | None = None) -> None:
    value = url or database_url()
    p = _placeholder(value)
    sql = f"""
        INSERT INTO scrape_state (key, value, updated_at) VALUES ({p},{p},CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
    """
    with connect(value) as conn:
        cur = conn.cursor()
        for key, value_item in state.items():
            cur.execute(sql, (key, str(value_item)))


def _post_from_row(row: Any) -> dict:
    raw = _loads(row["raw_json"], {})
    post = dict(raw) if isinstance(raw, dict) else {}
    post.update({
        "id": row["id"],
        "author": row["author"],
        "title": row["title"],
        "text": row["text"],
        "flair": row["flair"],
        "upvotes": row["upvotes"],
        "upvote_ratio": row["upvote_ratio"],
        "num_comments": row["num_comments"],
        "awards": row["awards"],
        "created_at": str(row["created_at"]) if row["created_at"] is not None else None,
        "permalink": row["permalink"],
    })
    return post


def _comment_from_row(row: Any) -> dict:
    raw = _loads(row["raw_json"], {})
    comment = dict(raw) if isinstance(raw, dict) else {}
    comment.update({
        "comment_id": row["comment_id"],
        "parent_id": row["parent_id"],
        "author": row["author"],
        "body": row["body"],
        "score": row["score"],
        "awards": row["awards"],
        "created_at": str(row["created_at"]) if row["created_at"] is not None else None,
    })
    return comment


def _json_expr(value: str, url: str | None = None) -> str:
    return f"CAST({value} AS JSONB)" if using_postgres(url) else value


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def save_sentiment_snapshot(window_days: int, payload: dict, url: str | None = None) -> None:
    value = url or database_url()
    p = _placeholder(value)
    payload_expr = _json_expr(p, value)
    generated_at = (payload.get("meta") or {}).get("generated_at")
    sql = f"""
        INSERT INTO sentiment_snapshots (window_days, payload, generated_at, updated_at)
        VALUES ({p},{payload_expr},{p},CURRENT_TIMESTAMP)
        ON CONFLICT(window_days) DO UPDATE SET
            payload=excluded.payload,
            generated_at=excluded.generated_at,
            updated_at=CURRENT_TIMESTAMP
    """
    with connect(value) as conn:
        conn.cursor().execute(sql, (int(window_days), _json(payload), generated_at))


def load_sentiment_snapshot(window_days: int, url: str | None = None) -> dict | None:
    value = url or database_url()
    p = _placeholder(value)
    with connect(value) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT payload FROM sentiment_snapshots WHERE window_days = {p}", (int(window_days),))
        row = cur.fetchone()
    return _loads(_row_get(row, "payload"), None) if row else None


def save_daily_sentiment_rows(rows: list[dict], replace_from_day: str | None = None, url: str | None = None) -> None:
    value = url or database_url()
    p = _placeholder(value)
    sql = f"""
        INSERT INTO daily_ticker_sentiment (day, ticker, refined_sentiment, updated_at)
        VALUES ({p},{p},{p},CURRENT_TIMESTAMP)
        ON CONFLICT(day, ticker) DO UPDATE SET
            refined_sentiment=excluded.refined_sentiment,
            updated_at=CURRENT_TIMESTAMP
    """
    with connect(value) as conn:
        cur = conn.cursor()
        if replace_from_day:
            cur.execute(f"DELETE FROM daily_ticker_sentiment WHERE day >= {p}", (replace_from_day,))
        for row in rows:
            day = str(row.get("day") or "").strip()
            ticker = str(row.get("ticker") or "").upper().strip()
            if not day or not ticker:
                continue
            try:
                refined = float(row.get("refined_sentiment"))
            except (TypeError, ValueError):
                continue
            cur.execute(sql, (day, ticker, refined))


def load_daily_sentiment_rows(min_day: str | None = None, max_day: str | None = None, url: str | None = None) -> list[dict]:
    value = url or database_url()
    p = _placeholder(value)
    clauses = []
    params: list[Any] = []
    if min_day:
        clauses.append(f"day >= {p}")
        params.append(min_day)
    if max_day:
        clauses.append(f"day <= {p}")
        params.append(max_day)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect(value) as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT day, ticker, refined_sentiment FROM daily_ticker_sentiment {where} ORDER BY day, ticker",
            tuple(params),
        )
        rows = cur.fetchall()
    return [
        {
            "day": str(_row_get(row, "day")),
            "ticker": str(_row_get(row, "ticker")).upper(),
            "refined_sentiment": float(_row_get(row, "refined_sentiment", 0.0)),
        }
        for row in rows
    ]


def save_portfolio_result(result: dict, run_id: str = "default", url: str | None = None) -> None:
    value = url or database_url()
    p = _placeholder(value)
    payload_expr = _json_expr(p, value)
    run_sql = f"""
        INSERT INTO portfolio_runs (id, payload, generated_at, updated_at)
        VALUES ({p},{payload_expr},CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            payload=excluded.payload,
            updated_at=CURRENT_TIMESTAMP
    """
    daily_sql = f"""
        INSERT INTO portfolio_daily_values (
            run_id, day, investment, today_profit, total_profit, payload, updated_at
        ) VALUES ({p},{p},{p},{p},{p},{payload_expr},CURRENT_TIMESTAMP)
        ON CONFLICT(run_id, day) DO UPDATE SET
            investment=excluded.investment,
            today_profit=excluded.today_profit,
            total_profit=excluded.total_profit,
            payload=excluded.payload,
            updated_at=CURRENT_TIMESTAMP
    """
    trade_sql = f"""
        INSERT INTO portfolio_trades (
            run_id, day, trade_index, trade_type, ticker, side, payload, updated_at
        ) VALUES ({p},{p},{p},{p},{p},{p},{payload_expr},CURRENT_TIMESTAMP)
        ON CONFLICT(run_id, day, trade_type, trade_index) DO UPDATE SET
            ticker=excluded.ticker,
            side=excluded.side,
            payload=excluded.payload,
            updated_at=CURRENT_TIMESTAMP
    """
    daily_records = result.get("daily_data") or []
    with connect(value) as conn:
        cur = conn.cursor()
        cur.execute(run_sql, (run_id, _json(result)))
        cur.execute(f"DELETE FROM portfolio_daily_values WHERE run_id = {p}", (run_id,))
        cur.execute(f"DELETE FROM portfolio_trades WHERE run_id = {p}", (run_id,))
        stats_by_day = {
            str(row.get("date")): row
            for row in result.get("portfolio_statistics") or []
            if row.get("date")
        }
        for record in daily_records:
            day = str(record.get("date") or "").strip()
            if not day:
                continue
            stat = stats_by_day.get(day, {})
            investment = record.get("total_investment", stat.get("investment"))
            today_profit = record.get("today_profit", stat.get("today_profit"))
            total_profit = record.get("total_profit", stat.get("total_profit"))
            cur.execute(daily_sql, (run_id, day, investment, today_profit, total_profit, _json(record)))
            trades = []
            trades.extend(("entry", trade) for trade in record.get("entries") or [])
            trades.extend(("exit", trade) for trade in record.get("exits") or [])
            for index, (trade_type, trade) in enumerate(trades):
                cur.execute(trade_sql, (
                    run_id,
                    day,
                    index,
                    trade_type,
                    trade.get("ticker"),
                    trade.get("side"),
                    _json(trade),
                ))


def load_portfolio_result(run_id: str = "default", url: str | None = None) -> dict | None:
    value = url or database_url()
    p = _placeholder(value)
    with connect(value) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT payload FROM portfolio_runs WHERE id = {p}", (run_id,))
        row = cur.fetchone()
    return _loads(_row_get(row, "payload"), None) if row else None



def load_finbert_cache(url: str | None = None) -> dict[str, tuple[str, float]]:
    value = url or database_url()
    with connect(value) as conn:
        cur = conn.cursor()
        cur.execute("SELECT cache_key, payload FROM finbert_cache")
        rows = cur.fetchall()
    cache: dict[str, tuple[str, float]] = {}
    for row in rows:
        key = str(_row_get(row, "cache_key") or "")
        payload = _loads(_row_get(row, "payload"), {})
        if not key or not isinstance(payload, dict):
            continue
        try:
            cache[key] = (str(payload["label"]), float(payload["score"]))
        except (KeyError, TypeError, ValueError):
            continue
    return cache


def save_finbert_cache(cache: dict[str, tuple[str, float]], url: str | None = None) -> None:
    value = url or database_url()
    p = _placeholder(value)
    payload_expr = _json_expr(p, value)
    sql = f"""
        INSERT INTO finbert_cache (cache_key, payload, updated_at)
        VALUES ({p},{payload_expr},CURRENT_TIMESTAMP)
        ON CONFLICT(cache_key) DO UPDATE SET
            payload=excluded.payload,
            updated_at=CURRENT_TIMESTAMP
    """
    with connect(value) as conn:
        cur = conn.cursor()
        for key, (label, score) in cache.items():
            cur.execute(sql, (key, _json({"label": label, "score": score})))
