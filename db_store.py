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


def count_posts(url: str | None = None) -> int:
    with connect(url) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM reddit_posts")
        return int(cur.fetchone()[0])


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
