import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from functools import lru_cache

app = FastAPI()

_inference_module = None


def _get_inference():
    global _inference_module
    if _inference_module is None:
        import inference
        _inference_module = inference
    return _inference_module


DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sentiment JSON files live in wsb/, one level up from backend/.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SENTIMENT_FILE = os.path.join(ROOT_DIR, "ticker_sentiment.json")
SENTIMENT_WINDOW_FILES = {
    1: os.path.join(ROOT_DIR, "ticker_sentiment_1d.json"),
    3: os.path.join(ROOT_DIR, "ticker_sentiment_3d.json"),
    7: os.path.join(ROOT_DIR, "ticker_sentiment_7d.json"),
    14: SENTIMENT_FILE,
}
AVAILABLE_SENTIMENT_WINDOWS = tuple(sorted(SENTIMENT_WINDOW_FILES))
COMPANY_CACHE_FILE = os.path.join(ROOT_DIR, "ticker_company_names.json")
TOP_N = int(os.getenv("TOP_POSTS_LIMIT", "100"))
API_STORAGE = os.getenv("WSB_API_STORAGE", "json").strip().lower()
PORTFOLIO_JSON_FILE = os.path.join(ROOT_DIR, "frontend", "src", "portfolio_data", "portfolio_total_investment.json")


def _alias_company_names() -> dict[str, str]:
    try:
        import analyze_wsb
    except Exception:
        return {}

    names: dict[str, str] = {}
    for alias, ticker in getattr(analyze_wsb, "MANUAL_ALIASES", {}).items():
        symbol = str(ticker).upper().strip()
        if not symbol:
            continue
        candidate = str(alias).replace("  ", " ").strip().title()
        current = names.get(symbol)
        if current is None or len(candidate) > len(current):
            names[symbol] = candidate
    return names


COMPANY_NAME_FALLBACKS = _alias_company_names()


def _load_company_cache() -> dict[str, str]:
    try:
        with open(COMPANY_CACHE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k).upper(): str(v) for k, v in raw.items() if v}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_company_cache(cache: dict[str, str]) -> None:
    try:
        with open(COMPANY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(cache.items())), f, indent=2)
    except OSError:
        pass


COMPANY_NAME_CACHE = _load_company_cache()
MAJOR_ETF_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "ARKK", "SQQQ", "TQQQ",
    "SOXL", "SOXS", "XLF", "XLE", "XLK", "XBI", "SMH", "TLT", "HYG",
}


@lru_cache(maxsize=1024)
def _lookup_company_name(symbol: str) -> str:
    ticker = symbol.upper().strip()
    if not ticker:
        return ""

    cached = COMPANY_NAME_CACHE.get(ticker)
    if cached:
        return cached

    fallback = COMPANY_NAME_FALLBACKS.get(ticker, "")

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).get_info()
        value = info.get("longName") or info.get("shortName") or info.get("displayName") or fallback
    except Exception:
        value = fallback

    if value:
        cleaned = _clean_company_name(str(value))
        COMPANY_NAME_CACHE[ticker] = cleaned
        _save_company_cache(COMPANY_NAME_CACHE)
        return cleaned
    return ""


# ─── Existing chat endpoints ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    def stream():
        inference = _get_inference()
        reply = inference.chat(req.message)
        for char in reply:
            yield f"data: {json.dumps(char)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/chat/status")
async def chat_status():
    return _get_inference().provider_status()


@app.get("/history")
async def history():
    return {
        "history": [
            {"role": m["role"], "text": m.get("display", m["content"])}
            for m in _get_inference().messages if m["role"] != "system"
        ]
    }


@app.post("/reset")
async def reset():
    inference = _get_inference()
    inference.messages[:] = [{"role": "system", "content": inference.SYSTEM_PROMPT}]
    return {"status": "ok"}


# ─── Sentiment endpoints ──────────────────────────────────────────────────────

def _clean_company_name(value: str) -> str:
    text = str(value or "").strip()
    if " - " in text:
        text = text.split(" - ", 1)[0].strip()
    for suffix in (
        " Common Stock",
        " Ordinary Shares",
        " American Depositary Shares",
        " American Depositary Receipt",
    ):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


def _asset_type(ticker_row: dict) -> str:
    value = ticker_row.get("asset_type") or ticker_row.get("type") or ticker_row.get("quote_type")
    if value:
        text = str(value).strip().upper()
        if text in {"ETF", "FUND"}:
            return "ETF" if text == "ETF" else "Fund"
        if text in {"EQUITY", "COMMON STOCK", "STOCK"}:
            return "Company"
        return str(value).strip()
    symbol = str(ticker_row.get("ticker", "")).upper().strip()
    if symbol in MAJOR_ETF_TICKERS:
        return "ETF"
    return "Company"


def _company_name(ticker_row: dict, allow_lookup: bool = False) -> str:
    value = ticker_row.get("company") or ticker_row.get("company_name") or ticker_row.get("name")
    if value:
        return _clean_company_name(str(value))
    symbol = str(ticker_row.get("ticker", "")).upper().strip()
    if allow_lookup:
        return _lookup_company_name(symbol)
    return COMPANY_NAME_CACHE.get(symbol) or COMPANY_NAME_FALLBACKS.get(symbol, "")


def _sentiment_file_for_window(window_days: int) -> str:
    try:
        normalized = int(window_days)
    except (TypeError, ValueError):
        normalized = 14
    if normalized not in SENTIMENT_WINDOW_FILES:
        allowed = ", ".join(str(day) for day in AVAILABLE_SENTIMENT_WINDOWS)
        raise HTTPException(status_code=400, detail=f"window_days must be one of: {allowed}")
    return SENTIMENT_WINDOW_FILES[normalized]


@lru_cache(maxsize=8)
def _read_sentiment_payload(path: str, mtime_ns: int) -> dict:
    _ = mtime_ns
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_sentiment_payload(window_days: int = 14) -> dict:
    if API_STORAGE == "db":
        try:
            import db_store
            db_store.init_db()
            payload = db_store.load_sentiment_snapshot(window_days)
            if payload and "tickers" in payload:
                return payload
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load sentiment from database: {e}")

    path = os.path.abspath(_sentiment_file_for_window(window_days))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"sentiment file not found at {path}")
    try:
        stat = os.stat(path)
        payload = _read_sentiment_payload(path, stat.st_mtime_ns)
        if "tickers" not in payload:
            raise KeyError("tickers")
        return payload
    except (OSError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Malformed sentiment file: {e}")


def _load_tickers(window_days: int = 14) -> list[dict]:
    return _load_sentiment_payload(window_days)["tickers"]


def _sentiment_window_days(meta: dict) -> float:
    value = meta.get("aggregate_window_days") or (meta.get("recency_decay") or {}).get("window_days")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(os.getenv("SENTIMENT_AGGREGATE_WINDOW_DAYS", "14"))


def _bullish_score(t: dict) -> tuple[float, int]:
    return (float(t.get("semantic_score", 0)), int(t.get("mentions", 0)))


def _bearish_score(t: dict) -> tuple[float, int]:
    return (-float(t.get("semantic_score", 0)), int(t.get("mentions", 0)))


def _scaled_sentiment_split(t: dict) -> tuple[float, float, float]:
    n = int(t.get("mentions", 0))
    if n <= 0:
        return 0.0, 0.0, 0.0

    positive = int(t.get("positive_count", 0))
    negative = int(t.get("negative_count", 0))
    directional = positive + negative
    if directional <= 0:
        return 0.0, 0.0, 100.0

    raw_directional_share = directional / n
    scaled_directional_share = raw_directional_share ** 0.5
    scaled_directional_pct = scaled_directional_share * 100

    positive_share = positive / directional
    bullish_pct = round(scaled_directional_pct * positive_share, 1)
    bearish_pct = round(scaled_directional_pct * (1 - positive_share), 1)
    neutral_pct = round(max(0.0, 100 - bullish_pct - bearish_pct), 1)
    return bullish_pct, bearish_pct, neutral_pct


def _raw_sentiment_split(t: dict) -> tuple[float, float, float]:
    n = int(t.get("mentions", 0))
    if n <= 0:
        return 0.0, 0.0, 0.0
    bullish_pct = round(int(t.get("positive_count", 0)) / n * 100, 1)
    bearish_pct = round(int(t.get("negative_count", 0)) / n * 100, 1)
    neutral_pct = round(max(0.0, 100 - bullish_pct - bearish_pct), 1)
    return bullish_pct, bearish_pct, neutral_pct


def _format_ticker(t: dict, rank: int, score: float) -> dict:
    n = t["mentions"]
    bullish_pct, bearish_pct, neutral_pct = _scaled_sentiment_split(t)
    raw_bullish_pct, raw_bearish_pct, raw_neutral_pct = _raw_sentiment_split(t)

    return {
        "rank":             rank,
        "ticker":           t["ticker"],
        "company":          _company_name(t),
        "asset_type":       _asset_type(t),
        "mentions":         n,
        "sentiment":        t["overall_sentiment"],
        "bullish_pct":      bullish_pct,
        "bearish_pct":      bearish_pct,
        "neutral_pct":      neutral_pct,
        "raw_bullish_pct":  raw_bullish_pct,
        "raw_bearish_pct":  raw_bearish_pct,
        "raw_neutral_pct":  raw_neutral_pct,
        "normalized_score": round(t["normalized_semantic_score"], 4),
        "score":            round(score, 4),
    }


@app.get("/top-posts")
async def top_posts(
    limit: int = Query(TOP_N, ge=1, le=5000),
    window_days: int = Query(14, ge=1, le=14),
):
    payload = _load_sentiment_payload(window_days)
    tickers = payload["tickers"]
    source_meta = payload.get("meta") or {}

    trending = sorted(tickers, key=lambda t: int(t.get("mentions", 0)), reverse=True)[:limit]
    bullish = sorted(
        (t for t in tickers if t.get("overall_sentiment") == "bullish"),
        key=_bullish_score,
        reverse=True,
    )[:limit]
    bearish = sorted(
        (t for t in tickers if t.get("overall_sentiment") == "bearish"),
        key=_bearish_score,
        reverse=True,
    )[:limit]

    return {
        "trending": [
            _format_ticker(t, i + 1, float(t.get("mentions", 0)))
            for i, t in enumerate(trending)
        ],
        "bullish": [
            _format_ticker(t, i + 1, _bullish_score(t)[0])
            for i, t in enumerate(bullish)
        ],
        "bearish": [
            _format_ticker(t, i + 1, -_bearish_score(t)[0])
            for i, t in enumerate(bearish)
        ],
        "meta": {
            "total_tickers": len(tickers),
            "limit": limit,
            "sentiment_window_days": _sentiment_window_days(source_meta),
            "selected_window_days": window_days,
            "available_window_days": list(AVAILABLE_SENTIMENT_WINDOWS),
            "source_total_posts": source_meta.get("total_posts"),
            "aggregate_posts": source_meta.get("aggregate_posts"),
            "aggregate_comments": source_meta.get("aggregate_comments"),
            "aggregate_items_with_mentions": source_meta.get("aggregate_items_with_mentions"),
        },
    }




def _load_portfolio_payload() -> dict:
    if API_STORAGE == "db":
        try:
            import db_store
            db_store.init_db()
            payload = db_store.load_portfolio_result()
            if payload:
                return payload
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load portfolio from database: {e}")

    if not os.path.exists(PORTFOLIO_JSON_FILE):
        raise HTTPException(status_code=404, detail=f"portfolio file not found at {PORTFOLIO_JSON_FILE}")
    try:
        with open(PORTFOLIO_JSON_FILE, encoding="utf-8") as f:
            payload = json.load(f)
        if "portfolio_statistics" not in payload:
            raise KeyError("portfolio_statistics")
        return payload
    except (OSError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Malformed portfolio file: {e}")


@app.get("/portfolio")
async def portfolio_summary():
    return _load_portfolio_payload()


@app.get("/portfolio/{day}")
async def portfolio_day(day: str):
    payload = _load_portfolio_payload()
    for row in payload.get("daily_data") or []:
        if str(row.get("date")) == day:
            return row
    raise HTTPException(status_code=404, detail=f"Portfolio day {day} not found")


@app.get("/ticker/{ticker}")
async def ticker_detail(ticker: str):
    symbol = ticker.upper().strip()
    tickers = _load_tickers()
    row = next((item for item in tickers if str(item.get("ticker", "")).upper() == symbol), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")

    def sample_payload(sample: dict) -> dict:
        text = str(sample.get("text") or "")
        return {
            "source": sample.get("source", "unknown"),
            "post_id": sample.get("post_id"),
            "comment_id": sample.get("comment_id"),
            "permalink": sample.get("permalink"),
            "text": text,
            "upvotes": sample.get("upvotes", 0),
            "awards": sample.get("awards", 0),
            "created_utc": sample.get("created_utc"),
            "sentiment": sample.get("sentiment"),
            "sentiment_score": sample.get("sentiment_score"),
            "semantic_value": sample.get("semantic_value"),
            "confidence": sample.get("confidence"),
            "method": sample.get("method"),
        }

    bullish = [sample_payload(s) for s in row.get("top_bullish_posts", [])]
    bearish = [sample_payload(s) for s in row.get("top_bearish_posts", [])]

    return {
        "ticker": symbol,
        "company": _company_name(row, allow_lookup=True),
        "asset_type": _asset_type(row),
        "summary": _format_ticker(row, 1, float(row.get("semantic_score", 0))),
        "semantic_score": row.get("semantic_score", 0),
        "positive_score": row.get("positive_score", 0),
        "negative_score": row.get("negative_score", 0),
        "avg_confidence": row.get("avg_confidence", 0),
        "detection_methods": row.get("detection_methods", {}),
        "positive_samples": bullish,
        "negative_samples": bearish,
        "has_positive_sample": len(bullish) > 0,
        "has_negative_sample": len(bearish) > 0,
    }
