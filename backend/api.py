import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

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

# Path to ticker_sentiment.json — lives in wsb/, one level up from backend/
SENTIMENT_FILE = os.path.join(os.path.dirname(__file__), "..", "ticker_sentiment.json")
TOP_N = 20


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

def _company_name(ticker_row: dict) -> str:
    value = ticker_row.get("company") or ticker_row.get("company_name") or ""
    return str(value)


def _load_tickers() -> list[dict]:
    path = os.path.abspath(SENTIMENT_FILE)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"ticker_sentiment.json not found at {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["tickers"]
    except (KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Malformed sentiment file: {e}")


def _bullish_score(t: dict) -> float:
    """
    Continuous ranking — no hard filter by overall_sentiment.
    Primary signal: sentiment_ratio (% bullish mentions).
    This naturally puts confirmed bullish tickers first, then neutral,
    then bearish at the bottom — ordered by least bearish among those.
    Multiplied by normalized_semantic_score magnitude to separate tickers
    with identical ratios by conviction strength.
    """
    return t["sentiment_ratio"] * (1 + abs(t["normalized_semantic_score"]))


def _bearish_score(t: dict) -> float:
    """
    Mirror of bullish — ranks by highest bearish % first (1 - sentiment_ratio),
    then least bullish as the fallback. Tickers with identical bearish ratios
    are separated by conviction strength.
    """
    bearish_ratio = 1.0 - t["sentiment_ratio"]
    return bearish_ratio * (1 + abs(t["normalized_semantic_score"]))


def _format_ticker(t: dict, score: float) -> dict:
    n = t["mentions"]
    # True three-way split — these three always sum to 100%
    bullish_pct = round(t["positive_count"] / n * 100, 1) if n else 0
    bearish_pct = round(t["negative_count"] / n * 100, 1) if n else 0
    neutral_pct = round(100 - bullish_pct - bearish_pct, 1)

    return {
        "ticker":           t["ticker"],
        "company":          _company_name(t),
        "mentions":         n,
        "sentiment":        t["overall_sentiment"],
        "bullish_pct":      bullish_pct,
        "bearish_pct":      bearish_pct,
        "neutral_pct":      neutral_pct,
        "normalized_score": round(t["normalized_semantic_score"], 4),
        "score":            round(score, 4),
    }


@app.get("/top-posts")
async def top_posts():
    tickers = _load_tickers()

    trending = sorted(tickers, key=lambda t: t["mentions"],  reverse=True)[:TOP_N]
    bullish  = sorted(tickers, key=_bullish_score,           reverse=True)[:TOP_N]
    bearish  = sorted(tickers, key=_bearish_score,           reverse=True)[:TOP_N]

    return {
        "trending": [_format_ticker(t, t["mentions"])     for t in trending],
        "bullish":  [_format_ticker(t, _bullish_score(t)) for t in bullish],
        "bearish":  [_format_ticker(t, _bearish_score(t)) for t in bearish],
        "meta":     {"total_tickers": len(tickers)},
    }