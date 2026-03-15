import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import math

from inference import chat, messages, SYSTEM_PROMPT

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
        reply = chat(req.message)
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
            for m in messages if m["role"] != "system"
        ]
    }


@app.post("/reset")
async def reset():
    messages[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return {"status": "ok"}


# ─── Sentiment endpoints ──────────────────────────────────────────────────────

# Company name lookup — covers the most commonly discussed tickers on WSB.
# Unknown tickers fall back to an empty string (frontend shows nothing).
COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.", "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corp.",
    "META": "Meta Platforms", "TSLA": "Tesla Inc.", "AVGO": "Broadcom Inc.",
    "ORCL": "Oracle Corp.", "AMD": "Advanced Micro Devices", "INTC": "Intel Corp.",
    "QCOM": "Qualcomm Inc.", "TSM": "TSMC", "ASML": "ASML Holding",
    "NFLX": "Netflix Inc.", "DIS": "Walt Disney Co.", "SPOT": "Spotify Technology",
    "WMT": "Walmart Inc.", "TGT": "Target Corp.", "COST": "Costco Wholesale",
    "AMZN": "Amazon.com Inc.", "BBY": "Best Buy Co.", "HD": "Home Depot Inc.",
    "GME": "GameStop Corp.", "AMC": "AMC Entertainment", "BB": "BlackBerry Ltd.",
    "PLTR": "Palantir Technologies", "COIN": "Coinbase Global", "HOOD": "Robinhood Markets",
    "SOFI": "SoFi Technologies", "PYPL": "PayPal Holdings", "V": "Visa Inc.",
    "MA": "Mastercard Inc.", "JPM": "JPMorgan Chase", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "BLK": "BlackRock Inc.", "BAC": "Bank of America",
    "WFC": "Wells Fargo", "C": "Citigroup Inc.", "IBKR": "Interactive Brokers",
    "RIVN": "Rivian Automotive", "LCID": "Lucid Group", "F": "Ford Motor Co.",
    "GM": "General Motors", "UBER": "Uber Technologies", "LYFT": "Lyft Inc.",
    "ABNB": "Airbnb Inc.", "SNAP": "Snap Inc.", "RDDT": "Reddit Inc.",
    "SNOW": "Snowflake Inc.", "SHOP": "Shopify Inc.", "CRM": "Salesforce Inc.",
    "NOW": "ServiceNow Inc.", "NET": "Cloudflare Inc.", "DDOG": "Datadog Inc.",
    "CRWD": "CrowdStrike Holdings", "OKTA": "Okta Inc.", "TWLO": "Twilio Inc.",
    "MDB": "MongoDB Inc.", "GTLB": "GitLab Inc.", "ADBE": "Adobe Inc.",
    "ORCL": "Oracle Corp.", "IBM": "IBM Corp.", "CSCO": "Cisco Systems",
    "BABA": "Alibaba Group", "BIDU": "Baidu Inc.", "NIO": "NIO Inc.",
    "PFE": "Pfizer Inc.", "MRNA": "Moderna Inc.", "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group", "CVS": "CVS Health", "ABBV": "AbbVie Inc.",
    "LLY": "Eli Lilly and Co.", "BA": "Boeing Co.", "XOM": "Exxon Mobil Corp.",
    "CVX": "Chevron Corp.", "CAT": "Caterpillar Inc.", "DE": "Deere & Co.",
    "RTX": "RTX Corp.", "LMT": "Lockheed Martin", "SPCE": "Virgin Galactic",
    "RKLB": "Rocket Lab USA", "ASTS": "AST SpaceMobile", "OKLO": "Oklo Inc.",
    "CVNA": "Carvana Co.", "HIMS": "Hims & Hers Health", "MSTR": "MicroStrategy",
    "MARA": "Marathon Digital", "RIOT": "Riot Platforms", "IBIT": "iShares Bitcoin ETF",
    "QQQ": "Invesco QQQ ETF", "SPY": "SPDR S&P 500 ETF", "VOO": "Vanguard S&P 500 ETF",
    "TQQQ": "ProShares UltraPro QQQ", "SOXL": "Direxion Semi Bull 3X",
    "VTI": "Vanguard Total Market ETF", "ARKK": "ARK Innovation ETF",
    "GLD": "SPDR Gold Shares", "SLV": "iShares Silver Trust",
    "USO": "United States Oil Fund", "XLE": "Energy Select SPDR ETF",
    "MVST": "Microvast Holdings", "OSK": "Oshkosh Corp.", "SNDK": "SanDisk Corp.",
    "WBD": "Warner Bros. Discovery", "DUOL": "Duolingo Inc.", "LULU": "Lululemon Athletica",
    "PANW": "Palo Alto Networks", "CRDO": "Credo Technology", "MRVL": "Marvell Technology",
    "AMAT": "Applied Materials", "LRCX": "Lam Research", "KLAC": "KLA Corp.",
    "MU": "Micron Technology", "WDC": "Western Digital", "STX": "Seagate Technology",
    "DELL": "Dell Technologies", "HPE": "Hewlett Packard Enterprise",
    "ZIM": "ZIM Integrated Shipping", "DHT": "DHT Holdings", "STNG": "Scorpio Tankers",
    "NVO": "Novo Nordisk", "NBIS": "Nebius Group", "CRWV": "CoreWeave Inc.",
    "IONQ": "IonQ Inc.", "RGTI": "Rigetti Computing", "QBTS": "D-Wave Quantum",
    "RBRK": "Rubrik Inc.", "WDAY": "Workday Inc.", "DOCU": "DocuSign Inc.",
    "DASH": "DoorDash Inc.", "LYFT": "Lyft Inc.", "EBAY": "eBay Inc.",
    "ETSY": "Etsy Inc.", "PINS": "Pinterest Inc.", "SNAP": "Snap Inc.",
    "TTD": "The Trade Desk", "ROKU": "Roku Inc.", "FUBO": "FuboTV Inc.",
    "SOUN": "SoundHound AI", "BBAI": "BigBear.ai", "ONDS": "Ondas Holdings",
    "KTOS": "Kratos Defense", "AVAV": "AeroVironment", "RCAT": "Red Cat Holdings",
    "DJT": "Trump Media & Technology", "MAGA": "Point Bridge America First ETF",
    "XRP": "Ripple (XRP ETF)", "ETH": "Ethereum (ETF proxy)", "BTC": "Bitcoin (ETF proxy)",
}


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
        "company":          COMPANY_NAMES.get(t["ticker"], ""),
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