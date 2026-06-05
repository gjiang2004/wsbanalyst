import os
import json
import time
import torch
import re
import warnings
from datetime import datetime, timezone

import yfinance as yf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

warnings.filterwarnings("ignore")
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

BASE_MODEL = os.getenv("WSB_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
FINETUNED_DIR = os.getenv(
    "WSB_FINETUNED_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "wsb-mistral-finetuned2"),
)

SYSTEM_PROMPT = (
    "You are a WallStreetBets-style chatbot. Sound casual, blunt, skeptical, and concise. "
    "Use only the verified stock and sentiment context provided in square brackets for prices, percentages, dates, ratios, and statistics. "
    "If verified context is missing, say you do not have live data for that exact fact instead of inventing it. "
    "Do not claim breaking news, exact prices, or exact returns unless they appear in the verified context. "
    "Never include URLs or links."
)

MARKET_KEYWORDS = re.compile(
    r"\b(market|dow|nasdaq|s&p|stock market|the market)\b", re.IGNORECASE
)

_DOLLAR_TICKER = re.compile(r"\$([A-Za-z]{1,5})\b")
_CAPS_WORD     = re.compile(r"\b[A-Z]{2,5}\b")
MAX_CANDIDATES = 5
SENTIMENT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_sentiment.json")
_STOCK_CACHE_TTL_SECONDS = int(os.getenv("STOCK_CACHE_TTL_SECONDS", "300"))
_stock_cache: dict[str, tuple[float, dict | None]] = {}
_sentiment_cache: tuple[float, list[dict]] | None = None
_SENTIMENT_CACHE_TTL_SECONDS = int(os.getenv("SENTIMENT_CACHE_TTL_SECONDS", "120"))


def search_ticker(query: str) -> str | None:
    try:
        results = yf.Search(query, max_results=1).quotes
        if results:
            return results[0].get("symbol")
    except Exception:
        pass
    return None


def extract_candidates(text: str) -> list[str]:
    found = []
    for m in _DOLLAR_TICKER.finditer(text):
        found.append(m.group(1).upper())
    for word in _CAPS_WORD.findall(text):
        found.append(word)
    if len(found) < MAX_CANDIDATES:
        words = re.findall(r"[a-z]{3,}", text.lower())
        for word in words:
            if len(found) >= MAX_CANDIDATES:
                break
            ticker = search_ticker(word)
            if ticker:
                found.append(ticker)
    return list(dict.fromkeys(found))[:MAX_CANDIDATES]



def get_stock_data(ticker: str) -> dict | None:
    ticker = ticker.upper()
    cached = _stock_cache.get(ticker)
    if cached and time.time() - cached[0] < _STOCK_CACHE_TTL_SECONDS:
        return cached[1]
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty or "Close" not in hist:
            _stock_cache[ticker] = (time.time(), None)
            return None
        current_price = round(float(hist["Close"].iloc[-1]), 2)
        first_price = float(hist["Close"].iloc[0])
        change_pct = round(((current_price - first_price) / first_price) * 100, 2) if first_price else 0.0
        fast_info = getattr(stock, "fast_info", {}) or {}
        result = {
            "ticker": ticker,
            "price": current_price,
            "1mo_change": f"{change_pct:+.2f}%",
            "52w_high": fast_info.get("yearHigh", "N/A"),
            "52w_low": fast_info.get("yearLow", "N/A"),
            "market_cap": fast_info.get("marketCap", "N/A"),
            "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist and not hist["Volume"].empty else "N/A",
            "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
        _stock_cache[ticker] = (time.time(), result)
        return result
    except Exception:
        _stock_cache[ticker] = (time.time(), None)
        return None


def _load_sentiment_rows() -> list[dict]:
    global _sentiment_cache
    if _sentiment_cache and time.time() - _sentiment_cache[0] < _SENTIMENT_CACHE_TTL_SECONDS:
        return _sentiment_cache[1]
    if not os.path.exists(SENTIMENT_FILE):
        return []
    try:
        with open(SENTIMENT_FILE, encoding="utf-8") as f:
            rows = json.load(f).get("tickers", [])
    except (OSError, json.JSONDecodeError):
        rows = []
    _sentiment_cache = (time.time(), rows)
    return rows


def _sample_line(label: str, sample: dict) -> str | None:
    text = re.sub(r"\s+", " ", str(sample.get("text") or "")).strip()
    if not text:
        return None
    return f"{label}: {text[:180]}"


def get_sentiment_context(tickers: list[str]) -> list[str]:
    rows = _load_sentiment_rows()
    if not rows:
        return []

    wanted = {t.upper() for t in tickers}
    by_ticker = {row.get("ticker", "").upper(): row for row in rows}
    lines = []
    for ticker in wanted:
        row = by_ticker.get(ticker)
        if not row:
            continue
        mentions = row.get("mentions", 0)
        sentiment = row.get("overall_sentiment", "unknown")
        bullish = row.get("positive_count", 0)
        bearish = row.get("negative_count", 0)
        neutral = row.get("neutral_count", 0)
        score = row.get("normalized_semantic_score", "N/A")
        sample_lines = []
        for sample in row.get("top_bullish_posts", [])[:1]:
            line = _sample_line("bullish sample", sample)
            if line:
                sample_lines.append(line)
        for sample in row.get("top_bearish_posts", [])[:1]:
            line = _sample_line("bearish sample", sample)
            if line:
                sample_lines.append(line)
        sample_text = "; ".join(sample_lines)
        lines.append(
            f"[WSB sentiment {ticker}: {sentiment}, mentions={mentions}, "
            f"bullish={bullish}, bearish={bearish}, neutral={neutral}, "
            f"normalized_score={score}"
            + (f", {sample_text}" if sample_text else "")
            + "]"
        )
    return lines

def get_market_summary() -> dict:
    results = {}
    for ticker in ["SPY", "QQQ", "DIA", "VIX"]:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                price  = round(float(hist["Close"].iloc[-1]), 2)
                prev   = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
                change = round(((price - prev) / prev) * 100, 2)
                results[ticker] = {"price": price, "day_change": f"{change:+.2f}%"}
        except Exception:
            pass
    return results


def fetch_context(user_message: str) -> str:
    lines = []
    candidates = extract_candidates(user_message)
    for ticker in candidates:
        data = get_stock_data(ticker)
        if data:
            lines.append(
                f"[{data['ticker']}: price=${data['price']}, "
                f"1mo={data['1mo_change']}, "
                f"52w_high=${data['52w_high']}, 52w_low=${data['52w_low']}, "
                f"market_cap={data['market_cap']}, volume={data['volume']}, as_of={data['as_of']}]"
            )
    lines.extend(get_sentiment_context(candidates))
    if MARKET_KEYWORDS.search(user_message):
        mkt = get_market_summary()
        if mkt:
            parts = [f"{t}: ${v['price']} ({v['day_change']})" for t, v in mkt.items()]
            lines.append("[Market: " + ", ".join(parts) + "]")
    if lines:
        lines.insert(0, f"[Verified context generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]")
    return "\n".join(dict.fromkeys(lines))


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
model.eval()


def build_prompt(messages: list[dict]) -> str:
    prompt          = ""
    system_injected = False
    for msg in messages:
        if msg["role"] == "system":
            continue
        if msg["role"] == "user":
            content = msg["content"]
            if not system_injected:
                content = f"{SYSTEM_PROMPT}\n\n{content}"
                system_injected = True
            prompt += f"[INST] {content} [/INST]"
        elif msg["role"] == "assistant":
            prompt += f" {msg['content']}</s>"
    return prompt


def generate(messages: list[dict]) -> str:
    inputs = tokenizer(build_prompt(messages), return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=260,
            temperature=0.72,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    raw = re.sub(r"</s>.*", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\[/?INST\]", "", raw)
    raw = re.sub(r"https?://\S+|www\.\S+", "", raw)
    return raw.strip()


messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
MAX_HISTORY = 14


def chat(user_message: str) -> str:
    global messages
    context   = fetch_context(user_message)
    augmented = f"Verified context:\n{context}\n\nUser message: {user_message}".strip() if context else user_message

    temp = messages + [{"role": "user", "content": augmented}]
    reply = generate(temp) or "lol idk"

    messages.append({"role": "user",      "content": augmented, "display": user_message})
    messages.append({"role": "assistant", "content": reply,     "display": reply})
    if len(messages) > MAX_HISTORY + 1:
        messages[:] = [messages[0]] + messages[-(MAX_HISTORY):]
    return reply

if __name__ == "__main__":
    print("✅ WSB Bot ready. 'quit' to exit, 'reset' to clear history.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "reset":
            messages[:] = [messages[0]]
            print("History cleared.\n")
            continue
        print(f"\nWSB Bot: {chat(user_input)}\n")