import json
import logging
import os
import re
import time
import warnings
from datetime import datetime, timezone

import yfinance as yf

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
if load_dotenv:
    load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

BASE_MODEL = os.getenv("WSB_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
FINETUNED_DIR = os.getenv(
    "WSB_FINETUNED_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "wsb-mistral-finetuned2"),
)
CHAT_PROVIDER = os.getenv("WSB_CHAT_PROVIDER", "auto").strip().lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
LOCAL_MODEL_ENABLED = os.getenv("WSB_ENABLE_LOCAL_MODEL", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

SYSTEM_PROMPT = (
    "You are a WallStreetBets-style chat model for a WSB analytics app. "
    "Talk like a sharp WSB regular: casual, blunt, skeptical, funny, and concise, with market slang when it fits. "
    "Keep the tone edgy without hate speech, harassment, or explicit slurs. "
    "For ticker, market, sentiment, post, and comment claims, use only verified context in square brackets. "
    "Your stance should follow the collected WSB evidence: if the collected posts/comments are bearish, be bearish; if bullish, be bullish; if mixed, say it is mixed. "
    "Verified context comes from market data plus collected WSB posts/comments and sentiment files. "
    "If verified context is missing, say you do not have collected evidence for that exact claim instead of inventing it. "
    "Do not claim breaking news, exact prices, exact returns, or what WSB users said unless it appears in verified context. "
    "Never include URLs or links."
)

MARKET_KEYWORDS = re.compile(
    r"\b(market|dow|nasdaq|s&p|stock market|the market)\b", re.IGNORECASE
)

_DOLLAR_TICKER = re.compile(r"\$([A-Za-z]{1,5})\b")
_CAPS_WORD = re.compile(r"\b[A-Z]{2,5}\b")
TICKER_STOPWORDS = {
    "A", "AI", "ALL", "AND", "ARE", "ATH", "CEO", "CFO", "DD", "EPS", "ETF",
    "GDP", "IPO", "LOL", "LMAO", "NYSE", "OTM", "SEC", "THE", "USA", "USD",
    "WSB", "YOLO",
}
MAX_CANDIDATES = 5
SENTIMENT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_sentiment.json")
RAW_POSTS_FILE = os.getenv(
    "WSB_RAW_POSTS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "wsb_posts.json"),
)
_STOCK_CACHE_TTL_SECONDS = int(os.getenv("STOCK_CACHE_TTL_SECONDS", "300"))
_SENTIMENT_CACHE_TTL_SECONDS = int(os.getenv("SENTIMENT_CACHE_TTL_SECONDS", "120"))
_RAW_POSTS_CACHE_TTL_SECONDS = int(os.getenv("RAW_POSTS_CACHE_TTL_SECONDS", "120"))

_stock_cache: dict[str, tuple[float, dict | None]] = {}
_sentiment_cache: tuple[float, list[dict]] | None = None
_raw_posts_cache: tuple[float, list[dict]] | None = None
_local_model = None
_local_tokenizer = None
_gemini_model = None
_provider_error: str | None = None


def _known_tickers() -> set[str]:
    return {str(row.get("ticker", "")).upper() for row in _load_sentiment_rows() if row.get("ticker")}


def _alias_map(known: set[str] | None = None) -> dict[str, str]:
    try:
        import analyze_wsb
    except Exception:
        return {}
    known_tickers = _known_tickers() if known is None else known
    aliases = getattr(analyze_wsb, "MANUAL_ALIASES", {})
    return {
        str(alias).lower(): str(ticker).upper()
        for alias, ticker in aliases.items()
        if not known_tickers or str(ticker).upper() in known_tickers
    }


def extract_candidates(text: str) -> list[str]:
    found = []
    known_tickers = _known_tickers()
    aliases = _alias_map(known_tickers)
    lowered = text.lower()

    for alias, ticker in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", lowered):
            found.append(ticker)

    for m in _DOLLAR_TICKER.finditer(text):
        ticker = m.group(1).upper()
        if ticker not in TICKER_STOPWORDS:
            found.append(ticker)

    for word in _CAPS_WORD.findall(text):
        ticker = word.upper()
        if ticker in TICKER_STOPWORDS:
            continue
        if known_tickers and ticker not in known_tickers and ticker not in found:
            continue
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


def _clean_context_text(value: object, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text).strip()
    return text[:limit]


def _sample_line(label: str, sample: dict) -> str | None:
    text = _clean_context_text(sample.get("text"), 300)
    if not text:
        return None
    return f"{label}: {text}"


def _load_raw_posts() -> list[dict]:
    global _raw_posts_cache
    if _raw_posts_cache and time.time() - _raw_posts_cache[0] < _RAW_POSTS_CACHE_TTL_SECONDS:
        return _raw_posts_cache[1]
    if not os.path.exists(RAW_POSTS_FILE):
        return []
    try:
        with open(RAW_POSTS_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        posts = raw if isinstance(raw, list) else raw.get("posts", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        posts = []
    _raw_posts_cache = (time.time(), posts)
    return posts


def _mentions_term(text: str, term: str, allow_dollar_prefix: bool = False) -> bool:
    if not text or not term:
        return False
    prefix = r"\$?" if allow_dollar_prefix else ""
    return bool(re.search(rf"(?<![A-Za-z0-9]){prefix}{re.escape(term)}(?![A-Za-z0-9])", text, re.IGNORECASE))


def _ticker_search_terms(tickers: list[str], aliases: dict[str, str] | None = None) -> dict[str, list[tuple[str, bool]]]:
    aliases_by_ticker: dict[str, list[tuple[str, bool]]] = {ticker.upper(): [(ticker.upper(), True)] for ticker in tickers}
    for alias, ticker in (aliases or _alias_map()).items():
        ticker = ticker.upper()
        if ticker in aliases_by_ticker:
            aliases_by_ticker[ticker].append((alias, False))
    return aliases_by_ticker


def _collected_item_line(item: dict) -> str:
    kind = item.get("kind", "item")
    ticker = item.get("ticker", "")
    created = item.get("created_at") or "unknown date"
    score = item.get("score", 0)
    title = _clean_context_text(item.get("title"), 120)
    body = _clean_context_text(item.get("text"), 360)
    title_part = f", title={title!r}" if title else ""
    return f"[Collected WSB {kind} {ticker}: date={created}, score={score}{title_part}, text={body!r}]"


def get_collected_post_context(tickers: list[str], max_items: int = 6) -> list[str]:
    posts = _load_raw_posts()
    if not posts or not tickers:
        return []

    wanted = [t.upper() for t in tickers]
    terms_by_ticker = _ticker_search_terms(wanted, _alias_map(set(wanted)))
    matches: list[dict] = []
    for post in posts:
        title = str(post.get("title") or "")
        text = str(post.get("text") or "")
        post_blob = f"{title} {text}"
        post_score = int(post.get("upvotes") or 0) + int(post.get("num_comments") or 0)
        for ticker in wanted:
            if any(_mentions_term(post_blob, term, allow_dollar) for term, allow_dollar in terms_by_ticker.get(ticker, [])):
                matches.append({
                    "kind": "post",
                    "ticker": ticker,
                    "created_at": post.get("created_at"),
                    "score": post_score,
                    "title": title,
                    "text": text,
                })
        for comment in post.get("comments") or []:
            comment_text = str(comment.get("body") or comment.get("text") or "")
            for ticker in wanted:
                if any(_mentions_term(comment_text, term, allow_dollar) for term, allow_dollar in terms_by_ticker.get(ticker, [])):
                    matches.append({
                        "kind": "comment",
                        "ticker": ticker,
                        "created_at": comment.get("created_at") or post.get("created_at"),
                        "score": int(comment.get("upvotes") or comment.get("score") or 0),
                        "title": title,
                        "text": comment_text,
                    })

    matches.sort(key=lambda item: (str(item.get("created_at") or ""), int(item.get("score") or 0)), reverse=True)
    seen = set()
    lines = []
    for item in matches:
        key = (item.get("ticker"), item.get("kind"), item.get("created_at"), _clean_context_text(item.get("text"), 120))
        if key in seen:
            continue
        seen.add(key)
        lines.append(_collected_item_line(item))
        if len(lines) >= max_items:
            break
    return lines


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
                price = round(float(hist["Close"].iloc[-1]), 2)
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
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
    lines.extend(get_collected_post_context(candidates))
    if MARKET_KEYWORDS.search(user_message):
        mkt = get_market_summary()
        if mkt:
            parts = [f"{t}: ${v['price']} ({v['day_change']})" for t, v in mkt.items()]
            lines.append("[Market: " + ", ".join(parts) + "]")
    if lines:
        lines.insert(0, f"[Verified context generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]")
    return "\n".join(dict.fromkeys(lines))


def _build_prompt(messages_in: list[dict]) -> str:
    prompt = ""
    system_injected = False
    for msg in messages_in:
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


def _load_local_model():
    global _local_model, _local_tokenizer
    if _local_model is not None and _local_tokenizer is not None:
        return _local_model, _local_tokenizer

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not os.path.isdir(FINETUNED_DIR):
        raise RuntimeError(f"Fine-tuned adapter directory not found: {FINETUNED_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
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
    model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
    model.eval()
    _local_model = model
    _local_tokenizer = tokenizer
    return model, tokenizer


def _generate_local(messages_in: list[dict]) -> str:
    import torch

    model, tokenizer = _load_local_model()
    inputs = tokenizer(_build_prompt(messages_in), return_tensors="pt").to(model.device)
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


def _load_gemini_model():
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use the Gemini chat provider.")
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    return _gemini_model


def _gemini_prompt(messages_in: list[dict]) -> str:
    parts = [SYSTEM_PROMPT]
    for msg in messages_in:
        if msg["role"] == "system":
            continue
        speaker = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{speaker}: {msg['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _generate_gemini(messages_in: list[dict]) -> str:
    model = _load_gemini_model()
    response = model.generate_content(
        _gemini_prompt(messages_in),
        generation_config={"temperature": 0.72, "top_p": 0.92, "max_output_tokens": 260},
    )
    text = getattr(response, "text", "") or ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    return text.strip()


def _fallback_reply(user_message: str, context: str) -> str:
    if context:
        ticker_lines = [line for line in context.splitlines() if line.startswith("[") and ":" in line]
        if ticker_lines:
            return (
                "Model provider is not available, but here is the collected WSB context I can verify: "
                + " ".join(ticker_lines[:4])
            )
    return (
        "The chat model is not available right now. Ask about a ticker like $NVDA, or set "
        "GEMINI_API_KEY/GOOGLE_API_KEY so the WSB chat model can answer from collected context."
    )


def provider_status() -> dict:
    api_key_present = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    local_adapter_present = os.path.isdir(FINETUNED_DIR)
    local_enabled = CHAT_PROVIDER == "local" or LOCAL_MODEL_ENABLED
    provider = CHAT_PROVIDER
    if provider == "auto":
        provider = "gemini" if api_key_present else "local" if local_enabled and local_adapter_present else "fallback"
    return {
        "configured_provider": CHAT_PROVIDER,
        "active_provider": provider,
        "gemini_key_present": api_key_present,
        "gemini_model": GEMINI_MODEL,
        "local_model_enabled": local_enabled,
        "local_adapter_present": local_adapter_present,
        "local_adapter_dir": FINETUNED_DIR,
        "base_model": BASE_MODEL,
        "last_provider_error": _provider_error,
    }


def generate(messages_in: list[dict]) -> str:
    global _provider_error
    providers = [CHAT_PROVIDER]
    if CHAT_PROVIDER == "auto":
        providers = []
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            providers.append("gemini")
        if LOCAL_MODEL_ENABLED:
            providers.append("local")
        providers.append("fallback")

    errors = []
    for provider in providers:
        try:
            if provider == "gemini":
                reply = _generate_gemini(messages_in)
            elif provider == "local":
                reply = _generate_local(messages_in)
            elif provider == "fallback":
                last_user = next((m["content"] for m in reversed(messages_in) if m["role"] == "user"), "")
                context = ""
                if "Verified context:" in last_user:
                    context = last_user.split("User message:", 1)[0].replace("Verified context:", "").strip()
                reply = _fallback_reply(last_user, context)
            else:
                raise RuntimeError(f"Unknown WSB_CHAT_PROVIDER: {provider}")
            _provider_error = None
            return reply or "I do not have enough grounded context to answer that."
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
            _provider_error = "; ".join(errors)
            if CHAT_PROVIDER != "auto":
                raise
    raise RuntimeError(_provider_error or "No chat provider available")


messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
MAX_HISTORY = 14


def chat(user_message: str) -> str:
    global messages
    context = fetch_context(user_message)
    augmented = f"Verified context:\n{context}\n\nUser message: {user_message}".strip() if context else user_message

    temp = messages + [{"role": "user", "content": augmented}]
    try:
        reply = generate(temp)
    except Exception:
        reply = _fallback_reply(user_message, context)

    messages.append({"role": "user", "content": augmented, "display": user_message})
    messages.append({"role": "assistant", "content": reply, "display": reply})
    if len(messages) > MAX_HISTORY + 1:
        messages[:] = [messages[0]] + messages[-MAX_HISTORY:]
    return reply


if __name__ == "__main__":
    print("WSB Bot ready. 'quit' to exit, 'reset' to clear history.")
    print(provider_status())
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
