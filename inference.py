import os
import torch
import re
import warnings

import yfinance as yf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

warnings.filterwarnings("ignore")
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

BASE_MODEL    = "mistralai/Mistral-7B-Instruct-v0.3"
FINETUNED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wsb-mistral-finetuned")

SYSTEM_PROMPT = (
    "You are a WallStreetBets user. "
    "Be direct and crude like a real WSB reply. "
    "Match your response length to what the question needs. "
    "When stock data is provided in the message, use those exact numbers. "
    "Never invent prices, percentages, or statistics. "
    "Never include URLs or links."
)

MARKET_KEYWORDS = re.compile(
    r"\b(market|dow|nasdaq|s&p|stock market|the market)\b", re.IGNORECASE
)

_DOLLAR_TICKER = re.compile(r"\$([A-Za-z]{1,5})\b")
_CAPS_WORD     = re.compile(r"\b[A-Z]{2,5}\b")
MAX_CANDIDATES = 5


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


_VALID_QUOTE_TYPES = {"EQUITY", "ETF", "INDEX", "FUTURE"}


def get_stock_data(ticker: str) -> dict | None:
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        if info.get("quoteType", "") not in _VALID_QUOTE_TYPES:
            return None
        hist = t.history(period="1mo")
        if hist.empty:
            return None
        current_price = round(float(hist["Close"].iloc[-1]), 2)
        change_pct    = round(
            ((current_price - float(hist["Close"].iloc[0])) / float(hist["Close"].iloc[0])) * 100, 2
        )
        pe = info.get("trailingPE")
        return {
            "ticker":      ticker.upper(),
            "price":       current_price,
            "1mo_change":  f"{change_pct:+.2f}%",
            "pe_ratio":    round(pe, 2) if pe else "N/A",
            "52w_high":    info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low":     info.get("fiftyTwoWeekLow", "N/A"),
            "short_ratio": info.get("shortRatio", "N/A"),
        }
    except Exception:
        return None


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
    for ticker in extract_candidates(user_message):
        data = get_stock_data(ticker)
        if data:
            lines.append(
                f"[{data['ticker']}: price=${data['price']}, "
                f"1mo={data['1mo_change']}, P/E={data['pe_ratio']}, "
                f"52w_high=${data['52w_high']}, 52w_low=${data['52w_low']}, "
                f"short_ratio={data['short_ratio']}]"
            )
    if MARKET_KEYWORDS.search(user_message):
        mkt = get_market_summary()
        if mkt:
            parts = [f"{t}: ${v['price']} ({v['day_change']})" for t, v in mkt.items()]
            lines.append("[Market: " + ", ".join(parts) + "]")
    return "\n".join(lines)


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
            max_new_tokens=400,
            temperature=0.85,
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
    augmented = f"{context}\n{user_message}".strip() if context else user_message

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