"""
WSB Ticker + Sentiment Analyzer
--------------------------------
Input:  wsb_posts.json  (produced by getdata.py)
Output: ticker_sentiment.json

Detection paths (highest → lowest confidence):
  1. $TICKER   — conf 0.97, blacklist bypassed
  2. Alias     — conf 0.84–0.98, blacklist bypassed  ("tesla" → TSLA)
  3. ALL CAPS  — conf 0.60–0.98, blacklist applied, scored by position/context
  4. Lowercase — conf 0.72–0.88, blacklist applied, requires financial context

Sentiment: FinBERT per ticker, each ticker scored on its own ±10/5 word window.
All sentiment inference is batched globally for GPU efficiency.
"""

import argparse
import os
import re
import math
import json
import time
import hashlib
import urllib.request
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Iterator

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE     = os.getenv("WSB_POSTS_FILE", "wsb_posts.json")
OUTPUT_FILE    = os.getenv("TICKER_SENTIMENT_FILE", "ticker_sentiment.json")
DAILY_SENTIMENT_FILE = os.getenv("DAILY_SENTIMENT_FILE", "backend/agg_sentiment.json")
AGGREGATE_WINDOW_DAYS = float(os.getenv("SENTIMENT_AGGREGATE_WINDOW_DAYS", "14"))
FINBERT_MODEL  = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
BATCH_SIZE     = int(os.getenv("FINBERT_BATCH_SIZE", "4"))
MIN_MENTIONS   = int(os.getenv("MIN_TICKER_MENTIONS", "3"))
MIN_CONFIDENCE = float(os.getenv("MIN_TICKER_CONFIDENCE", "0.65"))
MAX_ENGAGEMENT = int(os.getenv("MAX_SENTIMENT_ENGAGEMENT", "1000"))               # cap for log2 engagement multiplier

# Shorter ticker = more ambiguous = stricter threshold.
# Note: CONF_DEFAULT serves as both the per-ticker fallback AND the
# post-aggregation minimum (MIN_CONFIDENCE). Keep them in sync.
CONF_THRESHOLD: dict[int, float] = {2: 0.92, 3: 0.85, 4: 0.73, 5: 0.68}
CONF_DEFAULT   = MIN_CONFIDENCE

# Per-ticker sentiment context window (words)
CONTEXT_BEFORE = 10
CONTEXT_AFTER  = 5

MAX_SAMPLES    = 50   # max raw samples stored per ticker

# Recency decay — power curve over the collection window.
# Posts from today → weight 1.0. Posts from DECAY_WINDOW_DAYS ago → DECAY_FLOOR.
#
# Shape is controlled by DECAY_MIDPOINT: the weight at the halfway mark (day 15).
# This lets you keep the same floor while flattening or steepening the curve.
#
# Formula:  weight = 1 - (1 - DECAY_FLOOR) * (age / window) ^ p
# where p is solved so the curve hits DECAY_MIDPOINT at day 15.
#
# Curve comparison (floor=0.1, window=30):
#
#   Midpoint  |  day 7  |  day 14  |  day 21  |  day 30  | feel
#   ----------+---------+----------+----------+----------+------------------
#     0.38    |  0.617  |  0.380   |  0.234   |  0.100   | original (steep)
#     0.55    |  0.735  |  0.550   |  0.355   |  0.100   | moderate
#     0.70    |  0.836  |  0.700   |  0.490   |  0.100   | gentle
#     0.82    |  0.909  |  0.820   |  0.610   |  0.100   | very gentle
#
DECAY_WINDOW_DAYS = float(os.getenv("SENTIMENT_DECAY_WINDOW_DAYS", "14"))
DECAY_FLOOR       = float(os.getenv("SENTIMENT_DECAY_FLOOR", "0.10"))
DECAY_MIDPOINT    = float(os.getenv("SENTIMENT_DECAY_MIDPOINT", "0.55"))

# =============================================================================
# BLACKLIST
# Applied to ALL CAPS and lowercase paths only.
# $prefix and alias detections always bypass this list.
#
# IMPORTANT: legitimate tickers that collide with common words (e.g. IT, ARE,
# OPEN, LOVE) are intentionally excluded here so they can still be detected
# via $TICKER or alias paths. Do NOT add tickers to this list.
# =============================================================================

BLACKLIST: frozenset[str] = frozenset({
    # Single letters (all 26)
    *'ABCDEFGHIJKLMNOPQRSTUVWXYZ',

    # Two-letter non-ticker words
    'AI','OK','NO','SO','DO','GO','BY','MY','US','AN','AS','AT','BE','IF',
    'IN','IS','IT','OF','ON','OR','TO','UP','WE','HE','AM','EX','HI','HO',
    'ID','IO','OX','OY','RE','UM','UN','UT',

    # Three-letter common words
    'ACT','ADD','AGE','AGO','AID','AIM','AIR','ALL','AND','ANY','ARE','ARK',
    'ARM','ART','ASH','ASK','ATE','AWE','AXE','BAD','BAG','BAN','BAR','BAT',
    'BAY','BED','BEG','BET','BIG','BIT','BOW','BOX','BOY','BUD','BUG','BUN',
    'BUS','BUT','BUY','CAN','CAP','CAR','CAT','COP','COT','COW','CRY','CUP',
    'CUT','DAY','DID','DIG','DIM','DIP','DOT','DRY','DUE','EAR','EAT','END',
    'ERA','EYE','FAN','FAR','FAT','FEW','FIT','FIX','FLY','FOR','FUN','GAP',
    'GAS','GET','GOT','GUM','GUN','GUY','HAD','HAS','HAT','HIM','HIS','HIT',
    'HOT','HOW','HUG','ICE','ILL','INK','ION','ITS','JAM','JAR','JET','JOB',
    'JOG','JOY','KEY','KID','LAB','LAD','LAG','LAP','LAW','LAY','LEG','LET',
    'LID','LIE','LIT','LOG','LOT','LOW','MAD','MAP','MAT','MAY','MEN','MIX',
    'MOB','MOM','MUD','NAP','NET','NEW','NOR','NOT','NOW','NUT','OAK','OAR',
    'OAT','ODD','OLD','ONE','OPT','ORE','OUR','OUT','OWL','OWN','PAD','PAL',
    'PAN','PAR','PAT','PAW','PAY','PEG','PEN','PET','PIE','PIG','PIN','PIT',
    'POD','POP','POT','PRY','PUB','PUN','PUP','PUT','RAG','RAM','RAP','RAT',
    'RAW','RAY','RED','RIB','RIG','RIM','RIP','ROD','ROT','ROW','RUB','RUG',
    'RUN','RUT','SAD','SAP','SAT','SAW','SAY','SEA','SET','SHE','SIN','SIP',
    'SIR','SIT','SIX','SKI','SKY','SOB','SOD','SON','SOW','SOY','SPA','SPY',
    'SUB','SUM','SUN','TAB','TAN','TAP','TAR','TAX','TEN','THE','TIN','TIP',
    'TON','TOO','TOW','TOY','TUB','TUG','TWO','UGH','URN','VAT','VIA','VOW',
    'WAD','WAR','WAS','WAY','WED','WET','WHO','WIG','WIN','WIT','WOE','WON',
    'WOW','YAK','YAP','YEA','YES','YET','ZAP','ZEN','ZIP',

    # Four-letter common words
    'ABLE','ALSO','BACK','BEEN','BEST','BOTH','CALL','CAME','COME','COST',
    'DEAD','DOES','DONE','DOWN','EACH','ELSE','EVEN','EVER','FELT','FEEL',
    'FIND','FIVE','FOUR','FROM','GAIN','GAVE','GIVE','GOES','GONE','GOOD',
    'HARD','HAVE','HELP','HERE','HIGH','HOLD','HOME','HOPE','HUGE','IDEA',
    'INTO','JUST','KEEP','KIND','KNOW','LAST','LATE','LIKE','LONG','LOOK',
    'LOSS','LOVE','MADE','MAKE','MANY','MIND','MISS','MORE','MOST','MOVE',
    'MUCH','NEED','NEXT','NONE','ONCE','ONLY','OPEN','OVER','PAST','PLAN',
    'PLAY','PUTS','RATE','REAL','RISK','SAID','SAME','SELL','SENT','SHOW',
    'SOME','SOON','STAY','STOP','SURE','TAKE','TALK','THAN','THAT','THEM',
    'THEN','THEY','THIS','TILL','TIME','TOLD','TOOK','TURN','VERY','WAIT',
    'WALK','WANT','WAYS','WEEK','WELL','WENT','WERE','WHAT','WHEN','WHOM',
    'WILL','WITH','WORD','WORK','YEAR','YOUR','ZERO',

    # Finance / Reddit / WSB jargon (not tickers)
    'DD','CEO','CFO','CTO','COO','IPO','ETF','ATH','ATL','AMA','DM','OP',
    'OC','IMO','TBH','NGL','SMH','IRL','IDK','IDC','LOL','LMAO','WTF','OMG',
    'FYI','TIL','TLDR','NSFW','HODL','YOLO','FOMO','FWIW','AFAIK','WSB',
    'PE','EPS','ROE','ROI','DCF','YOY','QOQ','TTM','IRR','NPV','EBIT',
    'CAGR','NAV','AUM','HFT','OTM','ATM','ITM','IV','GDP','CPI','PPI',
    'FED','SEC','FDA','IRS','IMF','ECB','VIX',
    'BEAR','BULL','CASH','DEBT','DUMP','FUND','LONG','MOON','PUMP',
    'SELL','CALL','HOLD','SOLD','BOND',
    'YOU','SEE','DON','USE','ARE','FOR','HIM','HER','DID','GET','HAS',
    'LET','SAY','RUN','SET','OUT','ALL','ONE','TWO','WHO','WHY','HOW',
    'TECH','POST','AWAY','LIFE','NEAR','FREE','GROW','TURN','MUCH',
    'HOOD','AREN',
    # Common words that are NOT real tickers (verified against NASDAQ/NYSE)
    # Do NOT add real tickers here — use REQUIRE_PREFIX instead.
    'USA','NATO','UAE','GOP','CIA','NYC','BBC','DEI',
    'MILK','VICE','MUST','DIPS','PAYS','WASH','DRUG','FORM',
    'ROAD','WISE','RULE','MATH','IRON','COAL','FARM','BIRD',
    'BOOT','BELT','DAWN','HERO','SWIM','TREE','KONG','TOWN',
    'TAXI','CORP','TASK','RAIN','COLD','INFO','CORN','TEND',
    'COOK','GIFT','EARN','FOLD','HIDE','DIVE','RING','SOUL',
    'POLE','SWAN','ALTO','BOUT','JACK','BALL','FLAG','WAVE',
    'SPIN','FURY','DARE','RACE','SEAT','BUCK','BAND','SOAR',
    'AMID','REKT','SEPT','WOOF','POOL','SITE','DRIP','DYOR',
    'ASIA','MATE','ALOT','GAINZ','YALL','AINT','MAGA','PUSH',
    'MARS','PINK','MINT','COM','PRE','MAN','TOP','GRAB',
    'MOAT','NAIL','NAVY','NEWS','NODE','NOON','NORM','NOSE',
    'NUDE','OATH','ODDS','ORAL','PACE','PACT','PAGE','PAIN',
    'PALE','PALM','PART','PASS','PEAK','PEAR','PEEL','PEER',
    'PILE','PINE','PLAN','PLOT','POEM','POLL','POND','PORT',
    'POSE','PREY','PULL','PURE','RACK','RAGE','RAID','RAIL',
    'RANK','RANT','REED','REEF','REEL','RICE','RICH','RIDE',
    'RIOT','RISE','ROAR','ROBE','ROCK','ROLE','ROLL','ROOF',
    'ROOM','ROOT','ROPE','ROSE','RUBY','RUIN','RUSH','RUST',
    'SAGE','SAKE','SALE','SALT','SAND','SCAN','SCAR','SEAM',
    'SHED','SHIN','SHOT','SICK','SIDE','SIGH','SIGN','SILK',
    'SING','SINK','SKIP','SLIM','SLOT','SLOW','SOAP','SOCK',
    'SOFT','SOIL','SOLE','SONG','SORT','SOUP','SPAN','SPUR',
    'STAR','STEW','STUB','STUN','SUIT','SURF','SWAP','SWAT',
    'TALE','TALL','TAME','TAPE','TAUT','TEAK','TEAL','TEAR',
    'TELL','TENT','TIDE','TILE','TILT','TIRE','TOLL','TONE',
    'TOSS','TOUR','TRAM','TREK','TRIM','TRIO','TROD','TROY',
    'TUCK','TUFT','TURF','TWIN','TYPE','UNDO','UPON','URGE',
    'USED','USER','VARY','VAST','VEIL','VEIN','VENT','VEST',
    'VIEW','VINE','VOID','WADE','WAGE','WAKE','WANE','WARD',
    'WARM','WARN','WARP','WEAR','WEED','WEEP','WELD','WEST',
    'WHIM','WHIP','WIDE','WILT','WINE','WINK','WISH','WISP',
    'WOKE','WOLF','WOMB','WOOL','WORE','WORN','WRAP','WREN',
    'WRIT','YAWN','YOGA','YORE','ZEAL','ZEST','ZINC','ZOOM',
})

# Lowercase version — built once at module load
_BLACKLIST_LOWER: frozenset[str] = frozenset(w.lower() for w in BLACKLIST)

# Real tickers that collide so heavily with common words that we only accept
# them when the post explicitly uses $TICKER or a known company name alias.
# Bare caps/word-match alone is not sufficient signal for these.
#
# Rule of thumb: if you'd use the word in a sentence without thinking about
# stocks, it belongs here. Add new entries when you see false positives.
REQUIRE_PREFIX: frozenset[str] = frozenset({
    # Currencies / commodities / indices that people write as plain words
    'USD','EUR','GBP','JPY','CNY','OIL','GAS',
    # Short words that are real tickers but extremely common in prose
    'NOW',   # ServiceNow         — "we need this now"
    'NET',   # Cloudflare         — "net income", "net loss"
    'CAR',   # Avis Budget        — "car payment"
    'FIT',   # Fitbit             — "fit for purpose"
    'KEY',   # KeyCorp            — "key factor"
    'PAY',   # Paymentus          — "I'll pay tomorrow"
    'PIN',   # Pinduoduo          — "pin the blame"
    'RIO',   # Rio Tinto ADR      — city name
    'WAR',   # Westrock Coffee    — "trade war"
    'WEN',   # Wendy's            — "when does it open"
    'MAN',   # MAN Group          — "the man behind"
    'MAP',   # WisdomTree ETF     — "road map"
    'MAR',   # Marriott           — "it will mar"
    # Common 4-letter words that are real but highly ambiguous tickers
    'REAL',  # Realty Income      — adjective
    'GOLF',  # Acushnet           — sport
    'BOOT',  # BOOT Barn (wait — BOOT is blacklisted, this is redundant but harmless)
    'COST',  # Costco             — "at what cost" (alias handles it anyway)
    'FLOW',  # Globalink          — "cash flow"
    'SAFE',  # Bancorp            — adjective
    'EASY',  # eOn Computing      — adjective
    'DEEP',  # Deep Medicine      — adjective
    'MUST',  # MustGrow           — modal verb
    'LIVE',  # Live Nation/LIQT   — adjective/verb
    'CARE',  # CARE Acquisition   — verb/noun
    'GAME',  # GameTech           — noun
    'LINK',  # IDT/Intervoice     — noun/verb
    'PICK',  # ARCA biopharma     — verb
    'FACT',  # Factset (FACT etf) — noun
    'HOUR',  # Hour Holdings      — noun
    'LINE',  # LINE Corp          — noun
    'ONTO',  # Onto Innovation    — also a common preposition
    'FAST',  # Fastenal           — adjective (alias "fastenal" handles it)
    'EDIT',  # Editas Medicine    — verb
    'WOOD',  # iShares Clean ETF  — noun
    'FLEX',  # Flex Ltd           — verb/noun
    'STEM',  # Stem Inc           — noun
    'LUCK',  # Lucky Strike       — noun
    'MEME',  # Roundhill ETF      — noun
    'CAKE',  # Cheesecake Factory — noun
    'FROG',  # JFrog              — noun
    'NERD',  # Roundhill ETF      — noun
    'BEAT',  # Cardium Theraputics — verb (very common false positive)
    'NICE',  # NICE Systems       — adjective (very common false positive)
    'GOLD',  # Barrick Gold       — noun (very common false positive)
    'TACO',  # Tortoise Capital   — noun (very common false positive)
})

# =============================================================================
# ALIASES — company name (lowercase) → ticker, bypasses blacklist
# =============================================================================

MANUAL_ALIASES: dict[str, str] = {
    'tesla':'TSLA', 'apple':'AAPL', 'microsoft':'MSFT',
    'google':'GOOGL', 'alphabet':'GOOGL', 'amazon':'AMZN',
    'nvidia':'NVDA', 'meta':'META', 'facebook':'META',
    'costco':'COST', 'best buy':'BBY', 'bestbuy':'BBY',
    'walmart':'WMT', 'target':'TGT', 'home depot':'HD', 'homedepot':'HD',
    'gamestop':'GME', 'game stop':'GME', 'amc':'AMC', 'blackberry':'BB',
    'palantir':'PLTR', 'coinbase':'COIN', 'robinhood':'HOOD', 'sofi':'SOFI',
    'paypal':'PYPL', 'jpmorgan':'JPM', 'jp morgan':'JPM',
    'goldman':'GS', 'goldman sachs':'GS', 'morgan stanley':'MS',
    'blackrock':'BLK', 'berkshire':'BRK', 'berkshire hathaway':'BRK',
    'rivian':'RIVN', 'lucid':'LCID', 'ford':'F', 'gm':'GM', 'general motors':'GM',
    'disney':'DIS', 'netflix':'NFLX', 'spotify':'SPOT',
    'uber':'UBER', 'lyft':'LYFT', 'airbnb':'ABNB',
    'amd':'AMD', 'intel':'INTC', 'qualcomm':'QCOM', 'broadcom':'AVGO', 'tsmc':'TSM',
    'snowflake':'SNOW', 'shopify':'SHOP', 'salesforce':'CRM',
    'servicenow':'NOW', 'cloudflare':'NET', 'datadog':'DDOG',
    'crowdstrike':'CRWD', 'okta':'OKTA', 'twilio':'TWLO',
    'mongodb':'MDB', 'gitlab':'GTLB',
    'alibaba':'BABA', 'baidu':'BIDU', 'nio':'NIO',
    'pfizer':'PFE', 'moderna':'MRNA', 'johnson':'JNJ',
    'unitedhealth':'UNH', 'cvs':'CVS', 'abbvie':'ABBV',
    'boeing':'BA', 'exxon':'XOM', 'chevron':'CVX',
    'caterpillar':'CAT', 'deere':'DE', 'john deere':'DE',
    'mastercard':'MA', 'visa':'V',
}

FINANCIAL_CONTEXT: frozenset[str] = frozenset({
    'calls','puts','shares','stock','buy','sell','short','long','bullish',
    'bearish','position','options','strike','expiry','earnings','revenue',
    'price','target','yolo','gains','loss','profit','tendies','squeeze',
    'breakout','oversold','overbought','volume','float','dividend','split',
    'merger','acquisition','support','resistance','hodl','hold','moon',
    'pump','dump','rally','ath','rip',
})

# =============================================================================
# TICKER UNIVERSE
# =============================================================================

def _clean_security_name(name: str) -> str:
    value = (name or "").strip()
    if " - " in value:
        value = value.split(" - ", 1)[0].strip()
    suffixes = (
        " Common Stock",
        " Ordinary Shares",
        " American Depositary Shares",
        " American Depositary Receipt",
    )
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[: -len(suffix)].strip()
    return value


def _asset_type_from_security_name(name: str, etf_flag: str = "") -> str:
    lowered = (name or "").lower()
    if etf_flag.upper() == "Y":
        return "ETF"
    if any(term in lowered for term in ("etf", "etn", "exchange traded", "index fund")):
        return "ETF"
    if any(term in lowered for term in ("fund", "trust", "notes due", "preferred", "warrant", "unit")):
        return "Fund/Other"
    return "Company"


def fetch_ticker_metadata() -> dict[str, dict[str, str]]:
    """Fetch valid ticker symbols plus display metadata from NASDAQ trader files."""
    metadata: dict[str, dict[str, str]] = {}
    errors: list[str] = []

    sources = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    for url in sources:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                lines = r.read().decode('utf-8').splitlines()
        except Exception as e:
            errors.append(f"{url}: {e}")
            continue
        if not lines:
            continue
        headers = [h.strip() for h in lines[0].split('|')]
        for line in lines[1:]:
            if line.startswith('File Creation Time'):
                continue
            parts = line.split('|')
            row = dict(zip(headers, parts))
            sym = (row.get('Symbol') or row.get('ACT Symbol') or row.get('NASDAQ Symbol') or '').strip().upper()
            if not re.match(r'^[A-Z]{1,5}$', sym):
                continue
            name = (row.get('Security Name') or '').strip()
            etf_flag = (row.get('ETF') or '').strip()
            metadata[sym] = {
                'company': _clean_security_name(name),
                'asset_type': _asset_type_from_security_name(name, etf_flag),
            }

    if not metadata:
        raise RuntimeError(
            "Could not fetch any tickers from NASDAQ.\n" + "\n".join(errors)
        )
    if errors:
        print(f"  Warning: some ticker sources failed:\n" + "\n".join(f"    {e}" for e in errors))
    return metadata


def fetch_ticker_list() -> set[str]:
    return set(fetch_ticker_metadata())


def build_alias_map(valid_tickers: set[str]) -> dict[str, str]:
    return {k: v for k, v in MANUAL_ALIASES.items() if v in valid_tickers}

# =============================================================================
# TICKER EXTRACTION
# =============================================================================

@dataclass
class TickerMention:
    ticker:     str
    confidence: float
    method:     str


def _ctx_bonus(text_lower: str) -> float:
    """Small confidence boost when the surrounding text contains finance words."""
    return min(sum(1 for w in FINANCIAL_CONTEXT if w in text_lower) * 0.04, 0.16)


def _caps_confidence(token: str, all_words: list[str], idx: int, ctx: float) -> float:
    """
    Score the likelihood that an ALL-CAPS token is a ticker based on:
      - Whether it is surrounded by lowercase prose words
      - Financial-context bonus
      - Token length (longer = less ambiguous)

    Floors: len 2→0.60, 3→0.67, 4→0.74, 5→0.81
    Ceiling: 0.98
    """
    before       = all_words[max(0, idx - 3) : idx]
    after        = all_words[idx + 1 : idx + 4]
    lower_before = [w for w in before if w == w.lower() and len(w) > 1]
    lower_after  = [w for w in after  if w == w.lower() and len(w) > 1]

    score = 0.0
    if lower_before:                  score += 0.40
    if lower_after:                   score += 0.25
    if lower_before and lower_after:  score += 0.20
    score += ctx * 0.8
    score += min(len(token) - 2, 3) * 0.08

    floor = 0.60 + min(len(token) - 2, 3) * 0.07
    return min(max(score, floor), 0.98)


def extract_tickers(
    text: str,
    valid_tickers: set[str],
    alias_map: dict[str, str],
) -> list[TickerMention]:
    """
    Detect ticker mentions in text via four paths (see module docstring).
    Returns one TickerMention per ticker — the detection with the highest
    confidence is kept when multiple paths fire for the same symbol.
    Multiple detections of the same ticker are combined: the method kept is
    the highest-confidence one, but the confidence itself is the weighted
    average so that repeated signal isn't silently discarded.
    """
    if not text or len(text.strip()) < 5:
        return []

    tl  = text.lower()
    ctx = _ctx_bonus(tl)

    raw: list[TickerMention] = []

    # ── 1. $TICKER ────────────────────────────────────────────────────────────
    for m in re.finditer(r'\$([A-Za-z]{1,5})\b', text):
        sym = m.group(1).upper()
        if sym in valid_tickers:
            raw.append(TickerMention(sym, 0.97, '$prefix'))

    # ── 2. ALL CAPS in mixed-case prose ───────────────────────────────────────
    all_words = re.findall(r'\b[A-Za-z]+\b', text)
    for i, word in enumerate(all_words):
        if not re.match(r'^[A-Z]{2,5}$', word):
            continue
        if word not in valid_tickers or word.lower() in _BLACKLIST_LOWER:
            continue
        conf = _caps_confidence(word, all_words, i, ctx)
        raw.append(TickerMention(word, conf, 'caps_in_context'))

    # ── 3. Lowercase word match (requires financial context for short tokens) ─
    words_lower = re.findall(r'\b[a-z]{2,20}\b', tl)
    CTX_WIN = 8

    def has_financial_context(i: int) -> bool:
        window = words_lower[max(0, i - CTX_WIN) : i + CTX_WIN + 1]
        return any(w in FINANCIAL_CONTEXT for w in window)

    for i, token in enumerate(words_lower):
        sym = token.upper()
        if sym not in valid_tickers or token in _BLACKLIST_LOWER:
            continue
        if len(token) >= 5 or has_financial_context(i):
            conf = 0.72 + min(len(token) - 2, 3) * 0.04 + ctx
            raw.append(TickerMention(sym, conf, 'word_match'))

    # ── 4. Alias / company name ───────────────────────────────────────────────
    bigrams = [
        f"{words_lower[i]} {words_lower[i + 1]}"
        for i in range(len(words_lower) - 1)
    ]
    for token in bigrams + words_lower:
        if token in alias_map:
            conf = min((0.90 if ' ' in token else 0.84) + ctx, 0.98)
            raw.append(TickerMention(alias_map[token], conf, 'alias'))

    # ── Deduplicate: weighted-average confidence across detections ────────────
    grouped: dict[str, list[TickerMention]] = defaultdict(list)
    for m in raw:
        grouped[m.ticker].append(m)

    results: list[TickerMention] = []
    for ticker, hits in grouped.items():
        # High-collision tickers must have at least one strong signal ($prefix
        # or alias) to be accepted — bare caps/word matches alone are not enough.
        if ticker in REQUIRE_PREFIX:
            if not any(h.method in ('$prefix', 'alias') for h in hits):
                continue

        best      = max(hits, key=lambda h: h.confidence)
        avg_conf  = sum(h.confidence for h in hits) / len(hits)
        blended   = 0.7 * best.confidence + 0.3 * avg_conf
        threshold = CONF_THRESHOLD.get(len(ticker), CONF_DEFAULT)
        if blended >= threshold:
            results.append(TickerMention(ticker, round(blended, 4), best.method))

    return results

# =============================================================================
# SENTIMENT  (global batched inference)
# =============================================================================

_finbert_pipe = None


def _load_finbert(force_cpu: bool = False):
    global _finbert_pipe
    if _finbert_pipe is not None and not force_cpu:
        return _finbert_pipe
    try:
        from transformers import pipeline
        import torch
        device = -1 if force_cpu or not torch.cuda.is_available() else 0
        label  = 'CPU' if device == -1 else 'GPU'
        if force_cpu:
            print("  Falling back to CPU for FinBERT...")
        print(f"  Loading FinBERT on {label}...")
        _finbert_pipe = pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL,
            truncation=True,
            max_length=512,
            batch_size=BATCH_SIZE,
            device=device,
        )
        print("  FinBERT ready")
        return _finbert_pipe
    except ImportError:
        raise SystemExit("Install: pip install transformers torch")


def _ticker_context(ticker: str, text: str) -> str:
    """
    Build a FinBERT input string that isolates sentiment directed at a specific
    ticker, handling two failure modes:

    1. "NVDA and ORACLE TO THE MOON" — both tickers share the same sentence.
       A fixed word window would give both the same context and the same score.
       Fix: score each ticker on its own sentence only, not a window that bleeds
       across sentence boundaries.

    2. "NVDA sucks, buy ORACLE" — sentiment words belong to different tickers.
       Fix: within the ticker's sentence(s), duplicate the words immediately
       adjacent to the ticker so they carry more weight when FinBERT reads the
       sequence. This is a soft proximity bias — FinBERT still sees the full
       sentence for coherence, but the closest words appear twice.

    Strategy:
      - Split the full text into sentences.
      - Collect all sentences that contain the ticker (by any surface form).
      - If no sentence matches, fall back to the old fixed word window.
      - Within each matched sentence, repeat the 3 words immediately before and
        after the ticker so they're weighted more heavily.
      - Join matched sentences and truncate to 512 chars for FinBERT.
    """
    # ── 1. Split into sentences ───────────────────────────────────────────────
    # Split on common sentence-ending punctuation. Keep the delimiter attached
    # so each fragment is still a readable sentence.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) == 1:
        # No sentence boundaries found — try comma/semicolon as soft boundaries
        sentences = re.split(r'[,;]\s+', text.strip())

    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()

    # ── 2. Find sentences containing this ticker ──────────────────────────────
    matched: list[str] = []
    for sent in sentences:
        sent_upper = sent.upper()
        # Match $TICKER, TICKER as a word, or lowercase ticker
        if (re.search(rf'\${re.escape(ticker_upper)}\b', sent, re.IGNORECASE)
                or re.search(rf'\b{re.escape(ticker_upper)}\b', sent_upper)
                or re.search(rf'\b{re.escape(ticker_lower)}\b', sent.lower())):
            matched.append(sent)

    # ── 3. Fall back to fixed word window if no sentence matched ──────────────
    if not matched:
        words = text.split()
        idx = next(
            (i for i, w in enumerate(words)
             if re.sub(r'[^A-Za-z]', '', w).upper() == ticker_upper),
            None,
        )
        if idx is None:
            return text[:512]
        left  = words[max(0, idx - CONTEXT_BEFORE) : idx]
        right = words[idx + 1 : idx + CONTEXT_AFTER + 1]
        return ' '.join(left + [words[idx]] + right)[:512]

    # ── 4. Apply proximity weighting within each matched sentence ─────────────
    weighted: list[str] = []
    for sent in matched:
        words = sent.split()
        idx = next(
            (i for i, w in enumerate(words)
             if re.sub(r'[^A-Za-z]', '', w).upper() == ticker_upper),
            None,
        )
        if idx is None:
            weighted.append(sent)
            continue
        # Repeat the 3 closest words on each side so FinBERT weights them more
        prox_before = words[max(0, idx - 3) : idx]
        prox_after  = words[idx + 1 : idx + 4]
        # Format: [proximity words] + full sentence, so the key words appear
        # early (FinBERT is position-sensitive) and again in full context
        weighted.append(' '.join(prox_before + prox_after + words))

    return ' '.join(weighted)[:512]


def _load_sentiment_cache(cache_file: str | None) -> dict[str, tuple[str, float]]:
    if not cache_file or not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    cache: dict[str, tuple[str, float]] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and value.get("label") and value.get("score") is not None:
            cache[str(key)] = (str(value["label"]), float(value["score"]))
    return cache


def _write_sentiment_cache(cache_file: str | None, cache: dict[str, tuple[str, float]]) -> None:
    if not cache_file:
        return
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    payload = {key: {"label": label, "score": score} for key, (label, score) in sorted(cache.items())}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _sentiment_cache_key(context: str) -> str:
    payload = f"{FINBERT_MODEL}\0{context}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def run_sentiment_batch(
    ticker_text_pairs: list[tuple[str, str]],
    cache_file: str | None = None,
) -> list[tuple[str, float]]:
    """
    Score every (ticker, text) pair in one global GPU pass.

    Strategy:
      1. Build a context string per pair.
      2. Deduplicate identical contexts so the GPU sees each unique string once.
      3. Process in chunks with cache clearing between each to avoid OOM.
      4. Fall back to CPU on the first OOM and finish there.

    Returns a list of (label, score) tuples in the same order as the input.
    """
    import torch

    # Build context strings — pass full text so _ticker_context can do
    # sentence splitting and proximity weighting itself
    contexts: list[str] = []
    for ticker, text in ticker_text_pairs:
        contexts.append(_ticker_context(ticker, text))

    # Deduplicate while preserving insertion order
    unique_contexts: list[str] = list(dict.fromkeys(contexts))
    cache = _load_sentiment_cache(cache_file)
    ctx_to_result: dict[str, tuple[str, float]] = {}
    missing_contexts: list[str] = []
    for ctx in unique_contexts:
        cached = cache.get(_sentiment_cache_key(ctx))
        if cached:
            ctx_to_result[ctx] = cached
        else:
            missing_contexts.append(ctx)

    print(f"  Sentiment cache: {len(unique_contexts) - len(missing_contexts):,} hit(s), {len(missing_contexts):,} new context(s)")

    def _run_chunk(pipe, chunk: list[str], keys: list[str]) -> None:
        out = pipe(chunk)
        for ctx, r in zip(keys, out):
            result = (r['label'].lower(), round(r['score'], 3))
            ctx_to_result[ctx] = result
            cache[_sentiment_cache_key(ctx)] = result

    chunk_size   = BATCH_SIZE * 2
    force_cpu    = False
    i            = 0

    while i < len(missing_contexts):
        chunk      = missing_contexts[i : i + chunk_size]
        chunk_keys = missing_contexts[i : i + chunk_size]   # same slice, clarity
        try:
            pipe = _load_finbert(force_cpu=force_cpu)
            _run_chunk(pipe, chunk, chunk_keys)
            i += chunk_size
        except (RuntimeError, Exception) as e:
            if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                print(f"\n  [OOM] GPU exhausted at chunk {i}. Switching to CPU...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                force_cpu = True   # all remaining chunks run on CPU
                # do NOT advance i — retry this chunk on CPU
            else:
                raise
        finally:
            if torch.cuda.is_available() and not force_cpu:
                torch.cuda.empty_cache()

    if missing_contexts:
        _write_sentiment_cache(cache_file, cache)

    return [ctx_to_result[c] for c in contexts]

# =============================================================================
# RECENCY WEIGHTING
# =============================================================================

def recency_weight(post_ts: float | None, now_ts: float) -> float:
    """
    Power-curve decay anchored at both ends and shaped by a midpoint weight.

        weight = 1 - (1 - DECAY_FLOOR) * t^p

    where t = age / window  (0 → today, 1 → oldest edge)
    and   p is solved so the curve hits DECAY_MIDPOINT at t=0.5 (day 15).

    Posts with no timestamp fall back to mid-window weight.
    """
    if post_ts is None:
        age_days = DECAY_WINDOW_DAYS / 2.0
    else:
        age_days = max(now_ts - post_ts, 0.0) / 86_400.0

    age_days = min(age_days, DECAY_WINDOW_DAYS)
    k        = math.log(DECAY_FLOOR) / DECAY_WINDOW_DAYS
    return math.exp(k * age_days)


# =============================================================================
# SEMANTIC VALUE
# =============================================================================

_DIRECTION: dict[str, float] = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}


def estimate_engagement(upvotes: int, awards: int, upvote_ratio: float | None = None) -> float:
    """
    Use attention, not only net score. For posts with an upvote ratio, estimate
    total votes from net score and ratio. For comments, Reddit only exposes net
    score, so abs(score) is the best available attention proxy.
    """
    net_score = abs(upvotes)
    total_votes = net_score
    if upvote_ratio is not None and 0 < upvote_ratio < 1 and abs(2 * upvote_ratio - 1) > 0.05:
        total_votes = max(net_score, net_score / abs(2 * upvote_ratio - 1))
    return min(total_votes + awards * 3 + 2, MAX_ENGAGEMENT)


def semantic_value(
    label: str,
    score: float,
    confidence: float,
    upvotes: int,
    awards: int,
    recency: float,
    upvote_ratio: float | None = None,
) -> float:
    """
    direction × sentiment_strength × detection_confidence
              × log2(attention_cap) × recency_weight

    Attention uses estimated total votes when possible, so controversial posts
    can still carry weight instead of disappearing behind a low net score.
    """
    engagement = estimate_engagement(upvotes, awards, upvote_ratio)
    return (
        _DIRECTION.get(label, 0.0)
        * score
        * confidence
        * math.log2(engagement)
        * recency
    )

# =============================================================================
# PER-TICKER ACCUMULATOR
# =============================================================================

@dataclass
class TickerBucket:
    mentions:       int   = 0
    semantic_sum:   float = 0.0
    positive_sum:   float = 0.0
    negative_sum:   float = 0.0
    positive_count: int   = 0
    negative_count: int   = 0
    neutral_count:  int   = 0
    confidence_sum: float = 0.0
    methods:        dict  = field(default_factory=lambda: defaultdict(int))
    samples:        deque = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))

    def update(self, mention: TickerMention, label: str, score: float, sv: float, sample: dict) -> None:
        self.mentions       += 1
        self.confidence_sum += mention.confidence
        self.semantic_sum   += sv
        self.methods[mention.method] += 1
        self.samples.append(sample)

        if label == 'positive':
            self.positive_count += 1
            self.positive_sum   += sv
        elif label == 'negative':
            self.negative_count += 1
            self.negative_sum   += sv
        else:
            self.neutral_count += 1

# =============================================================================
# INPUT HELPERS
# =============================================================================

def _parse_int(value) -> int:
    """
    Coerce a Reddit score field to int.
    Handles None, int, float, and strings like '1.2k' or '3,400'.
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower().replace(',', '')
    if s.endswith('k'):
        try:
            return int(float(s[:-1]) * 1_000)
        except ValueError:
            return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


# Timestamp formats seen in the wild from Reddit scrapers
_TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S",   # "2026-03-14 20:12:19"  ← your data
    "%Y-%m-%dT%H:%M:%S",   # "2026-03-14T20:12:19"
    "%Y-%m-%dT%H:%M:%SZ",  # "2026-03-14T20:12:19Z"
    "%Y-%m-%d",            # "2026-03-14"
)

def _parse_timestamp(value) -> float | None:
    """
    Coerce a Reddit timestamp field to a Unix float.
    Handles None, numeric Unix timestamps, and common datetime strings.
    Returns None if the value cannot be parsed so the caller can fall back
    gracefully rather than crashing.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None  # unrecognised format — caller will use mid-window fallback

# =============================================================================
# MAIN
# =============================================================================

def run(
    input_file: str = INPUT_FILE,
    output_file: str = OUTPUT_FILE,
    daily_sentiment_file: str | None = DAILY_SENTIMENT_FILE,
    aggregate_window_days: float | None = AGGREGATE_WINDOW_DAYS,
    finbert_model: str | None = None,
    batch_size: int | None = None,
    min_mentions: int | None = None,
    min_confidence: float | None = None,
    sentiment_cache_file: str | None = None,
) -> None:
    global FINBERT_MODEL, BATCH_SIZE, MIN_MENTIONS, MIN_CONFIDENCE
    if finbert_model is not None:
        FINBERT_MODEL = finbert_model
    if batch_size is not None:
        BATCH_SIZE = batch_size
    if min_mentions is not None:
        MIN_MENTIONS = min_mentions
    if min_confidence is not None:
        MIN_CONFIDENCE = min_confidence
    print("=" * 62)
    print("  WSB Ticker + Sentiment Analyzer")
    print("=" * 62)

    with open(input_file, encoding='utf-8') as f:
        posts = json.load(f)
    print(f"\n  {len(posts):,} posts loaded")

    print("\n  Fetching NASDAQ ticker list...")
    ticker_metadata = fetch_ticker_metadata()
    valid_tickers = set(ticker_metadata)
    print(f"  {len(valid_tickers):,} tickers")

    alias_map = build_alias_map(valid_tickers)
    print(f"  {len(alias_map):,} aliases\n")


    # ── Pass 1: extract all ticker mentions across posts + comments ────────────
    # Each item: (text, upvotes, awards, timestamp_or_None, [TickerMention, ...], meta_dict)
    all_items: list[tuple[str, int, int, float | None, list[TickerMention], dict]] = []
    stats: dict[str, int] = defaultdict(int)

    now_ts = time.time()
    aggregate_cutoff_ts = (
        now_ts - aggregate_window_days * 86_400
        if aggregate_window_days is not None else None
    )

    for post in tqdm(posts, desc="Extracting tickers", unit="post"):
        post_text = f"{post.get('title') or ''} {post.get('text') or ''}".strip()
        p_upvotes = _parse_int(post.get('upvotes'))
        p_awards  = _parse_int(post.get('awards'))
        p_ts      = _parse_timestamp(
            post.get('created_at') or post.get('created_utc') or post.get('timestamp')
        )

        post_mentions = extract_tickers(post_text, valid_tickers, alias_map)
        if post_mentions:
            all_items.append((post_text, p_upvotes, p_awards, p_ts, post_mentions, {
                'source': 'post',
                'post_id': post.get('id'),
                'permalink': post.get('permalink'),
                'upvotes': p_upvotes,
                'upvote_ratio': post.get('upvote_ratio'),
                'awards': p_awards,
                'created_utc': p_ts,
            }))
            stats['proc_posts'] += 1
        else:
            stats['disc_posts'] += 1

        for comment in post.get('comments') or []:
            body    = (comment.get('body') or '').strip()
            c_score = _parse_int(comment.get('score'))
            c_award = _parse_int(comment.get('awards'))
            c_ts    = _parse_timestamp(
                comment.get('created_at') or comment.get('created_utc') or comment.get('timestamp')
            ) or p_ts  # fall back to parent post time

            if not body or body in ('[deleted]', '[removed]'):
                continue

            c_mentions = extract_tickers(body, valid_tickers, alias_map)
            if c_mentions:
                all_items.append((body, c_score, c_award, c_ts, c_mentions, {
                    'source': 'comment',
                    'post_id': post.get('id'),
                    'comment_id': comment.get('comment_id'),
                    'upvotes': c_score,
                    'awards': c_award,
                    'created_utc': c_ts,
                }))
                stats['proc_comments'] += 1
            else:
                stats['disc_comments'] += 1

    print(f"\n  {len(all_items):,} items with ticker mentions")

    # ── Pass 2: single global sentiment inference ──────────────────────────────
    print("\n  Running FinBERT sentiment (global batch)...")
    ticker_text_pairs = [
        (m.ticker, text)
        for (text, _, _, _, mentions, _) in all_items
        for m in mentions
    ]
    sentiments: Iterator[tuple[str, float]] = iter(
        run_sentiment_batch(ticker_text_pairs, cache_file=sentiment_cache_file)
    )

    # ── Pass 3: accumulate results ─────────────────────────────────────────────
    data: dict[str, TickerBucket] = defaultdict(TickerBucket)
    daily_sentiment: dict[tuple[str, str], float] = defaultdict(float)

    for (text, upvotes, awards, ts, mentions, meta) in all_items:
        recency = recency_weight(ts, now_ts)
        item_day = (
            datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            if ts is not None else
            datetime.fromtimestamp(now_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        )
        age_days = ((now_ts - ts) / 86_400.0) if ts is not None else 0.0
        in_aggregate_window = aggregate_window_days is None or age_days <= aggregate_window_days
        if in_aggregate_window:
            stats['aggregate_items_with_mentions'] += 1
            if meta.get('source') == 'post':
                stats['aggregate_posts'] += 1
            elif meta.get('source') == 'comment':
                stats['aggregate_comments'] += 1

        for m in mentions:
            label, score = next(sentiments)
            ratio = meta.get('upvote_ratio')
            try:
                ratio = float(ratio) if ratio is not None else None
            except (TypeError, ValueError):
                ratio = None
            sv = semantic_value(label, score, m.confidence, upvotes, awards, recency, ratio)
            sample = {
                **meta,
                'text':            text[:220],
                'ticker':          m.ticker,
                'sentiment':       label,
                'sentiment_score': round(score, 3),
                'semantic_value':  round(sv, 4),
                'recency_weight':  round(recency, 4),
                'confidence':      round(m.confidence, 3),
                'method':          m.method,
            }
            daily_sentiment[(m.ticker, item_day)] += sv
            if in_aggregate_window:
                data[m.ticker].update(m, label, score, sv, sample)

    # ── Build output ───────────────────────────────────────────────────────────
    tickers_out = []
    for ticker, d in data.items():
        n = d.mentions
        pos_sum = d.positive_sum
        neg_sum = d.negative_sum

        overall = (
            'bullish' if pos_sum      > abs(neg_sum) else
            'bearish' if abs(neg_sum) > pos_sum      else
            'neutral'
        )

        # Top 5 samples split by direction so consumers get clear signal
        sorted_samples = sorted(d.samples, key=lambda x: abs(x['semantic_value']), reverse=True)
        top_bullish = [s for s in sorted_samples if s['sentiment'] == 'positive'][:5]
        top_bearish = [s for s in sorted_samples if s['sentiment'] == 'negative'][:5]

        meta_row = ticker_metadata.get(ticker, {})
        tickers_out.append({
            'ticker':                   ticker,
            'company':                  meta_row.get('company', ''),
            'asset_type':               meta_row.get('asset_type', 'Unknown'),
            'mentions':                 n,
            'semantic_score':           round(d.semantic_sum,              4),
            'normalized_semantic_score':round(d.semantic_sum / n,          4),
            'positive_score':           round(pos_sum,                     4),
            'negative_score':           round(neg_sum,                     4),
            'positive_count':           d.positive_count,
            'negative_count':           d.negative_count,
            'neutral_count':            d.neutral_count,
            'avg_confidence':           round(d.confidence_sum / n,        3),
            'overall_sentiment':        overall,
            'sentiment_ratio':          round(d.positive_count / n,        3),
            'detection_methods':        dict(d.methods),
            'top_bullish_posts':        top_bullish,
            'top_bearish_posts':        top_bearish,
        })

    before = len(tickers_out)
    tickers_out = [
        t for t in tickers_out
        if t['mentions'] >= MIN_MENTIONS and t['avg_confidence'] >= MIN_CONFIDENCE
    ]
    tickers_out.sort(key=lambda x: x['mentions'], reverse=True)

    output = {
        'meta': {
            'total_posts':        len(posts),
            'processed_posts':    stats['proc_posts'],
            'discarded_posts':    stats['disc_posts'],
            'processed_comments': stats['proc_comments'],
            'discarded_comments': stats['disc_comments'],
            'unique_tickers':     len(tickers_out),
            'generated_at':       datetime.fromtimestamp(now_ts, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'aggregate_window_days': aggregate_window_days,
            'aggregate_cutoff': (
                datetime.fromtimestamp(aggregate_cutoff_ts, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                if aggregate_cutoff_ts is not None else None
            ),
            'aggregate_posts':    stats['aggregate_posts'],
            'aggregate_comments': stats['aggregate_comments'],
            'aggregate_items_with_mentions': stats['aggregate_items_with_mentions'],
            'noise_removed':      before - len(tickers_out),
            'sentiment_model':    FINBERT_MODEL,
            'recency_decay': {
                'window_days': DECAY_WINDOW_DAYS,
                'floor':       DECAY_FLOOR,
            },
        },
        'tickers': tickers_out,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if daily_sentiment_file:
        valid_output_tickers = {t['ticker'] for t in tickers_out}
        daily_rows = [
            {
                'day': day,
                'ticker': ticker,
                'refined_sentiment': round(value, 6),
            }
            for (ticker, day), value in daily_sentiment.items()
            if ticker in valid_output_tickers and value != 0
        ]
        daily_rows.sort(key=lambda row: (row['day'], row['ticker']))
        daily_dir = os.path.dirname(daily_sentiment_file)
        if daily_dir:
            os.makedirs(daily_dir, exist_ok=True)
        with open(daily_sentiment_file, 'w', encoding='utf-8') as f:
            json.dump(daily_rows, f, indent=2, ensure_ascii=False)

    # ── Summary table ──────────────────────────────────────────────────────────
    m = output['meta']
    print(f"\n{'='*62}")
    print(f"  Done!  {m['unique_tickers']} tickers  ({m['noise_removed']} filtered)")
    print(f"  Posts:    {m['processed_posts']:,} analyzed | {m['discarded_posts']:,} discarded")
    print(f"  Comments: {m['processed_comments']:,} analyzed | {m['discarded_comments']:,} discarded")
    print(f"\n  {'#':<4} {'Ticker':<8} {'Mentions':>8} {'Bull%':>6} {'NormScore':>10} {'Sentiment':<12} {'AvgConf':>8}  Methods")
    print(f"  {'─'*82}")
    for i, t in enumerate(tickers_out, 1):
        methods = ', '.join(f"{k}:{v}" for k, v in t['detection_methods'].items())
        print(
            f"  {i:<4} {t['ticker']:<8} {t['mentions']:>8,}"
            f" {t['sentiment_ratio']*100:>5.0f}%"
            f" {t['normalized_semantic_score']:>10.4f}"
            f"  {t['overall_sentiment']:<12} {t['avg_confidence']:>8.3f}  {methods}"
        )
    print(f"\n  Saved → {output_file}")
    if daily_sentiment_file:
        print(f"  Daily sentiment → {daily_sentiment_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze WSB ticker sentiment with FinBERT.")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--daily-output", default=DAILY_SENTIMENT_FILE)
    parser.add_argument("--aggregate-days", type=float, default=AGGREGATE_WINDOW_DAYS)
    parser.add_argument("--finbert-model", default=FINBERT_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--min-mentions", type=int, default=MIN_MENTIONS)
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--sentiment-cache", default=os.getenv("FINBERT_SENTIMENT_CACHE", "finbert_sentiment_cache.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_file=args.input,
        output_file=args.output,
        daily_sentiment_file=args.daily_output,
        aggregate_window_days=args.aggregate_days,
        finbert_model=args.finbert_model,
        batch_size=args.batch_size,
        min_mentions=args.min_mentions,
        min_confidence=args.min_confidence,
    )