import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
import yfinance as yf

DATE_FMT = "%Y-%m-%d"
TRADING_TIMEZONE = ZoneInfo(os.getenv("SIM_TRADING_TIMEZONE", "America/New_York"))
MARKET_OPEN_TIME = dt_time(
    int(os.getenv("SIM_MARKET_OPEN_HOUR", "9")),
    int(os.getenv("SIM_MARKET_OPEN_MINUTE", "30")),
)


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, DATE_FMT)


def load_sentiment(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected {path} to contain a JSON list")
    clean = []
    for row in rows:
        try:
            ticker = str(row["ticker"]).upper().strip()
            day = parse_date(str(row["day"]))
            sentiment = float(row["refined_sentiment"])
        except (KeyError, TypeError, ValueError):
            continue
        if ticker and math.isfinite(sentiment) and sentiment != 0:
            clean.append({"ticker": ticker, "day": day, "sentiment": sentiment})
    return sorted(clean, key=lambda row: row["day"])


def rolling_signals(rows: list[dict], trade_day: datetime, window_days: int) -> dict[str, float]:
    # Strictly exclude the trade day. Daily sentiment rows are bucketed by
    # the America/New_York 9:30am ET cutoff: pre-open items are assigned to
    # the previous signal day, and items at/after open stay on that day so
    # they can only affect the next trade generation.
    window_end = trade_day
    window_start = trade_day - timedelta(days=window_days)
    signals: dict[str, float] = defaultdict(float)
    for row in rows:
        if window_start <= row["day"] < window_end:
            signals[row["ticker"]] += row["sentiment"]
    return {ticker: value for ticker, value in signals.items() if value != 0}


def select_signals(signals: dict[str, float], max_positions: int) -> dict[str, float]:
    ranked = sorted(signals.items(), key=lambda item: abs(item[1]), reverse=True)
    if max_positions > 0:
        ranked = ranked[:max_positions]
    return dict(ranked)


def extract_open_series(data: pd.DataFrame, ticker: str | None = None) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        if ticker and (ticker, "Open") in data.columns:
            return data[(ticker, "Open")].dropna()
        if ticker and ("Open", ticker) in data.columns:
            return data[("Open", ticker)].dropna()
        for column in data.columns:
            if isinstance(column, tuple) and "Open" in column:
                return data[column].dropna()
        return pd.Series(dtype=float)
    return data.get("Open", pd.Series(dtype=float)).dropna()


def get_market_open_days(start: datetime, end: datetime, calendar_ticker: str = "SPY") -> list[datetime]:
    start_s = start.strftime(DATE_FMT)
    end_s = (end + timedelta(days=1)).strftime(DATE_FMT)
    data = yf.download(
        calendar_ticker,
        start=start_s,
        end=end_s,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if data.empty:
        return []
    open_series = extract_open_series(data, calendar_ticker)
    return [datetime.strptime(index.strftime(DATE_FMT), DATE_FMT) for index in open_series.index]


def get_open_prices(tickers: list[str], start: datetime, end: datetime) -> dict[str, dict[str, float]]:
    if not tickers:
        return {}
    start_s = start.strftime(DATE_FMT)
    end_s = (end + timedelta(days=1)).strftime(DATE_FMT)
    data = yf.download(
        tickers,
        start=start_s,
        end=end_s,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    prices: dict[str, dict[str, float]] = {ticker: {} for ticker in tickers}
    if data.empty:
        return prices

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            series = extract_open_series(data, ticker)
            for idx, value in series.items():
                prices[ticker][idx.strftime(DATE_FMT)] = float(value)
    else:
        series = extract_open_series(data, tickers[0])
        ticker = tickers[0]
        for idx, value in series.items():
            prices[ticker][idx.strftime(DATE_FMT)] = float(value)
    return prices


def extract_close_series(data: pd.DataFrame, ticker: str | None = None) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        if ticker and (ticker, "Close") in data.columns:
            return data[(ticker, "Close")].dropna()
        if ticker and ("Close", ticker) in data.columns:
            return data[("Close", ticker)].dropna()
        for column in data.columns:
            if isinstance(column, tuple) and "Close" in column:
                return data[column].dropna()
        return pd.Series(dtype=float)
    return data.get("Close", pd.Series(dtype=float)).dropna()


def get_latest_prices(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    data = yf.download(
        tickers,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    prices: dict[str, float] = {}
    if data.empty:
        return prices
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            series = extract_close_series(data, ticker)
            if not series.empty:
                prices[ticker] = float(series.iloc[-1])
    else:
        series = extract_close_series(data, tickers[0])
        if not series.empty:
            prices[tickers[0]] = float(series.iloc[-1])
    return prices


def current_market_time() -> datetime:
    return datetime.now(timezone.utc).astimezone(TRADING_TIMEZONE)


def trade_close_time(entry_day: datetime) -> datetime:
    candidate = entry_day.date() + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return datetime.combine(candidate, MARKET_OPEN_TIME, tzinfo=TRADING_TIMEZONE)


def is_trade_still_open(entry_day: datetime, now: datetime | None = None) -> bool:
    local_now = now or current_market_time()
    return local_now < trade_close_time(entry_day)


def mark_to_market(entries: list[dict]) -> tuple[float, str | None]:
    tickers = [entry["ticker"] for entry in entries if entry.get("entry_price") and entry.get("shares")]
    latest_prices = get_latest_prices(tickers)
    total_unrealized = 0.0
    marked_at = current_market_time().strftime("%Y-%m-%d %H:%M:%S %Z")
    for entry in entries:
        current_price = latest_prices.get(entry.get("ticker"))
        if not current_price or not entry.get("entry_price") or not entry.get("shares"):
            continue
        pnl = position_pnl(entry, current_price)
        current_value = entry.get("notional", 0) + pnl
        entry["current_price"] = current_price
        entry["current_mark_price"] = current_price
        entry["current_marked_at"] = marked_at
        entry["current_unrealized_pnl"] = pnl
        entry["current_trade_result"] = pnl
        entry["current_value"] = current_value
        total_unrealized += pnl
    return total_unrealized, marked_at if latest_prices else None


def build_trade_days(rows: list[dict], window_days: int) -> list[datetime]:
    raw_days = sorted({row["day"] for row in rows})
    if not raw_days:
        return []
    first_candidate = raw_days[0] + timedelta(days=window_days)
    last_candidate = min(raw_days[-1], datetime.now())
    if first_candidate > last_candidate:
        return []
    return get_market_open_days(first_candidate, last_candidate)


def build_planned_entries(signals: dict[str, float], day_s: str, account_value: float) -> list[dict]:
    total_abs_signal = sum(abs(value) for value in signals.values())
    if total_abs_signal <= 0 or account_value <= 0:
        return []
    entries = []
    for ticker, signal in signals.items():
        allocation = account_value * (abs(signal) / total_abs_signal)
        entries.append({
            "ticker": ticker,
            "side": "long" if signal > 0 else "short",
            "entry_date": day_s,
            "entry_price": None,
            "shares": None,
            "notional": allocation,
            "sentiment": signal,
            "weight": allocation / account_value,
        })
    return entries


def next_weekday_after(day: datetime, latest_sentiment_day: datetime) -> datetime | None:
    candidate = day + timedelta(days=1)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    latest = latest_sentiment_day.replace(hour=0, minute=0, second=0, microsecond=0)
    limit = min(today, latest + timedelta(days=1))
    while candidate <= limit:
        if candidate.weekday() < 5:
            return candidate
        candidate += timedelta(days=1)
    return None


def latest_existing_simulation_date(output_dir: Path) -> datetime | None:
    path = output_dir / "portfolio_total_investment.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    rows = data.get("daily_data") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    dates = []
    for row in rows:
        try:
            dates.append(parse_date(str(row["date"])))
        except (KeyError, TypeError, ValueError):
            continue
    return max(dates) if dates else None


def load_existing_simulation(output_dir: Path) -> dict | None:
    path = output_dir / "portfolio_total_investment.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def existing_needs_intraday_refresh(existing: dict) -> bool:
    rows = existing.get("daily_data") if isinstance(existing, dict) else None
    if not isinstance(rows, list) or not rows:
        return False
    last = rows[-1]
    try:
        record_day = parse_date(str(last["date"])).date()
    except (KeyError, TypeError, ValueError):
        return False
    if record_day > datetime.now().date():
        return False
    entries = last.get("entries") or []
    has_unpriced_entries = any(entry.get("entry_price") is None for entry in entries if isinstance(entry, dict))
    return bool(last.get("pre_open_plan") or has_unpriced_entries)


def build_entries(
    signals: dict[str, float],
    prices: dict[str, dict[str, float]],
    day_s: str,
    account_value: float,
    next_day_s: str | None = None,
) -> list[dict]:
    usable = {
        ticker: signal
        for ticker, signal in signals.items()
        if prices.get(ticker, {}).get(day_s)
        and (next_day_s is None or prices.get(ticker, {}).get(next_day_s))
    }
    total_abs_signal = sum(abs(value) for value in usable.values())
    if total_abs_signal <= 0 or account_value <= 0:
        return []

    entries = []
    for ticker, signal in usable.items():
        entry_price = prices[ticker][day_s]
        allocation = account_value * (abs(signal) / total_abs_signal)
        side = "long" if signal > 0 else "short"
        entries.append({
            "ticker": ticker,
            "side": side,
            "entry_date": day_s,
            "entry_price": entry_price,
            "shares": allocation / entry_price,
            "notional": allocation,
            "sentiment": signal,
            "weight": allocation / account_value,
        })
    return entries


def position_pnl(position: dict, exit_price: float) -> float:
    if position["side"] == "long":
        return (exit_price - position["entry_price"]) * position["shares"]
    return (position["entry_price"] - exit_price) * position["shares"]


def simulate(
    sentiment_file: Path,
    output_dir: Path,
    initial_capital: float,
    window_days: int,
    max_positions: int,
    continue_existing: bool = False,
) -> dict:
    rows = load_sentiment(sentiment_file)
    if not rows:
        raise SystemExit(f"No usable sentiment rows in {sentiment_file}")

    trade_days = build_trade_days(rows, window_days)
    if len(trade_days) < 2:
        raise SystemExit("Need at least two market-open days to simulate open-to-open returns")

    planned_day = next_weekday_after(trade_days[-1], rows[-1]["day"])
    newest_needed_day = planned_day or trade_days[-1]
    if continue_existing:
        latest_existing = latest_existing_simulation_date(output_dir)
        existing = load_existing_simulation(output_dir)
        if latest_existing and existing and latest_existing >= newest_needed_day and not existing_needs_intraday_refresh(existing):
            print(f"Simulation already current through {latest_existing.strftime(DATE_FMT)}.")
            return existing

    all_signals_by_day = {
        day.strftime(DATE_FMT): select_signals(rolling_signals(rows, day, window_days), max_positions)
        for day in trade_days
    }
    all_tickers = sorted({ticker for signals in all_signals_by_day.values() for ticker in signals})
    prices = get_open_prices(all_tickers, trade_days[0], trade_days[-1])

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("portfolio_*.json"):
        if stale_file.name != "portfolio_total_investment.json":
            stale_file.unlink()

    account_value = initial_capital
    initial_investment_date = trade_days[0].strftime(DATE_FMT)
    portfolio_statistics = []
    daily_data = []
    previous_positions: list[dict] = []
    total_profit = 0.0

    for index, day in enumerate(trade_days[:-1]):
        day_s = day.strftime(DATE_FMT)
        next_day = trade_days[index + 1]
        next_day_s = next_day.strftime(DATE_FMT)

        realized_pnl = 0.0
        exits = []
        for position in previous_positions:
            exit_price = prices.get(position["ticker"], {}).get(day_s)
            if not exit_price:
                continue
            pnl = position_pnl(position, exit_price)
            realized_pnl += pnl
            exits.append({
                **position,
                "exit_date": day_s,
                "exit_price": exit_price,
                "pnl": pnl,
            })

        account_value += realized_pnl
        total_profit = account_value - initial_capital

        signals = all_signals_by_day.get(day_s, {})
        entries = build_entries(signals, prices, day_s, account_value, next_day_s)
        new_positions = entries

        daily_record = {
            "date": day_s,
            "next_trade_date": next_day_s,
            "starting_value": account_value - realized_pnl,
            "realized_pnl": realized_pnl,
            "ending_value_before_rebalance": account_value,
            "allocated_value": sum(entry["notional"] for entry in entries),
            "cash_after_rebalance": account_value - sum(entry["notional"] for entry in entries),
            "exits": exits,
            "entries": entries,
            "positions": {
                "long": {p["ticker"]: p for p in entries if p["side"] == "long"},
                "short": {p["ticker"]: p for p in entries if p["side"] == "short"},
            },
            "today_profit": realized_pnl,
            "total_profit": total_profit,
            "total_investment": account_value,
        }
        daily_data.append(daily_record)
        portfolio_statistics.append({
            "date": day_s,
            "investment": account_value,
            "today_profit": realized_pnl,
            "total_profit": total_profit,
        })

        with (output_dir / f"portfolio_{day_s}.json").open("w", encoding="utf-8") as f:
            json.dump(daily_record, f, indent=2)

        previous_positions = new_positions

    final_day = trade_days[-1]
    final_day_s = final_day.strftime(DATE_FMT)
    final_pnl = 0.0
    final_exits = []
    for position in previous_positions:
        exit_price = prices.get(position["ticker"], {}).get(final_day_s)
        if not exit_price:
            continue
        pnl = position_pnl(position, exit_price)
        final_pnl += pnl
        final_exits.append({**position, "exit_date": final_day_s, "exit_price": exit_price, "pnl": pnl})
    account_value += final_pnl
    total_profit = account_value - initial_capital

    planned_entries = build_entries(all_signals_by_day.get(final_day_s, {}), prices, final_day_s, account_value)
    trade_is_open = is_trade_still_open(final_day)
    current_marked_at = None
    if trade_is_open:
        current_unrealized_pnl, current_marked_at = mark_to_market(planned_entries)
    else:
        current_unrealized_pnl = 0.0
    marked_account_value = account_value + current_unrealized_pnl
    final_record = {
        "date": final_day_s,
        "starting_value": account_value - final_pnl,
        "realized_pnl": final_pnl,
        "ending_value_before_rebalance": account_value,
        "allocated_value": sum(entry["notional"] for entry in planned_entries),
        "cash_after_rebalance": account_value - sum(entry["notional"] for entry in planned_entries),
        "exits": final_exits,
        "entries": planned_entries,
        "positions": {
            "long": {p["ticker"]: p for p in planned_entries if p["side"] == "long"},
            "short": {p["ticker"]: p for p in planned_entries if p["side"] == "short"},
        },
        "planned_only": True,
        "trade_status": "open_mark_to_market" if trade_is_open else "closed",
        "trade_close_time": trade_close_time(final_day).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "current_marked_at": current_marked_at,
        "current_unrealized_pnl": current_unrealized_pnl,
        "current_trade_result": current_unrealized_pnl,
        "marked_account_value": marked_account_value,
        "today_profit": final_pnl + current_unrealized_pnl,
        "total_profit": total_profit + current_unrealized_pnl,
        "total_investment": marked_account_value,
    }
    daily_data.append(final_record)
    portfolio_statistics.append({
        "date": final_day_s,
        "investment": marked_account_value,
        "today_profit": final_pnl + current_unrealized_pnl,
        "total_profit": total_profit + current_unrealized_pnl,
    })
    with (output_dir / f"portfolio_{final_day_s}.json").open("w", encoding="utf-8") as f:
        json.dump(final_record, f, indent=2)

    if planned_day and planned_day.strftime(DATE_FMT) != final_day_s:
        planned_day_s = planned_day.strftime(DATE_FMT)
        planned_signals = select_signals(rolling_signals(rows, planned_day, window_days), max_positions)
        future_entries = build_planned_entries(planned_signals, planned_day_s, account_value)
        future_record = {
            "date": planned_day_s,
            "starting_value": account_value,
            "realized_pnl": 0.0,
            "ending_value_before_rebalance": account_value,
            "allocated_value": sum(entry["notional"] for entry in future_entries),
            "cash_after_rebalance": account_value - sum(entry["notional"] for entry in future_entries),
            "exits": [],
            "entries": future_entries,
            "positions": {
                "long": {p["ticker"]: p for p in future_entries if p["side"] == "long"},
                "short": {p["ticker"]: p for p in future_entries if p["side"] == "short"},
            },
            "planned_only": True,
            "pre_open_plan": True,
            "today_profit": 0.0,
            "total_profit": total_profit,
            "total_investment": account_value,
        }
        daily_data.append(future_record)
        portfolio_statistics.append({
            "date": planned_day_s,
            "investment": account_value,
            "today_profit": 0.0,
            "total_profit": total_profit,
        })
        with (output_dir / f"portfolio_{planned_day_s}.json").open("w", encoding="utf-8") as f:
            json.dump(future_record, f, indent=2)

    result = {
        "meta": {
            "strategy": "daily_open_to_open_sentiment_rebalance",
            "initial_capital": initial_capital,
            "final_value": account_value,
            "total_return_pct": (account_value / initial_capital - 1) * 100,
            "rolling_sentiment_window_days": window_days,
            "max_positions": max_positions,
            "sentiment_file": str(sentiment_file),
            "warmup_days": window_days,
            "trading_calendar": "SPY market-open dates",
            "signal_timing": "daily rows use America/New_York 9:30am ET cutoff buckets; trade day excludes same-day rows so post-open data is next-generation only",
            "mark_to_market": "latest open trade is refreshed with current/latest prices until the next 9:30am ET open",
        },
        "daily_data": daily_data,
        "portfolio_statistics": portfolio_statistics,
        "initial_investment_date": initial_investment_date,
        "total_investment": account_value,
    }
    with (output_dir / "portfolio_total_investment.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily WSB sentiment open-to-open rebalance simulation.")
    parser.add_argument("--sentiment-file", default=os.getenv("SIM_SENTIMENT_FILE", "backend/agg_sentiment.json"))
    parser.add_argument("--output-dir", default=os.getenv("SIM_OUTPUT_DIR", "frontend/src/portfolio_data"))
    parser.add_argument("--initial-capital", type=float, default=float(os.getenv("SIM_INITIAL_CAPITAL", "1000000")))
    parser.add_argument("--window-days", type=int, default=int(os.getenv("SIM_WINDOW_DAYS", "14")))
    parser.add_argument("--max-positions", type=int, default=int(os.getenv("SIM_MAX_POSITIONS", "25")))
    parser.add_argument("--continue-existing", action="store_true", help="Skip regeneration when the saved simulation is already current.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = simulate(
        sentiment_file=Path(args.sentiment_file),
        output_dir=Path(args.output_dir),
        initial_capital=args.initial_capital,
        window_days=args.window_days,
        max_positions=args.max_positions,
        continue_existing=args.continue_existing,
    )
    meta = result["meta"]
    print(f"Final value: ${meta['final_value']:,.2f}")
    print(f"Total return: {meta['total_return_pct']:.2f}%")
    print(f"Saved simulation to {args.output_dir}")
