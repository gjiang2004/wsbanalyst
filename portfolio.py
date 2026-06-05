import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

DATE_FMT = "%Y-%m-%d"


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, DATE_FMT)


def next_weekday(day: datetime) -> datetime:
    current = day + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def trading_day_on_or_after(day: datetime) -> datetime:
    current = day
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


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
            if ticker not in data.columns.get_level_values(0):
                continue
            series = data[(ticker, "Open")].dropna()
            for idx, value in series.items():
                prices[ticker][idx.strftime(DATE_FMT)] = float(value)
    else:
        series = data.get("Open", pd.Series(dtype=float)).dropna()
        ticker = tickers[0]
        for idx, value in series.items():
            prices[ticker][idx.strftime(DATE_FMT)] = float(value)
    return prices


def build_trade_days(rows: list[dict], window_days: int) -> list[datetime]:
    raw_days = sorted({row["day"] for row in rows})
    if not raw_days:
        return []
    first_trade_day = trading_day_on_or_after(raw_days[0] + timedelta(days=window_days))
    trade_days = sorted({trading_day_on_or_after(day) for day in raw_days})
    return [day for day in trade_days if day >= first_trade_day and day.date() < datetime.now().date()]


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
) -> dict:
    rows = load_sentiment(sentiment_file)
    if not rows:
        raise SystemExit(f"No usable sentiment rows in {sentiment_file}")

    trade_days = build_trade_days(rows, window_days)
    if len(trade_days) < 2:
        raise SystemExit("Need at least two trading days to simulate open-to-open returns")

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
        usable = {
            ticker: signal
            for ticker, signal in signals.items()
            if prices.get(ticker, {}).get(day_s) and prices.get(ticker, {}).get(next_day_s)
        }
        total_abs_signal = sum(abs(value) for value in usable.values())

        entries = []
        new_positions = []
        if total_abs_signal > 0 and account_value > 0:
            for ticker, signal in usable.items():
                entry_price = prices[ticker][day_s]
                allocation = account_value * (abs(signal) / total_abs_signal)
                shares = allocation / entry_price
                side = "long" if signal > 0 else "short"
                position = {
                    "ticker": ticker,
                    "side": side,
                    "entry_date": day_s,
                    "entry_price": entry_price,
                    "shares": shares,
                    "notional": allocation,
                    "sentiment": signal,
                    "weight": allocation / account_value,
                }
                entries.append(position)
                new_positions.append(position)

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

    final_record = {
        "date": final_day_s,
        "starting_value": account_value - final_pnl,
        "realized_pnl": final_pnl,
        "ending_value_before_rebalance": account_value,
        "allocated_value": 0.0,
        "cash_after_rebalance": account_value,
        "exits": final_exits,
        "entries": [],
        "positions": {"long": {}, "short": {}},
        "today_profit": final_pnl,
        "total_profit": total_profit,
        "total_investment": account_value,
    }
    daily_data.append(final_record)
    portfolio_statistics.append({
        "date": final_day_s,
        "investment": account_value,
        "today_profit": final_pnl,
        "total_profit": total_profit,
    })
    with (output_dir / f"portfolio_{final_day_s}.json").open("w", encoding="utf-8") as f:
        json.dump(final_record, f, indent=2)

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = simulate(
        sentiment_file=Path(args.sentiment_file),
        output_dir=Path(args.output_dir),
        initial_capital=args.initial_capital,
        window_days=args.window_days,
        max_positions=args.max_positions,
    )
    meta = result["meta"]
    print(f"Final value: ${meta['final_value']:,.2f}")
    print(f"Total return: {meta['total_return_pct']:.2f}%")
    print(f"Saved simulation to {args.output_dir}")
