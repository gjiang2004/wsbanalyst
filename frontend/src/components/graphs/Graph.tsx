import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import dayjs from "dayjs";
import sampleData from "../../portfolio_data/portfolio_total_investment.json";
import "./Graph.css";

interface PortfolioPoint {
  date: string;
  investment: number;
  today_profit: number;
  total_profit: number;
}

interface Trade {
  ticker?: string;
  side?: "long" | "short" | string;
  action?: string;
  entry_date?: string;
  entry_price?: number;
  exit_date?: string;
  exit_price?: number;
  shares?: number;
  notional?: number;
  weight?: number;
  sentiment?: number;
  pnl?: number;
  price?: number;
  cost?: number;
}

interface DailyRecord {
  date: string;
  next_trade_date?: string;
  starting_value?: number;
  realized_pnl?: number;
  ending_value_before_rebalance?: number;
  allocated_value?: number;
  cash_after_rebalance?: number;
  today_profit?: number;
  total_profit?: number;
  total_investment?: number;
  exits?: Trade[];
  entries?: Trade[];
  trades?: Trade[];
  planned_only?: boolean;
}

interface SimulationData {
  meta?: {
    initial_capital?: number;
    final_value?: number;
    total_return_pct?: number;
    rolling_sentiment_window_days?: number;
    max_positions?: number;
  };
  daily_data?: DailyRecord[];
  portfolio_statistics?: PortfolioPoint[];
}

interface ChartPoint extends PortfolioPoint {
  formattedValue: string;
}

const simulation = sampleData as SimulationData;

const currency = (value?: number) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 })
    : "--";

const number = (value?: number, digits = 2) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "--";

const compactCurrency = (value: number) =>
  value.toLocaleString("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1 });

const tradeLabel = (trade: Trade, fallback: "entry" | "exit") => {
  if (trade.side) return trade.side === "short" ? "Short" : "Long";
  if (trade.action) return trade.action.toLowerCase() === "short" ? "Short" : "Long";
  return fallback === "exit" ? "Close" : "Open";
};

const Graph = () => {
  const chartData = useMemo<ChartPoint[]>(() => {
    const stats = Array.isArray(simulation.portfolio_statistics) ? simulation.portfolio_statistics : [];
    return stats
      .filter((entry) => entry.date && Number.isFinite(entry.investment))
      .map((entry) => ({
        date: entry.date,
        investment: entry.investment,
        today_profit: entry.today_profit || 0,
        total_profit: entry.total_profit || 0,
        formattedValue: currency(entry.investment),
      }))
      .sort((a, b) => a.date.localeCompare(b.date));
  }, []);

  const dailyByDate = useMemo(() => {
    const map = new Map<string, DailyRecord>();
    const rows = Array.isArray(simulation.daily_data) ? simulation.daily_data : [];
    rows.forEach((row) => {
      if (row.date) map.set(row.date, row);
    });
    return map;
  }, []);

  const [selectedDate, setSelectedDate] = useState(() => chartData.length ? chartData[chartData.length - 1].date : "");
  const [showClosedTrades, setShowClosedTrades] = useState(false);
  const selectedPoint = chartData.find((point) => point.date === selectedDate) || chartData[chartData.length - 1];
  const selectedDay = selectedPoint ? dailyByDate.get(selectedPoint.date) : undefined;
  const chartWidth = Math.max(820, chartData.length * 54);

  const stats = useMemo(() => {
    const first = chartData[0];
    const last = chartData.length ? chartData[chartData.length - 1] : undefined;
    const initial = simulation.meta?.initial_capital || first?.investment || 1_000_000;
    const totalProfit = last ? last.investment - initial : 0;
    const returnPct = initial ? (totalProfit / initial) * 100 : 0;
    const bestDay = chartData.reduce<ChartPoint | undefined>((best, point) => {
      if (!best || point.today_profit > best.today_profit) return point;
      return best;
    }, undefined);
    const worstDay = chartData.reduce<ChartPoint | undefined>((worst, point) => {
      if (!worst || point.today_profit < worst.today_profit) return point;
      return worst;
    }, undefined);

    return { first, last, initial, totalProfit, returnPct, bestDay, worstDay };
  }, [chartData]);

  const handleChartClick = (state: { activeLabel?: string } | null) => {
    if (state?.activeLabel) setSelectedDate(state.activeLabel);
  };

  const renderTradeRows = (title: string, trades: Trade[] | undefined, fallback: "entry" | "exit") => {
    if (!trades?.length) {
      return <p className="trade-empty">No trades recorded for this side of the rebalance.</p>;
    }

    return (
      <div className="trade-list">
        {trades.map((trade, index) => (
          <article className="trade-row" key={`${title}-${trade.ticker || index}-${index}`}>
            <div>
              <strong>{trade.ticker || "Unknown"}</strong>
              <span className={tradeLabel(trade, fallback).toLowerCase()}>{tradeLabel(trade, fallback)}</span>
            </div>
            <dl>
              <div>
                <dt>Notional</dt>
                <dd>{currency(trade.notional ?? Math.abs(trade.cost || 0))}</dd>
              </div>
              <div>
                <dt>Shares</dt>
                <dd>{number(trade.shares, 4)}</dd>
              </div>
              <div>
                <dt>{fallback === "exit" ? "Exit" : "Entry"}</dt>
                <dd>{currency(fallback === "exit" ? trade.exit_price : trade.entry_price ?? trade.price)}</dd>
              </div>
              <div>
                <dt>{fallback === "exit" ? "P&L" : "Signal"}</dt>
                <dd className={(fallback === "exit" ? trade.pnl || 0 : trade.sentiment || 0) >= 0 ? "positive" : "negative"}>
                  {fallback === "exit" ? currency(trade.pnl) : number(trade.sentiment, 4)}
                </dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
    );
  };

  const renderTrades = (title: string, trades: Trade[] | undefined, fallback: "entry" | "exit") => (
    <section className="trade-panel">
      <h3>{title}</h3>
      {renderTradeRows(title, trades, fallback)}
    </section>
  );

  if (!chartData.length) {
    return (
      <div className="graph-page">
        <div className="graph-empty">
          <p>No simulation data available.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="graph-page">
      <div className="graph-header">
        <div>
          <h1>Simulation</h1>
          <p>Open-to-open sentiment rebalance using a rolling {simulation.meta?.rolling_sentiment_window_days || 14}-day WSB signal on market-open days only.</p>
        </div>
      </div>

      <div className="stats-container">
        <div className="stat-card">
          <p>Current Value</p>
          <p>{currency(stats.last?.investment)}</p>
        </div>
        <div className="stat-card">
          <p>Total Profit</p>
          <p className={stats.totalProfit >= 0 ? "positive" : "negative"}>{currency(stats.totalProfit)}</p>
        </div>
        <div className="stat-card">
          <p>Total Return</p>
          <p className={stats.returnPct >= 0 ? "positive" : "negative"}>{number(stats.returnPct)}%</p>
        </div>
        <div className="stat-card">
          <p>Trading Days</p>
          <p>{chartData.length}</p>
        </div>
      </div>

      <section className="simulation-rule-card">
        <strong>Trade selection</strong>
        <p>Each market-open day, the simulator ranks tickers by absolute rolling sentiment strength, keeps up to {simulation.meta?.max_positions || 25} tickers with usable open prices for both entry and next-session exit, and allocates the full account value across those trades by signal weight.</p>
      </section>

      <section className="simulation-chart-card">
        <div className="chart-title-row">
          <div>
            <h2>Account Value by Day</h2>
            <p>Scroll horizontally as the backtest grows. Click a day to inspect the rebalance.</p>
          </div>
          <div className="selected-pill">{selectedPoint ? dayjs(selectedPoint.date).format("MMM D, YYYY") : "--"}</div>
        </div>

        <div className="chart-scroll" role="region" aria-label="Scrollable simulation chart">
          <div className="chart-inner" style={{ width: chartWidth }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 28, left: 22, bottom: 34 }} onClick={handleChartClick}>
                <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                <XAxis
                  dataKey="date"
                  tickFormatter={(date) => dayjs(date).format("MMM D")}
                  tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(255,255,255,0.16)" }}
                  tickLine={false}
                  minTickGap={18}
                />
                <YAxis
                  dataKey="investment"
                  tickFormatter={compactCurrency}
                  tick={{ fill: "rgba(255,255,255,0.62)", fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                  width={84}
                />
                <Tooltip
                  formatter={(value: number, name: string) => [currency(value), name === "investment" ? "Account value" : name]}
                  labelFormatter={(date) => dayjs(String(date)).format("MMM D, YYYY")}
                  contentStyle={{
                    background: "rgba(16, 18, 27, 0.96)",
                    border: "1px solid rgba(255,255,255,0.14)",
                    borderRadius: "8px",
                    color: "rgba(255,255,255,0.9)",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="investment"
                  name="Account value"
                  stroke="#5eead4"
                  strokeWidth={3}
                  dot={{ r: 3, fill: "#5eead4", strokeWidth: 0 }}
                  activeDot={{ r: 7, fill: "#f8fafc", stroke: "#5eead4", strokeWidth: 3 }}
                />
                {selectedPoint && (
                  <ReferenceDot
                    x={selectedPoint.date}
                    y={selectedPoint.investment}
                    r={7}
                    fill="#f8fafc"
                    stroke="#5eead4"
                    strokeWidth={3}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="day-detail-card">
        <div className="day-detail-header">
          <div>
            <h2>{selectedPoint ? dayjs(selectedPoint.date).format("MMMM D, YYYY") : "Selected Day"}</h2>
            <p>{selectedDay?.next_trade_date ? `Positions opened for next market open: ${selectedDay.next_trade_date}` : selectedDay?.planned_only ? "Planned current-day entries. Results are unknown until the next market-open exit." : "Final liquidation or legacy simulation record."}</p>
          </div>
          <div className="day-value">
            <span>Account</span>
            <strong>{currency(selectedPoint?.investment)}</strong>
          </div>
        </div>

        <div className="detail-metrics">
          <div>
            <span>Daily P&L</span>
            <strong className={(selectedPoint?.today_profit || 0) >= 0 ? "positive" : "negative"}>{currency(selectedPoint?.today_profit)}</strong>
          </div>
          <div>
            <span>Total P&L</span>
            <strong className={(selectedPoint?.total_profit || 0) >= 0 ? "positive" : "negative"}>{currency(selectedPoint?.total_profit)}</strong>
          </div>
          <div>
            <span>Allocated</span>
            <strong>{currency(selectedDay?.allocated_value)}</strong>
          </div>
          <div>
            <span>Cash</span>
            <strong>{currency(selectedDay?.cash_after_rebalance)}</strong>
          </div>
        </div>

        <div className="trade-grid">
          <section className="trade-panel closed-trades-panel">
            <button
              className="closed-trades-toggle"
              type="button"
              onClick={() => setShowClosedTrades((current) => !current)}
              aria-expanded={showClosedTrades}
            >
              <span>Closed at 9:30</span>
              <strong>{selectedDay?.exits?.length || 0}</strong>
            </button>
            {showClosedTrades && renderTradeRows("Closed Trades", selectedDay?.exits, "exit")}
          </section>
          {renderTrades(selectedDay?.planned_only ? "Planned for 9:30" : "Opened at 9:30", selectedDay?.entries || selectedDay?.trades, "entry")}
        </div>
      </section>
    </div>
  );
};

export default Graph;
