import { useEffect, useState, useCallback, useMemo } from 'react';
import './topposts.css';
import Sidebar from '../sidebar/sidebar';
import { TrendingUp, TrendingDown, Activity, RefreshCw, AlertCircle, ChevronsUpDown, ChevronDown, ChevronUp } from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface TickerRow {
  ticker:           string;
  company:          string;
  mentions:         number;
  sentiment:        'bullish' | 'bearish' | 'neutral';
  bullish_pct:      number;
  bearish_pct:      number;
  neutral_pct:      number;
  normalized_score: number;
  score:            number;
}

interface SentimentData {
  trending: TickerRow[];
  bullish:  TickerRow[];
  bearish:  TickerRow[];
  meta:     { total_tickers: number };
}

type Tab        = 'trending' | 'bullish' | 'bearish';
type SortCol    = 'rank' | 'ticker' | 'mentions' | 'bullish_pct' | 'sentiment';
type SortDir    = 'default' | 'desc' | 'asc';

const TAB_ORDER: Tab[] = ['trending', 'bullish', 'bearish'];

// ─── Config ───────────────────────────────────────────────────────────────────

const TAB_CONFIG: Record<Tab, { label: string; hint: string; icon: typeof Activity }> = {
  trending: { label: 'Trending',     hint: 'ranked by mentions',          icon: Activity     },
  bullish:  { label: 'Most Bullish', hint: 'bullish conviction × volume', icon: TrendingUp   },
  bearish:  { label: 'Most Bearish', hint: 'bearish conviction × volume', icon: TrendingDown },
};

const TAB_ACTIVE_CLASS: Record<Tab, string> = {
  trending: 'tp-tab--trending',
  bullish:  'tp-tab--bullish',
  bearish:  'tp-tab--bearish',
};

const SENTIMENT_ORDER = { bullish: 0, neutral: 1, bearish: 2 };

const API_URL = 'http://localhost:8000/top-posts';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function nextDir(current: SortDir): SortDir {
  if (current === 'default') return 'desc';
  if (current === 'desc')    return 'asc';
  return 'default';
}

function sortRows(rows: TickerRow[], col: SortCol, dir: SortDir): TickerRow[] {
  if (dir === 'default') return rows;
  const sorted = [...rows].sort((a, b) => {
    switch (col) {
      case 'rank':       return a.mentions - b.mentions; // rank = original order proxy
      case 'ticker':     return a.ticker.localeCompare(b.ticker);
      case 'mentions':   return a.mentions - b.mentions;
      case 'bullish_pct':return a.bullish_pct - b.bullish_pct;
      case 'sentiment':  return SENTIMENT_ORDER[a.sentiment] - SENTIMENT_ORDER[b.sentiment];
      default:           return 0;
    }
  });
  return dir === 'desc' ? sorted.reverse() : sorted;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function SortIcon({ col, activeCol, dir }: { col: SortCol; activeCol: SortCol; dir: SortDir }) {
  if (col !== activeCol || dir === 'default') return <ChevronsUpDown size={10} className="tp-sort-icon tp-sort-icon--idle" />;
  return dir === 'desc'
    ? <ChevronDown size={10} className="tp-sort-icon tp-sort-icon--active" />
    : <ChevronUp   size={10} className="tp-sort-icon tp-sort-icon--active" />;
}

function SentimentBar({ bullishPct, bearishPct, neutralPct }: {
  bullishPct: number;
  bearishPct: number;
  neutralPct: number;
}) {
  return (
    <div className="tp-bar-wrap">
      <div className="tp-bar">
        <div className="tp-bar-segment tp-bar-segment--bull"    style={{ width: `${bullishPct}%` }} />
        <div className="tp-bar-segment tp-bar-segment--neutral" style={{ width: `${neutralPct}%` }} />
        <div className="tp-bar-segment tp-bar-segment--bear"    style={{ width: `${bearishPct}%` }} />
      </div>
      <div className="tp-bar-labels">
        <span className="tp-bar-label tp-bar-label--bull">{bullishPct.toFixed(0)}%</span>
        <span className="tp-bar-label tp-bar-label--bear">{bearishPct.toFixed(0)}%</span>
      </div>
    </div>
  );
}

function TickerCell({ ticker, company, sentiment }: {
  ticker:    string;
  company:   string;
  sentiment: string;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      className="tp-ticker-wrap"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <span className={`tp-ticker tp-ticker--${sentiment}`}>{ticker}</span>
      {company && hovered && <div className="tp-tooltip">{company}</div>}
    </div>
  );
}

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 10 }).map((_, i) => (
        <tr key={i} className="tp-skeleton-row">
          {[24, 64, 48, 160, 72, 80].map((w, j) => (
            <td key={j}><div className="tp-skeleton" style={{ width: w }} /></td>
          ))}
        </tr>
      ))}
    </>
  );
}

function TableRow({ row, rank }: { row: TickerRow; rank: number }) {
  return (
    <tr className="tp-row">
      <td className="tp-td tp-td--rank">{rank}</td>

      <td className="tp-td">
        <TickerCell ticker={row.ticker} company={row.company} sentiment={row.sentiment} />
      </td>

      <td className="tp-td tp-td--right">
        <span className="tp-mono">{row.mentions.toLocaleString()}</span>
      </td>

      <td className="tp-td">
        <SentimentBar
          bullishPct={row.bullish_pct}
          bearishPct={row.bearish_pct}
          neutralPct={row.neutral_pct}
        />
      </td>

      <td className="tp-td tp-td--center">
        <span className={`tp-sentiment-badge tp-sentiment-badge--${row.sentiment}`}>
          {row.sentiment}
        </span>
      </td>

      <td className="tp-td tp-td--right">
        <span className={`tp-mono tp-norm-score tp-norm-score--${row.normalized_score >= 0 ? 'pos' : 'neg'}`}>
          {row.normalized_score >= 0 ? '+' : ''}{row.normalized_score.toFixed(4)}
        </span>
      </td>
    </tr>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────

export const TopPosts = () => {
  const [data,        setData]        = useState<SentimentData | null>(null);
  const [error,       setError]       = useState<string | null>(null);
  const [loading,     setLoading]     = useState(true);
  const [tab,         setTab]         = useState<Tab>('trending');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [sortCol,     setSortCol]     = useState<SortCol>('rank');
  const [sortDir,     setSortDir]     = useState<SortDir>('default');

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(API_URL);
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const json: SentimentData & { error?: string } = await res.json();
      if (json.error) throw new Error(json.error);
      setData(json);
      setLastUpdated(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  // Reset sort when tab changes
  useEffect(() => {
    setSortCol('rank');
    setSortDir('default');
  }, [tab]);

  const handleColSort = (col: SortCol) => {
    if (col === sortCol) {
      setSortDir(prev => nextDir(prev));
    } else {
      setSortCol(col);
      setSortDir('desc');
    }
  };

  // Overall column cycles through tabs instead of sorting
  const handleOverallClick = () => {
    const idx  = TAB_ORDER.indexOf(tab);
    const next = TAB_ORDER[(idx + 1) % TAB_ORDER.length];
    setTab(next);
  };

  const rawRows = data?.[tab] ?? [];

  const rows = useMemo(
    () => sortRows(rawRows, sortCol, sortDir),
    [rawRows, sortCol, sortDir]
  );

  const { icon: TabIcon, hint } = TAB_CONFIG[tab];

  return (
    <div className="tp-layout">
      <Sidebar onToggle={() => {}} />

      <main className="tp-main">

        {/* Header */}
        <div className="tp-header">
          <div>
            <div className="tp-title-row">
              <h1 className="tp-title">WSB Sentiment</h1>
              <span className="tp-badge">r/wallstreetbets</span>
            </div>
            <p className="tp-subtitle">
              {data
                ? `${data.meta.total_tickers} tickers tracked · last 30 days`
                : 'loading market sentiment...'}
            </p>
          </div>

          <div className="tp-header-right">
            {lastUpdated && (
              <span className="tp-timestamp">
                {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            )}
            <button
              className={`tp-refresh ${loading ? 'tp-refresh--spinning' : ''}`}
              onClick={fetchData}
              disabled={loading}
              title="Refresh"
            >
              <RefreshCw size={13} />
            </button>
          </div>
        </div>

        {/* Card */}
        <div className="tp-card">

          {/* Tabs */}
          <div className="tp-tabs">
            {(Object.keys(TAB_CONFIG) as Tab[]).map((t) => {
              const { label, icon: Icon } = TAB_CONFIG[t];
              return (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={`tp-tab ${t === tab ? TAB_ACTIVE_CLASS[t] : ''}`}
                >
                  <Icon size={12} />
                  {label}
                </button>
              );
            })}
            <span className="tp-tab-hint">
              <TabIcon size={11} />
              {hint}
            </span>
          </div>

          {/* Error */}
          {error && (
            <div className="tp-error">
              <AlertCircle size={14} />
              {error}
            </div>
          )}

          {/* Table */}
          {!error && (
            <div className="tp-table-wrap">
              <table className="tp-table">
                <thead>
                  <tr>
                    <th
                      className="tp-th tp-th--right tp-th--sortable"
                      style={{ width: 28 }}
                      onClick={() => handleColSort('rank')}
                    >
                      # <SortIcon col="rank" activeCol={sortCol} dir={sortDir} />
                    </th>

                    <th
                      className="tp-th tp-th--left tp-th--sortable"
                      style={{ width: 88 }}
                      onClick={() => handleColSort('ticker')}
                    >
                      Ticker <SortIcon col="ticker" activeCol={sortCol} dir={sortDir} />
                    </th>

                    <th
                      className="tp-th tp-th--right tp-th--sortable"
                      style={{ width: 80 }}
                      onClick={() => handleColSort('mentions')}
                    >
                      Mentions <SortIcon col="mentions" activeCol={sortCol} dir={sortDir} />
                    </th>

                    <th
                      className="tp-th tp-th--left tp-th--sortable"
                      style={{ width: 200 }}
                      onClick={() => handleColSort('bullish_pct')}
                    >
                      Sentiment Split <SortIcon col="bullish_pct" activeCol={sortCol} dir={sortDir} />
                    </th>

                    <th
                      className="tp-th tp-th--center tp-th--sortable tp-th--overall"
                      style={{ width: 88 }}
                      onClick={handleOverallClick}
                      title="Click to cycle: Trending → Bullish → Bearish"
                    >
                      Overall
                      <span className="tp-th-cycle-hint">
                        {tab === 'trending' ? '→ bullish' : tab === 'bullish' ? '→ bearish' : '→ trending'}
                      </span>
                    </th>

                    <th className="tp-th tp-th--right" style={{ width: 96 }}>
                      Avg Score
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {loading && !data
                    ? <SkeletonRows />
                    : rows.map((row, i) => (
                        <TableRow key={row.ticker} row={row} rank={i + 1} />
                      ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Legend */}
          <div className="tp-legend">
            <span className="tp-legend-dot tp-legend-dot--bull" /> Bullish
            <span className="tp-legend-sep" />
            <span className="tp-legend-dot tp-legend-dot--neutral" /> Neutral
            <span className="tp-legend-sep" />
            <span className="tp-legend-dot tp-legend-dot--bear" /> Bearish
            <span className="tp-legend-credit">FinBERT · r/wallstreetbets</span>
          </div>

        </div>
      </main>
    </div>
  );
};

export default TopPosts;