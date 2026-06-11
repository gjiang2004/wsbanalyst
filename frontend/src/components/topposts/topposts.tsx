import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { Link } from 'react-router-dom';
import './topposts.css';
import { TrendingUp, TrendingDown, Activity, RefreshCw, AlertCircle, ChevronsUpDown, ChevronDown, ChevronUp, Search, Filter, CalendarDays } from 'lucide-react';
import { apiUrl } from '../../lib/api';
import Sidebar from '../sidebar/sidebar';

// ─── Types ────────────────────────────────────────────────────────────────────

interface TickerRow {
  rank:             number;
  ticker:           string;
  company:          string;
  asset_type:       string;
  mentions:         number;
  sentiment:        'bullish' | 'bearish' | 'neutral';
  bullish_pct:      number;
  bearish_pct:      number;
  neutral_pct:      number;
  normalized_score: number;
}

interface SentimentData {
  trending: TickerRow[];
  bullish:  TickerRow[];
  bearish:  TickerRow[];
  meta:     { total_tickers: number; limit?: number; sentiment_window_days?: number; selected_window_days?: number; available_window_days?: number[]; source_total_posts?: number };
}

type Tab        = 'trending' | 'bullish' | 'bearish';
type SortCol    = 'rank' | 'ticker' | 'mentions' | 'sentiment_split' | 'normalized_score';
type SortDir    = 'default' | 'desc' | 'asc';
type WindowDays = 1 | 3 | 7 | 14;


const DEFAULT_VISIBLE_ROWS = 100;
const FULL_SENTIMENT_LIMIT = 5000;
const WINDOW_OPTIONS: WindowDays[] = [1, 3, 7, 14];

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



// ─── Helpers ──────────────────────────────────────────────────────────────────

function nextDir(current: SortDir): SortDir {
  if (current === 'default') return 'desc';
  if (current === 'desc')    return 'asc';
  return 'default';
}

function sentimentSplitValue(row: TickerRow): number {
  return Math.max(row.bullish_pct, row.bearish_pct, row.neutral_pct);
}

function sortRows(rows: TickerRow[], col: SortCol, dir: SortDir): TickerRow[] {
  if (dir === 'default') return rows;
  const sorted = [...rows].sort((a, b) => {
    switch (col) {
      case 'rank':            return a.rank - b.rank;
      case 'ticker':          return a.ticker.localeCompare(b.ticker);
      case 'mentions':        return a.mentions - b.mentions;
      case 'sentiment_split': return sentimentSplitValue(a) - sentimentSplitValue(b);
      case 'normalized_score':return a.normalized_score - b.normalized_score;
      default:                return 0;
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
        <span className="tp-bar-label tp-bar-label--neutral">{neutralPct.toFixed(0)}%</span>
        <span className="tp-bar-label tp-bar-label--bear">{bearishPct.toFixed(0)}%</span>
      </div>
    </div>
  );
}

function TickerCell({ ticker, sentiment }: {
  ticker:    string;
  sentiment: string;
}) {
  return (
    <div className="tp-ticker-wrap">
      <Link className={`tp-ticker tp-ticker--${sentiment}`} to={`/ticker/${ticker}`}>{ticker}</Link>
    </div>
  );
}

function assetTagClass(assetType: string): string {
  return assetType.toLowerCase().replace(/[^a-z0-9]+/g, '-');
}

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 10 }).map((_, i) => (
        <tr key={i} className="tp-skeleton-row">
          {[24, 64, 72, 48, 160, 72, 80].map((w, j) => (
            <td key={j}><div className="tp-skeleton" style={{ width: w }} /></td>
          ))}
        </tr>
      ))}
    </>
  );
}

function TableRow({ row }: { row: TickerRow }) {
  return (
    <tr className="tp-row">
      <td className="tp-td tp-td--rank">{row.rank}</td>

      <td className="tp-td tp-td--ticker">
        <TickerCell ticker={row.ticker} sentiment={row.sentiment} />
      </td>

      <td className="tp-td tp-td--number">
        <span className="tp-mono">{row.mentions.toLocaleString()}</span>
      </td>

      <td className="tp-td tp-td--asset">
        <span className={`tp-asset-tag tp-asset-tag--${assetTagClass(row.asset_type)}`}>{row.asset_type || 'Unknown'}</span>
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

      <td className="tp-td tp-td--number">
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
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('All');
  const [sentimentFilter, setSentimentFilter] = useState<'All' | 'bullish' | 'bearish' | 'neutral'>('All');
  const [windowDays, setWindowDays] = useState<WindowDays>(14);
  const requestIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const fetchData = useCallback(async () => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    abortRef.current?.abort();

    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);
    setData(null);

    try {
      const params = new URLSearchParams({
        limit: String(FULL_SENTIMENT_LIMIT),
        window_days: String(windowDays),
      });
      const res = await fetch(apiUrl(`/top-posts?${params.toString()}`), {
        cache: 'no-store',
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const json: SentimentData & { error?: string } = await res.json();
      if (requestId !== requestIdRef.current) return;
      if (json.error) throw new Error(json.error);
      setData(json);
      setLastUpdated(new Date());
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return;
      if (requestId === requestIdRef.current) {
        setError(e instanceof Error ? e.message : 'Unknown error');
      }
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, [windowDays]);

  useEffect(() => { fetchData(); }, [fetchData]);

  useEffect(() => () => abortRef.current?.abort(), []);

  // Reset sort when tab changes
  useEffect(() => {
    setSortCol('rank');
    setSortDir('default');
    setTypeFilter('All');
    setSentimentFilter('All');
  }, [tab, windowDays]);

  const handleColSort = (col: SortCol) => {
    if (col === 'mentions') {
      setSortCol('mentions');
      setSortDir(prev => (sortCol === 'mentions' && prev === 'desc' ? 'asc' : 'desc'));
      return;
    }

    if (col === sortCol) {
      setSortDir(prev => nextDir(prev));
    } else {
      setSortCol(col);
      setSortDir('desc');
    }
  };

  const normalizedSearch = searchQuery.trim().toLowerCase();

  const availableTypes = useMemo(() => {
    const rawRows = data?.[tab] ?? [];
    return ['All', ...Array.from(new Set(rawRows.map((row) => row.asset_type || 'Unknown'))).sort()];
  }, [data, tab]);

  const handleTypeFilter = () => {
    setTypeFilter((current) => {
      const currentIndex = availableTypes.indexOf(current);
      return availableTypes[(currentIndex + 1) % availableTypes.length] || 'All';
    });
  };

  const handleSentimentFilter = () => {
    const order: Array<'All' | 'bullish' | 'bearish' | 'neutral'> = ['All', 'bullish', 'bearish', 'neutral'];
    setSentimentFilter((current) => order[(order.indexOf(current) + 1) % order.length]);
  };

  const rows = useMemo(() => {
    const rawRows = data?.[tab] ?? [];
    const searchedRows = normalizedSearch
      ? rawRows.filter((row) =>
          row.ticker.toLowerCase().includes(normalizedSearch) ||
          row.company.toLowerCase().includes(normalizedSearch) ||
          row.asset_type.toLowerCase().includes(normalizedSearch)
        )
      : rawRows;
    const typeFilteredRows = typeFilter === 'All'
      ? searchedRows
      : searchedRows.filter((row) => (row.asset_type || 'Unknown') === typeFilter);
    const sentimentFilteredRows = sentimentFilter === 'All'
      ? typeFilteredRows
      : typeFilteredRows.filter((row) => row.sentiment === sentimentFilter);
    return sortRows(sentimentFilteredRows, sortCol, sortDir);
  }, [data, tab, normalizedSearch, typeFilter, sentimentFilter, sortCol, sortDir]);

  const visibleRows = normalizedSearch ? rows : rows.slice(0, DEFAULT_VISIBLE_ROWS);

  const { icon: TabIcon, hint } = TAB_CONFIG[tab];
  const sentimentWindowDays = data?.meta.selected_window_days ?? data?.meta.sentiment_window_days ?? windowDays;

  return (
    <div className="tp-layout">
      <Sidebar />
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
                ? `${data.meta.total_tickers} tickers searchable · showing ${normalizedSearch ? rows.length : Math.min(DEFAULT_VISIBLE_ROWS, rows.length)} · last ${Number(sentimentWindowDays).toLocaleString(undefined, { maximumFractionDigits: 1 })} days`
                : 'loading market sentiment...'}
            </p>
          </div>

          <div className="tp-header-right">
            <label className="tp-search" aria-label="Search tickers">
              <Search size={15} />
              <input
                type="search"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search ticker or company"
              />
            </label>
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
            <div className="tp-window-control" aria-label="Sentiment window">
              <span className="tp-window-label"><CalendarDays size={11} /> Window</span>
              <div className="tp-window-buttons">
                {WINDOW_OPTIONS.map((days) => (
                  <button
                    key={days}
                    type="button"
                    className={`tp-window-button ${windowDays === days ? 'tp-window-button--active' : ''}`}
                    onClick={() => setWindowDays(days)}
                    aria-pressed={windowDays === days}
                    title={`Use last ${days} day${days === 1 ? '' : 's'} of sentiment`}
                  >
                    {days}D
                  </button>
                ))}
              </div>
            </div>
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
                      className="tp-th tp-th--left tp-th--sortable"
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
                      className="tp-th tp-th--left tp-th--sortable"
                      style={{ width: 80 }}
                      onClick={() => handleColSort('mentions')}
                    >
                      Mentions <SortIcon col="mentions" activeCol={sortCol} dir={sortDir} />
                    </th>

                    <th
                      className="tp-th tp-th--left tp-th--sortable tp-th--filter tp-th--type-filter"
                      style={{ width: 150 }}
                      onClick={handleTypeFilter}
                      title="Cycle type filter"
                    >
                      Type: {typeFilter} <Filter size={10} className="tp-sort-icon tp-sort-icon--active" />
                    </th>

                    <th
                      className="tp-th tp-th--left tp-th--sortable"
                      style={{ width: 200 }}
                      onClick={() => handleColSort('sentiment_split')}
                    >
                      Sentiment Split <SortIcon col="sentiment_split" activeCol={sortCol} dir={sortDir} />
                    </th>

                    <th
                      className="tp-th tp-th--center tp-th--sortable tp-th--overall tp-th--filter"
                      style={{ width: 150 }}
                      onClick={handleSentimentFilter}
                      title="Cycle sentiment filter"
                    >
                      Overall: {sentimentFilter} <Filter size={10} className="tp-sort-icon tp-sort-icon--active" />
                    </th>

                    <th
                      className="tp-th tp-th--left tp-th--sortable"
                      style={{ width: 96 }}
                      onClick={() => handleColSort('normalized_score')}
                    >
                      Avg Score <SortIcon col="normalized_score" activeCol={sortCol} dir={sortDir} />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {loading && !data
                    ? <SkeletonRows />
                    : visibleRows.length
                      ? visibleRows.map((row) => (
                          <TableRow key={row.ticker} row={row} />
                        ))
                      : (
                        <tr>
                          <td className="tp-empty-row" colSpan={7}>No tickers match your search.</td>
                        </tr>
                      )}
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