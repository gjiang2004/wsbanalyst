import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, ExternalLink } from 'lucide-react';
import { apiUrl } from '../../lib/api';
import Sidebar from '../sidebar/sidebar';
import './tickerdetail.css';

type Sentiment = 'bullish' | 'bearish' | 'neutral';

interface TickerSummary {
  ticker: string;
  company: string;
  mentions: number;
  sentiment: Sentiment;
  bullish_pct: number;
  bearish_pct: number;
  neutral_pct: number;
  normalized_score: number;
  asset_type: string;
}

interface SentimentSample {
  source: string;
  post_id?: string;
  comment_id?: string;
  permalink?: string;
  text: string;
  upvotes: number;
  awards: number;
  sentiment: 'positive' | 'negative' | 'neutral';
  sentiment_score: number;
  semantic_value: number;
  confidence: number;
  method: string;
}

interface TickerDetailData {
  ticker: string;
  company: string;
  asset_type: string;
  summary: TickerSummary;
  semantic_score: number;
  positive_score: number;
  negative_score: number;
  avg_confidence: number;
  detection_methods: Record<string, number>;
  positive_samples: SentimentSample[];
  negative_samples: SentimentSample[];
}

function formatNumber(value: number | undefined, digits = 2) {
  if (value === undefined || Number.isNaN(value)) return '0.00';
  return value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function SampleCard({ sample, tone }: { sample: SentimentSample; tone: 'positive' | 'negative' }) {
  const sourceLabel = sample.source === 'comment' ? 'Comment' : 'Post';
  const text = sample.text || 'No text available.';

  return (
    <article className={`td-sample td-sample--${tone}`}>
      <div className="td-sample-meta">
        <span>{sourceLabel}</span>
        <span>{sample.upvotes?.toLocaleString?.() ?? 0} votes</span>
        <span>score {formatNumber(sample.semantic_value, 3)}</span>
      </div>
      <p className="td-sample-text">{text}</p>
      <div className="td-sample-footer">
        <span>FinBERT {formatNumber(sample.sentiment_score, 3)}</span>
        <span>confidence {formatNumber(sample.confidence, 3)}</span>
        {sample.permalink && (
          <a href={sample.permalink} target="_blank" rel="noreferrer">
            <ExternalLink size={13} /> Reddit
          </a>
        )}
      </div>
    </article>
  );
}

export const TickerDetail = () => {
  const { ticker = '' } = useParams();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [data, setData] = useState<TickerDetailData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(apiUrl(`/ticker/${encodeURIComponent(ticker)}`))
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : 'Unknown error'))
      .finally(() => setLoading(false));
  }, [ticker]);

  const methods = useMemo(() => {
    if (!data) return [];
    return Object.entries(data.detection_methods).sort((a, b) => b[1] - a[1]);
  }, [data]);

  const companyName = data?.company || data?.summary?.company || '';

  return (
    <div className="td-layout">
      <Sidebar onToggle={setIsSidebarOpen} />
      <main className={`td-main ${isSidebarOpen ? '' : 'sidebar-closed'}`}>
        <Link className="td-back" to="/top-posts"><ArrowLeft size={16} /> Back to Top Posts</Link>

        {loading && <div className="td-panel">Loading ticker detail...</div>}
        {error && <div className="td-panel td-error">Failed to load ticker: {error}</div>}

        {data && (
          <>
            <section className="td-hero">
              <div className="td-hero-title">
                <h1>
                  {data.ticker}{companyName && <span className="td-company-name"> - {companyName}</span>}
                  <span className="td-asset-tag">{data.asset_type || data.summary.asset_type || 'Unknown'}</span>
                </h1>
              </div>
              <span className={`td-sentiment td-sentiment--${data.summary.sentiment}`}>{data.summary.sentiment}</span>
            </section>

            <section className="td-stats">
              <div><span>Mentions</span><strong>{data.summary.mentions.toLocaleString()}</strong></div>
              <div><span>Net Score</span><strong>{formatNumber(data.semantic_score, 3)}</strong></div>
              <div><span>Avg Score</span><strong>{formatNumber(data.summary.normalized_score, 4)}</strong></div>
              <div><span>Confidence</span><strong>{formatNumber(data.avg_confidence, 3)}</strong></div>
            </section>

            <section className="td-panel">
              <div className="td-split">
                <div className="td-bar">
                  <span className="td-bar-pos" style={{ width: `${data.summary.bullish_pct}%` }} />
                  <span className="td-bar-neutral" style={{ width: `${data.summary.neutral_pct}%` }} />
                  <span className="td-bar-neg" style={{ width: `${data.summary.bearish_pct}%` }} />
                </div>
                <div className="td-split-labels">
                  <span>{data.summary.bullish_pct.toFixed(0)}% bull</span>
                  <span>{data.summary.neutral_pct.toFixed(0)}% neutral</span>
                  <span>{data.summary.bearish_pct.toFixed(0)}% bear</span>
                </div>
              </div>
            </section>

            <section className="td-grid">
              <div>
                <div className="td-section-title"><h2>Positive Contributors</h2><span>{data.positive_samples.length}</span></div>
                {data.positive_samples.length ? data.positive_samples.map((sample, index) => (
                  <SampleCard key={`${sample.post_id}-${sample.comment_id}-${index}`} sample={sample} tone="positive" />
                )) : <div className="td-empty">No positive sample available for this ticker.</div>}
              </div>

              <div>
                <div className="td-section-title"><h2>Negative Contributors</h2><span>{data.negative_samples.length}</span></div>
                {data.negative_samples.length ? data.negative_samples.map((sample, index) => (
                  <SampleCard key={`${sample.post_id}-${sample.comment_id}-${index}`} sample={sample} tone="negative" />
                )) : <div className="td-empty">No negative sample available for this ticker.</div>}
              </div>
            </section>

            <section className="td-panel">
              <h2>Detection Methods</h2>
              <div className="td-methods">
                {methods.map(([method, count]) => <span key={method}>{method}: {count}</span>)}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
};

export default TickerDetail;
