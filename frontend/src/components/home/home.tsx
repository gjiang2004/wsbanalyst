import { ArrowRight, BarChart3, Bot, LineChart, MessageSquareText } from 'lucide-react';
import { Link } from 'react-router-dom';
import wsbLogo from '../../assets/WSB.png';
import { pageTopPosts, pageWsbChatbot, pageTrading } from '../../router/router';
import './home.css';

const navItems = [
  { label: 'Top Posts', to: pageTopPosts },
  { label: 'Chatbot', to: pageWsbChatbot },
  { label: 'Simulation', to: pageTrading },
];

const featureCards = [
  {
    title: 'Ticker Sentiment',
    text: 'Ranks WSB tickers from a rolling Reddit bank using FinBERT, engagement weighting, and recency decay.',
    icon: BarChart3,
  },
  {
    title: 'Evidence View',
    text: 'Click a ticker to see the posts and comments contributing most to bullish and bearish sentiment.',
    icon: MessageSquareText,
  },
  {
    title: 'Grounded WSB Bot',
    text: 'A Mistral chatbot tuned for WSB style while pulling current stock context and recent sentiment at runtime.',
    icon: Bot,
  },
  {
    title: 'Market-Open Simulation',
    text: 'Tests a rolling sentiment strategy using market-open dates, planned entries, exits, and daily account value.',
    icon: LineChart,
  },
];

const pipelineItems = [
  '28-day raw Reddit bank',
  '1/3/7/14-day sentiment views',
  '15-minute incremental updates',
  'Nightly score refresh',
];

export const Home = () => {
  return (
    <main className="home-page">
      <nav className="home-nav" aria-label="Home navigation">
        <Link className="home-brand" to="/">
          <img src={wsbLogo} alt="WSB Analyst logo" />
          <span>WSB Analyst</span>
        </Link>
        <div className="home-nav-links">
          {navItems.map((item) => (
            <Link key={item.to} to={item.to}>{item.label}</Link>
          ))}
        </div>
      </nav>

      <section className="home-hero">
        <div className="home-hero-copy">
          <span className="home-kicker">Reddit sentiment, market data, and simulation</span>
          <h1>Track what WSB is yelling about before you trade it.</h1>
          <p>
            WSB Analyst turns recent r/wallstreetbets posts and comments into ticker sentiment,
            shows the evidence behind each signal, and tests those signals in a market-open trading simulation.
          </p>
          <div className="home-actions">
            <Link className="home-primary" to={pageTopPosts}>
              View Top Posts <ArrowRight size={18} />
            </Link>
            <Link className="home-secondary" to={pageWsbChatbot}>Open Chatbot</Link>
          </div>
        </div>

        <div className="home-hero-panel" aria-label="System overview">
          <div className="hero-panel-header">
            <div className="hero-panel-icon" aria-hidden="true">
              <LineChart size={30} strokeWidth={2.1} />
            </div>
            <div>
              <span>Current strategy</span>
              <strong>Sentiment rebalance</strong>
            </div>
          </div>
          <div className="hero-metrics">
            <div>
              <span>Raw bank</span>
              <strong>28d</strong>
            </div>
            <div>
              <span>Simulation</span>
              <strong>7d</strong>
            </div>
            <div>
              <span>Cadence</span>
              <strong>15m</strong>
            </div>
          </div>
          <div className="hero-status-list">
            <div>Incremental Reddit updates</div>
            <div>Facts grounded at runtime</div>
            <div>Market-open trades only</div>
          </div>
        </div>
      </section>

      <section className="home-feature-band">
        <div className="section-heading">
          <h2>What the app does</h2>
          <p>A quick overview of the main pieces powering the app.</p>
        </div>
        <div className="home-feature-grid">
          {featureCards.map(({ title, text, icon: Icon }) => (
            <article className="home-feature-card" key={title}>
              <Icon size={22} />
              <h3>{title}</h3>
              <p>{text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="home-pipeline-section">
        <div className="pipeline-copy">
          <h2>Built for rolling data, not stale snapshots.</h2>
          <p>
            The sentiment pages can switch between 1, 3, 7, and 14 days, while the simulator uses a faster 7-day signal
            before opening trades.
          </p>
        </div>
        <div className="pipeline-list">
          {pipelineItems.map((item, index) => (
            <div className="pipeline-item" key={item}>
              <span>{String(index + 1).padStart(2, '0')}</span>
              <strong>{item}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="home-chat-strip">
        <div className="home-chat-icon" aria-hidden="true">
          <Bot size={46} strokeWidth={2.05} />
        </div>
        <div>
          <h2>Chat in WSB style, but keep facts grounded.</h2>
          <p>The chatbot learns tone from WSB comments and uses live stock/sentiment context instead of memorizing stale numbers.</p>
        </div>
        <Link to={pageWsbChatbot}>Try it <ArrowRight size={18} /></Link>
      </section>

      <footer className="home-footer">
        <span>WSB Analyst</span>
        <p>Research tool only. Not financial advice.</p>
      </footer>
    </main>
  );
};

export default Home;
