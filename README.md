# WSB Analyst

WSB Analyst is a React + FastAPI app for tracking r/wallstreetbets ticker sentiment, viewing the posts/comments behind each ticker, chatting with a WSB-style Mistral bot, and running a sentiment-driven trading simulation.

This is research software, not financial advice.

## Current Data Model

- Raw Reddit posts/comments are stored in a database by default. Local development uses SQLite at `wsb_data.sqlite3`; production can use Postgres via `DATABASE_URL`.
- `wsb_posts.json` is still exported as a compatibility snapshot for analysis/training scripts. It is configured for 28 days.
- `ticker_sentiment.json` is the active Top Posts sentiment output. It is calculated from the latest 14 days only.
- `backend/agg_sentiment.json` stores daily ticker sentiment rows used by the simulator.
- The simulator uses a rolling 14-day sentiment lookback. With a 28-day bank, the first 14 days warm up the signal and the next ~14 days can be simulated.

The regular Reddit API does not page indefinitely into older subreddit history, so the app collects forward from scheduled runs instead of relying on archive backfills.

## Requirements

- Python 3.11 or 3.12
- Node 20+
- Reddit API credentials in `.env`
- Optional NVIDIA GPU for FinBERT and Mistral training/inference

Create `.env` from `.env.example`:

```bash
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=wsbanalyst scraper
```

## Install

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r requirements-sentiment.txt
cd frontend
npm install
```

## Run Locally

Backend:

```bash
venv/bin/uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Build check:

```bash
cd frontend
npm run build
```


## Database

The sentiment updater uses DB storage by default (`WSB_STORAGE=db`). If `DATABASE_URL` is omitted, it writes to local SQLite:

```bash
DATABASE_URL=sqlite:///wsb_data.sqlite3
```

For production, point `DATABASE_URL` at Postgres:

```bash
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
```

The pipeline still exports `wsb_posts.json`, `ticker_sentiment.json`, and `backend/agg_sentiment.json` so the existing analyzer, frontend, simulation, and chatbot flows continue to work. On the first DB run, `update_sentiment.py` seeds the database from `wsb_posts.json` if the database is empty.

## Refresh Sentiment

Incremental refresh for normal use:

```bash
venv/bin/python update_sentiment.py   --storage db   --scrape-days 1   --window-days 28   --aggregate-days 14   --store-file wsb_posts.json   --output ticker_sentiment.json   --daily-output backend/agg_sentiment.json
```

GitHub Actions runs this every 15 minutes with overlap so boundary posts/comments are merged by ID instead of missed. A separate nightly workflow refreshes scores for the active 14-day sentiment window.

## Run Simulation

```bash
venv/bin/python portfolio.py   --sentiment-file backend/agg_sentiment.json   --output-dir frontend/src/portfolio_data   --initial-capital 1000000   --window-days 14   --max-positions 25
```

The simulation buys or shorts at the market open, closes at the next market open, then rebalances using the latest rolling sentiment signal.

## Mistral Chatbot

The chatbot should learn style from fine-tuning and get facts from runtime context. Do not train exact prices or news into the model; those go stale. Runtime inference pulls:

- current stock data from `yfinance`
- ticker sentiment from `ticker_sentiment.json`
- bullish/bearish sample snippets from recent WSB posts/comments

Build training data:

```bash
venv/bin/python pretrain.py --input wsb_posts.json --output wsb_training_data.jsonl
```

Fine-tune:

```bash
venv/bin/python finetune.py
```

Run CLI inference:

```bash
venv/bin/python inference.py
```

## Repository Layout

- `frontend/` - React app
- `backend/api.py` - FastAPI backend
- `getdata.py` - Reddit scraping helpers
- `update_sentiment.py` - rolling scrape + FinBERT sentiment pipeline
- `analyze_wsb.py` - ticker extraction and sentiment aggregation
- `portfolio.py` - open-to-open trading simulation
- `pretrain.py`, `finetune.py`, `inference.py` - Mistral style training and grounded chatbot inference
