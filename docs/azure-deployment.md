# Azure Deployment Plan

This app should run on Azure with:

- Azure Static Web Apps for `frontend/`
- Azure App Service or Azure Container Apps for the FastAPI API
- Azure Database for PostgreSQL Flexible Server for live data
- Azure Container Apps Jobs for the 15-minute updater and nightly refresh

## Required App Settings

Set these on the API app and worker jobs:

```text
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/wsb?sslmode=require
WSB_API_STORAGE=db
WSB_STORAGE=db
SIM_STORAGE=db
FINBERT_CACHE_STORAGE=db
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=wsbanalyst scraper
```

Set this on the frontend build:

```text
VITE_API_BASE_URL=https://YOUR-API.azurewebsites.net
```

## Build Images

```bash
docker build -f Dockerfile.api -t YOUR_REGISTRY/wsb-api:latest .
docker build -f Dockerfile.worker -t YOUR_REGISTRY/wsb-worker:latest .
docker push YOUR_REGISTRY/wsb-api:latest
docker push YOUR_REGISTRY/wsb-worker:latest
```

## API Startup

The API image starts with:

```bash
gunicorn -k uvicorn.workers.UvicornWorker backend.api:app --bind=0.0.0.0:8000 --timeout=600
```

If using App Service without Docker, use:

```bash
gunicorn -k uvicorn.workers.UvicornWorker backend.api:app --bind=0.0.0.0 --timeout 600
```

## Container Apps Jobs

15-minute incremental updater:

```bash
az containerapp job create \
  --name wsb-15min-updater \
  --resource-group wsb-prod-rg \
  --environment wsb-jobs-env \
  --trigger-type Schedule \
  --cron-expression "*/15 * * * *" \
  --replica-timeout 3600 \
  --replica-retry-limit 0 \
  --parallelism 1 \
  --replica-completion-count 1 \
  --image YOUR_REGISTRY/wsb-worker:latest \
  --cpu 2 \
  --memory 4Gi \
  --command python \
  --args update_sentiment.py --storage db --scrape-days 1 --window-days 28 --aggregate-days 14 --top-window-days 1,3,7 --request-delay 0.05 --overlap-minutes 30 --score-refresh-days 3 --max-score-refresh 100 --comment-refresh-days 3 --max-comment-refresh-posts 40
```

Nightly refresh:

```bash
az containerapp job create \
  --name wsb-nightly-refresh \
  --resource-group wsb-prod-rg \
  --environment wsb-jobs-env \
  --trigger-type Schedule \
  --cron-expression "0 4 * * *" \
  --replica-timeout 7200 \
  --replica-retry-limit 0 \
  --parallelism 1 \
  --replica-completion-count 1 \
  --image YOUR_REGISTRY/wsb-worker:latest \
  --cpu 2 \
  --memory 4Gi \
  --command python \
  --args update_sentiment.py --storage db --scrape-days 1 --window-days 28 --aggregate-days 14 --top-window-days 1,3,7 --request-delay 0.05 --overlap-minutes 60 --score-refresh-days 14 --max-score-refresh 0 --comment-refresh-days 14 --max-comment-refresh-posts 0
```

Azure scheduled job cron is UTC. `0 4 * * *` is midnight New York during daylight saving time.

## Current Migration Behavior

The app is DB-first when `WSB_API_STORAGE=db`, with JSON fallback kept for local development and rollback.

The updater in DB mode stores:

- raw Reddit posts/comments
- 1D/3D/7D/14D sentiment snapshots
- merged daily ticker sentiment history
- FinBERT sentiment cache

The simulator in DB mode reads daily ticker sentiment from DB and stores the full portfolio result in DB.

