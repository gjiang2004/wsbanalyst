#!/usr/bin/env bash
# Run the WSB app locally against the LIVE Azure Postgres database, so you can
# browse exactly the data that's in production right now.
#
#   ./run_local.sh          # start the backend API on http://localhost:8000
#   ./run_local.sh --help   # this message
#
# Then, in a second terminal:
#   cd frontend && npm run dev
# and open the printed URL (the dev frontend defaults to http://localhost:8000
# for the API, which is what this script serves).
#
# Requirements: the venv has the API deps (fastapi/uvicorn/psycopg) and your IP
# is allowed through the Postgres firewall (rule "temp-diag" / your home IP).
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  sed -n '2,16p' "$0"; exit 0
fi

# Load DATABASE_URL (and Postgres creds) from the Azure deploy env file.
if [ ! -f .azure-deploy.env ]; then
  echo "Missing .azure-deploy.env (needs DATABASE_URL for the Azure DB)." >&2
  exit 1
fi
set -a; . ./.azure-deploy.env; set +a

# Read live data from Postgres; disable the chatbot (no model configured locally).
export WSB_API_STORAGE=db
export WSB_STORAGE=db
export SIM_STORAGE=db
export WSB_CHAT_PROVIDER=none

echo "Serving live Azure DB at http://localhost:8000  (Ctrl-C to stop)"
exec ./venv/bin/uvicorn backend.api:app --host 127.0.0.1 --port 8000
