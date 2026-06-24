#!/usr/bin/env bash
# Worker entrypoint for Azure Container Apps Jobs.
#
# Runs the rolling sentiment update, then regenerates the open-to-open trading
# simulation so the Trading page stays fresh. Both steps write to Postgres
# (WSB_STORAGE / SIM_STORAGE = db); the simulation's JSON side-output goes to a
# throwaway dir since the API reads it from the database in production.
#
# The update_sentiment.py arguments are selected by WSB_JOB_MODE:
#   incremental (default) — the 15-minute cadence
#   nightly                — the deeper nightly score/comment refresh
# Passing explicit CLI args overrides the mode entirely.
set -euo pipefail

INCREMENTAL_ARGS=(--storage db --scrape-days 1 --window-days 28 --aggregate-days 14
  --top-window-days 1,3,7 --request-delay 0.05 --overlap-minutes 30
  --score-refresh-days 3 --max-score-refresh 100
  --comment-refresh-days 3 --max-comment-refresh-posts 40)

NIGHTLY_ARGS=(--storage db --scrape-days 1 --window-days 28 --aggregate-days 14
  --top-window-days 1,3,7 --request-delay 0.05 --overlap-minutes 60
  --score-refresh-days 14 --max-score-refresh 0
  --comment-refresh-days 14 --max-comment-refresh-posts 0)

if [ "$#" -gt 0 ]; then
  UPDATE_ARGS=("$@")
elif [ "${WSB_JOB_MODE:-incremental}" = "nightly" ]; then
  UPDATE_ARGS=("${NIGHTLY_ARGS[@]}")
else
  UPDATE_ARGS=("${INCREMENTAL_ARGS[@]}")
fi

echo "[worker] update_sentiment.py ${UPDATE_ARGS[*]}"
python update_sentiment.py "${UPDATE_ARGS[@]}"

echo "[worker] portfolio.py simulation"
# The simulation needs at least two market-open days of accumulated daily
# sentiment history. On a freshly seeded database that isn't available yet, so
# treat a portfolio failure as non-fatal: the sentiment update above is the
# critical work, and the simulation starts succeeding once history builds up.
if python portfolio.py \
  --storage "${SIM_STORAGE:-db}" \
  --output-dir "${SIM_OUTPUT_DIR:-/tmp/portfolio_data}" \
  --initial-capital "${SIM_INITIAL_CAPITAL:-1000000}" \
  --window-days "${SIM_WINDOW_DAYS:-7}" \
  --max-positions "${SIM_MAX_POSITIONS:-25}"; then
  echo "[worker] portfolio simulation updated"
else
  echo "[worker] portfolio simulation skipped (insufficient history yet); continuing"
fi

echo "[worker] done"
