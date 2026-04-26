#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${1:-}"

if [[ -z "$DEST_DIR" ]]; then
  echo "Usage: $(basename "$0") /absolute/or/relative/new_run_dir" >&2
  exit 2
fi

mkdir -p "$DEST_DIR"
DEST_DIR="$(cd "$DEST_DIR" && pwd)"

if [[ "$DEST_DIR" == "$SRC_DIR" ]]; then
  echo "Destination must be different from source." >&2
  exit 2
fi

rsync -a \
  --exclude '.git/' \
  --exclude '.venv*/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.cache/' \
  --exclude 'entities/' \
  --exclude 'graph/' \
  --exclude 'checkpoints/' \
  --exclude 'critic/' \
  --exclude 'imdb_schema/' \
  --exclude 'decision_logs/' \
  --exclude 'reports/' \
  --exclude '_step100_resume/' \
  --exclude '_dev/' \
  --exclude '*.arrow' \
  --exclude '*.duckdb' \
  --exclude 'movie.csv' \
  --exclude 'movies_flat.csv' \
  --exclude 'movies_analysis.csv' \
  --exclude 'cast_info.csv' \
  --exclude 'movie_directors.csv' \
  --exclude 'movie_companies.csv' \
  --exclude 'movie_keyword.csv' \
  --exclude 'movie_crew.csv' \
  --exclude 'release_dates.csv' \
  --exclude 'box_office_*.csv' \
  --exclude 'reviews.csv' \
  --exclude 'awards.csv' \
  --exclude 'locations.csv' \
  --exclude 'alternate_titles.csv' \
  --exclude 'ratings_breakdown.csv' \
  --exclude 'movie_links.csv' \
  --exclude 'person_demographics.csv' \
  --exclude 'tv_series.csv' \
  --exclude 'seasons.csv' \
  --exclude 'episodes.csv' \
  --exclude 'episode_cast.csv' \
  --exclude 'company_links.csv' \
  --exclude 'user_ratings.csv' \
  --exclude 'production_timeline.csv' \
  --exclude 'media_links.csv' \
  --exclude 'person_contracts.csv' \
  --exclude 'world_events.csv' \
  --exclude 'persons_enriched.csv' \
  --exclude 'companies_enriched.csv' \
  --exclude 'pipeline_timing.jsonl' \
  --exclude 'movie_generation_progress.jsonl' \
  --exclude 'llm_usage.jsonl' \
  --exclude 'research_mode_audit.json' \
  --exclude '.pipeline_checkpoint.json' \
  "$SRC_DIR/" "$DEST_DIR/"

chmod +x "$DEST_DIR"/*.sh 2>/dev/null || true
echo "Created code-only run directory: $DEST_DIR"
echo "Next:"
echo "  cd $DEST_DIR"
echo "  ./setup_linux_env.sh"
echo "  RUN_PROFILE=tiny10_1y ./run_local_ollama_test.sh check-provider"
