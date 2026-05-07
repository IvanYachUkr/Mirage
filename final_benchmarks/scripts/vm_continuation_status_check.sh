#!/usr/bin/env bash
set -u

cd /home/vanya/DATA_SYS_LAB/test42 || exit 2

echo '===PROCESS==='
ps -eo pid,ppid,etime,pcpu,pmem,cmd | grep -E 'generate_movies|run_pipeline' | grep -v grep || true

echo '===MANIFEST==='
.venv-linux/bin/python - <<'PY'
import json
from pathlib import Path

p = Path("_step100_resume/manifest.json")
if not p.exists():
    print("missing")
else:
    m = json.loads(p.read_text())
    keys = [
        "status",
        "movie_count",
        "produced_movie_count",
        "last_completed_year",
        "last_completed_sequence_index",
        "updated_at",
    ]
    print(json.dumps({k: m.get(k) for k in keys}, indent=2, sort_keys=True))
PY

echo '===SHARDS==='
ls -lh _step100_resume/shards/movie/year=2051.arrow \
       _step100_resume/shards/movie/year=2052.arrow \
       _step100_resume/shards/movie/year=2053.arrow \
       _step100_resume/shards/movie/year=2054.arrow 2>/dev/null || true

echo '===LOGTAIL==='
tail -n 80 _runner_logs/continuation200k_2051_2075/run.log 2>/dev/null || true

echo '===MEM==='
free -h

echo '===DISK==='
df -h /home/vanya/DATA_SYS_LAB/test42
