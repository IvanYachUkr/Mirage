"""Run all 113 original JOB queries against the synthetic IMDB database.

Reports: success/error count, rows returned, execution time, and any SQL errors.
"""
import sys, io, os, time, re, csv
from pathlib import Path

import psycopg2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB = dict(host="localhost", port=5433, dbname="imdb_benchmark", user="postgres", password="bench")
QUERY_DIR = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\test19\job_adapted")

conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()

# Natural sort for 1a, 1b, ..., 2a, ..., 10a, ...
def _sort_key(p):
    m = re.match(r"(\d+)([a-z]?)", p.stem)
    return (int(m.group(1)), m.group(2)) if m else (999, p.stem)

sql_files = sorted(QUERY_DIR.glob("*.sql"), key=_sort_key)
print(f"Found {len(sql_files)} JOB queries in {QUERY_DIR}")
print(f"Database: {DB['dbname']} @ {DB['host']}:{DB['port']}")
print()
print(f"  {'Query':<8} {'Status':<8} {'Rows':>8} {'ms':>8}  Note")
print(f"  {'-'*60}")

results = []
ok = err = empty = 0

for sf in sql_files:
    qid = sf.stem
    sql = sf.read_text(encoding="utf-8").strip().rstrip(";")
    
    # Wrap in SELECT COUNT(*) FROM (...) to get row count without transferring data
    count_sql = f"SELECT COUNT(*) FROM ({sql}) AS _q"
    
    t0 = time.time()
    try:
        cur.execute(count_sql)
        row_count = cur.fetchone()[0]
        elapsed = (time.time() - t0) * 1000
        
        status = "OK" if row_count > 0 else "EMPTY"
        note = ""
        if row_count == 0:
            empty += 1
            note = "(no matching rows)"
        ok += 1
        
        print(f"  {qid:<8} {status:<8} {row_count:>8,} {elapsed:>7.0f}  {note}", flush=True)
        results.append((qid, status, row_count, round(elapsed, 1), note))
        
    except Exception as ex:
        elapsed = (time.time() - t0) * 1000
        err += 1
        err_msg = str(ex).split("\n")[0][:80]
        print(f"  {qid:<8} {'ERROR':<8} {'—':>8} {elapsed:>7.0f}  {err_msg}", flush=True)
        results.append((qid, "ERROR", 0, round(elapsed, 1), err_msg))
        # Reset connection after error
        conn.rollback()

print(f"\n{'='*60}")
print(f"SUMMARY: {len(sql_files)} queries")
print(f"{'='*60}")
print(f"  OK (rows > 0):   {ok - empty}")
print(f"  OK (empty):      {empty}")
print(f"  ERRORS:          {err}")
print(f"  Total OK:        {ok}/{len(sql_files)}")

if results:
    ok_times = [r[3] for r in results if r[1] != "ERROR"]
    if ok_times:
        print(f"\n  Timing (successful queries):")
        print(f"    Min:    {min(ok_times):>8.1f} ms")
        print(f"    Max:    {max(ok_times):>8.1f} ms")
        print(f"    Median: {sorted(ok_times)[len(ok_times)//2]:>8.1f} ms")
        print(f"    Total:  {sum(ok_times):>8.1f} ms")

# Save CSV
out_csv = Path(__file__).parent / "job_original_results.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["query", "status", "row_count", "ms", "note"])
    for r in results:
        w.writerow(r)
print(f"\n  CSV -> {out_csv.name}")

cur.close()
conn.close()
