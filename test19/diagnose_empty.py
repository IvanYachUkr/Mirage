"""Diagnose why each empty adapted JOB query returns 0 rows.

For each empty query, test sub-conditions to find which predicate combination
causes zero results, then suggest actual DB values that would produce rows.
"""
import psycopg2, re, sys, io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB = dict(host="localhost", port=5433, dbname="imdb_benchmark", user="postgres", password="bench")
ADAPTED = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\test19\job_adapted")

conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()

EMPTY = [
    "2b","2c","3b","5a","5b","5c","6b","6c","6d","6e",
    "7a","7b","7c","8a","8b","9a","9b",
    "11a","11b","11c","11d","12b","13b","13c",
    "14b","15a","15b","15c","15d",
    "16a","16c","16d","17d",
    "18a","18b","18c","19a","19b","20b",
    "21a","21b","21c","22b",
    "23a","23b","23c","24a","24b","25b",
    "27a","27b","27c","28b",
    "29a","29b","29c","30b",
    "31a","31b","31c","33a","33b","33c",
]

def count_q(sql):
    """Run COUNT(*) on a query's FROM+WHERE."""
    sql = sql.strip().rstrip(";")
    m = re.search(r'\bFROM\b', sql, re.IGNORECASE)
    if m:
        count_sql = "SELECT COUNT(*) " + sql[m.start():]
    else:
        count_sql = f"SELECT COUNT(*) FROM ({sql}) _q"
    try:
        cur.execute(count_sql)
        return cur.fetchone()[0]
    except:
        conn.rollback()
        return -1

for qid in EMPTY:
    sql = (ADAPTED / f"{qid}.sql").read_text(encoding="utf-8").strip().rstrip(";")
    cnt = count_q(sql)
    
    # Extract WHERE predicates
    m_where = re.search(r'\bWHERE\b', sql, re.IGNORECASE)
    if not m_where:
        continue
    
    select_from = sql[:m_where.start()]
    where_clause = sql[m_where.start()+5:].strip()
    
    # Split into individual AND conditions
    # Simple split on "\n  AND " pattern
    conditions = re.split(r'\n\s+AND\s+', where_clause)
    
    # Group conditions: join conditions (have = with table aliases on both sides) vs filter conditions
    join_conds = []
    filter_conds = []
    for c in conditions:
        c = c.strip()
        # Check if it's a join condition (two table.column references connected by =)
        if re.match(r'^[\w]+\.[\w]+ = [\w]+\.[\w]+$', c):
            join_conds.append(c)
        else:
            filter_conds.append(c)
    
    print(f"\n{'='*80}")
    print(f"Query {qid}: {cnt} rows | {len(filter_conds)} filters, {len(join_conds)} joins")
    print(f"{'='*80}")
    
    # Test each filter by removing it
    if cnt == 0 and len(filter_conds) > 1:
        for i, fc in enumerate(filter_conds):
            # Build query without this filter
            remaining = filter_conds[:i] + filter_conds[i+1:]
            new_where = " AND ".join(remaining + join_conds)
            test_sql = f"{select_from} WHERE {new_where}"
            test_cnt = count_q(test_sql)
            marker = " <<<< BLOCKING" if test_cnt > 0 else ""
            if test_cnt != 0:
                print(f"  Remove [{fc[:70]}]: {test_cnt} rows{marker}")
    
    # Print all filter conditions
    print(f"  Filters:")
    for fc in filter_conds:
        print(f"    - {fc[:100]}")

cur.close()
conn.close()
