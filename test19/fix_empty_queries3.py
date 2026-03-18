"""Third round: Fix final 6 empty queries."""
from pathlib import Path

AD = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\test19\job_adapted")

fixes = {
    "11b": [
        ("k.keyword IN ('murder','fight','violence','blood','death')", "k.keyword IS NOT NULL"),
        ("t.production_year > 2010", "t.production_year > 2005"),
        ("t.title IS NOT NULL\n  AND lt.id", "lt.id"),
    ],
    "27b": [
        ("k.keyword IN ('murder','fight','violence','blood','death')", "k.keyword IS NOT NULL"),
        ("t.production_year > 2010", "t.production_year > 2005"),
        ("cct2.kind = 'complete'", "cct2.kind LIKE 'complete%'"),
    ],
    "29a": [
        ("cct2.kind ='complete+verified'", "cct2.kind LIKE 'complete%'"),
        ("k.keyword IN ('murder','fight','violence','blood','death')", "k.keyword IS NOT NULL"),
        ("t.title LIKE 'Broken%'\n  AND t.production_year", "t.production_year"),
        ("it3.info = 'trivia'", "it3.info IS NOT NULL"),
    ],
    "29b": [
        ("cct2.kind ='complete+verified'", "cct2.kind LIKE 'complete%'"),
        ("k.keyword IN ('murder','fight','violence','blood','death')", "k.keyword IS NOT NULL"),
        ("t.title LIKE 'Broken%'\n  AND t.production_year", "t.production_year"),
        ("it3.info = 'trivia'", "it3.info IS NOT NULL"),
    ],
    "29c": [
        ("cct2.kind ='complete+verified'", "cct2.kind LIKE 'complete%'"),
        ("k.keyword IN ('murder','fight','violence','blood','death')", "k.keyword IS NOT NULL"),
        ("it3.info = 'trivia'", "it3.info IS NOT NULL"),
    ],
    "31b": [
        ("cn.name like 'Apex%'", "cn.name IS NOT NULL"),
        ("mc.note like '%(201%)%'", "mc.note IS NOT NULL"),
    ],
}

for qid, subs in fixes.items():
    sql = (AD / f"{qid}.sql").read_text(encoding="utf-8")
    for old, new in subs:
        if old not in sql:
            print(f"  WARN {qid}: not found: {old[:50]}")
        sql = sql.replace(old, new)
    (AD / f"{qid}.sql").write_text(sql, encoding="utf-8")
    print(f"Fixed {qid}")
