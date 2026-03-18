"""Second round of fixes for remaining 30 empty adapted JOB queries.

Root causes identified:
1. 2b, 2c: Country codes [nl]/[sm] don't exist
2. 6c: Name pattern too specific with violence keyword
3. 7a, 7b: lt.link='features' -> has movie_link entries but not that link type
4. 11a-c, 21a-c, 27a-c: movie_link + keyword + company combo too narrow
5. 12b: bottom 10 rank has only 10 entries, combo with title is empty
6. 14b: murder + title pattern + country combo empty  
7. 15a, 15b: mc.note matches TWO year patterns simultaneously (impossible)
8. 16a,c,d: Episodes have NO keywords at all in our data
9. 24b: too many specific filters (cn.name + t.title + keyword + voice)
10. 25b: ci.note '(writer)' should be 'writer'
11. 29a-c: too many specific filters (complete_cast + voice + char_name + keyword)
12. 31b: title + company + notes combo too narrow
13. 33a-c: movie_link used with tv series kind, needs matching
"""
from pathlib import Path

ADAPTED = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\test19\job_adapted")

def read_q(qid):
    return (ADAPTED / f"{qid}.sql").read_text(encoding="utf-8")

def write_q(qid, sql):
    (ADAPTED / f"{qid}.sql").write_text(sql, encoding="utf-8")

FIXES = {}

# ── 2b: cn.country_code='[nl]' ──
FIXES["2b"] = [
    ("cn.country_code ='[nl]'", "cn.country_code ='[us]'"),
    ("k.keyword ='character-name-in-title'", "k.keyword ='murder'"),
]

# ── 2c: cn.country_code='[sm]' + old keywords left ──
FIXES["2c"] = [
    ("cn.country_code ='[sm]'", "cn.country_code ='[in]'"),
    ("k.keyword ='character-name-in-title'", "k.keyword ='fight'"),
]

# ── 6c: n.name LIKE '%Sofia%Uribe%' with violence -> 0. Broaden name. ──
FIXES["6c"] = [
    ("n.name LIKE '%Sofia%Uribe%'", "n.name LIKE '%a%'"),
]

# ── 7a, 7b: lt.link='features' not used in movie_link. Use 'unknown link'. ──
FIXES["7a"] = [
    ("lt.link ='features'", "lt.link ='unknown link'"),
]
FIXES["7b"] = [
    ("lt.link ='features'", "lt.link ='unknown link'"),
]

# ── 11a-c: movie_link + keyword + company. The movie_link movies are few. 
# Broaden: remove specific company name filter, broaden keyword ──
FIXES["11a"] = [
    ("k.keyword ='murder'", "k.keyword IN ('murder','fight','violence','blood','death')"),
    ("(cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Pictures%')",
     "cn.name IS NOT NULL"),
]
FIXES["11b"] = [
    ("k.keyword ='murder'", "k.keyword IN ('murder','fight','violence','blood','death')"),
    ("(cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Pictures%')",
     "cn.name IS NOT NULL"),
    ("t.title like '%Broken%'", "t.title IS NOT NULL"),
    ("t.production_year = 2018", "t.production_year > 2010"),
]
FIXES["11c"] = [
    ("k.keyword in ('murder',\n                    'violence',\n                    'blood')",
     "k.keyword IN ('murder','fight','violence','blood','death')"),
    ("ct.kind != 'production companies'", "ct.kind = 'production companies'"),
    ("(cn.name like 'Apex%'\n       OR cn.name like 'Mainframe%')",
     "cn.name IS NOT NULL"),
]

# ── 12b: bottom 10 rank (10 entries) + title LIKE doesn't overlap. Drop title filter. ──
FIXES["12b"] = [
    ("(t.title LIKE 'Broken%'\n       OR t.title LIKE '%Storm%')",
     "t.title IS NOT NULL"),
]

# ── 14b: murder + title + countries. Broaden: drop title filter. ──
FIXES["14b"] = [
    ("(t.title like '%Broken%'\n       OR t.title like '%Storm%'\n       OR t.title like '%Silent%')",
     "t.title IS NOT NULL"),
]

# ── 15a, 15b: mc.note LIKE two patterns simultaneously (impossible).
# mc.note = (2019) can't match BOTH '%(201%)%' AND '%(202%)%'.
# Fix: remove one pattern. ──
FIXES["15a"] = [
    ("mc.note like '%(202%)%'\n  AND mi.note like '%Streaming%'",
     "mi.note like '%Streaming%'"),
]
FIXES["15b"] = [
    ("mc.note like '%(202%)%'\n  AND mi.note like '%Streaming%'",
     "mi.note like '%Streaming%'"),
]

# ── 16a,c,d: Episodes have NO keywords. Use kind_id to restrict to movies instead,
# or drop episode_nr filter. Since the query MUST use episode_nr (that's its structure),
# we drop the keyword filter and use the episode_nr filter as the selectivity driver. ──
# Actually the problem is deeper: title+movie_keyword join on movie_id, but episodes 
# don't have keywords. Let me look at what tables these queries use.
# 16a uses: aka_name, cast_info, company_name, keyword, movie_companies, movie_keyword, name, title
# The movie_keyword join is the problem. Episodes don't have keywords.
# Fix: drop episode_nr filter and use production_year instead for selectivity.
FIXES["16a"] = [
    ("t.episode_nr >= 5\n  AND t.episode_nr < 24", "t.production_year BETWEEN 2020 AND 2025"),
]
FIXES["16c"] = [
    ("t.episode_nr < 24", "t.production_year > 2020"),
]
FIXES["16d"] = [
    ("t.episode_nr >= 1\n  AND t.episode_nr < 24", "t.production_year BETWEEN 2018 AND 2025"),
]

# ── 21a-c: movie_link + company + keyword + movie_info countries. 
# Very narrow. Broaden company and keyword. ──
FIXES["21a"] = [
    ("k.keyword ='fight'", "k.keyword IN ('murder','fight','violence','blood','death')"),
    ("(cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Glass Tower%')",
     "cn.name IS NOT NULL"),
]
FIXES["21b"] = [
    ("k.keyword ='fight'", "k.keyword IN ('murder','fight','violence','blood','death')"),
    ("(cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Glass Tower%')",
     "cn.name IS NOT NULL"),
    ("mi.info IN ('USA',\n                  'UK')",
     "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan')"),
]
FIXES["21c"] = [
    ("k.keyword ='fight'", "k.keyword IN ('murder','fight','violence','blood','death')"),
    ("(cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Glass Tower%')",
     "cn.name IS NOT NULL"),
]

# ── 24b: too specific. Drop cn.name and t.title filters. ──
FIXES["24b"] = [
    ("cn.name = 'Apex-Universal Pictures'", "cn.country_code = '[us]'"),
    ("t.title like 'Broken%'", "t.title IS NOT NULL"),
]

# ── 25b: ci.note '(writer)' should be 'writer'. Also title too specific. ──
FIXES["25b"] = [
    ("ci.note in ('(writer)',", "ci.note in ('writer',"),
    ("t.title like 'Broken%'", "t.title IS NOT NULL"),
]

# ── 27a-c: movie_link + complete_cast + company + keyword + mi. Very narrow.
# Broaden: company and keyword ──
FIXES["27a"] = [
    ("k.keyword ='fight'", "k.keyword IN ('murder','fight','violence','blood','death')"),
    ("(cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Glass Tower%')",
     "cn.name IS NOT NULL"),
]
FIXES["27b"] = [
    ("k.keyword ='fight'", "k.keyword IN ('murder','fight','violence','blood','death')"),
    ("(cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Glass Tower%')",
     "cn.name IS NOT NULL"),
    ("t.production_year = 2018", "t.production_year > 2010"),
]
FIXES["27c"] = [
    ("k.keyword ='fight'", "k.keyword IN ('murder','fight','violence','blood','death')"),
    ("(cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Glass Tower%')",
     "cn.name IS NOT NULL"),
]

# ── 29a-c: complete_cast + voice + char_name + keyword + person_info + release dates.
# Way too specific. Drop char_name and title filters. ──
FIXES["29a"] = [
    ("chn.name LIKE '%Shadow%'", "chn.name IS NOT NULL"),
    ("t.title LIKE 'Broken%'", "t.title IS NOT NULL"),
]
FIXES["29b"] = [
    ("chn.name LIKE '%Shadow%'", "chn.name IS NOT NULL"),
    ("t.title LIKE 'Broken%'", "t.title IS NOT NULL"),
]
FIXES["29c"] = []  # 29c doesn't have chn.name or t.title filter, but still empty
# It has ci.note voice + person_info trivia + complete_cast + keyword.
# person_info trivia + complete_cast + voice + murder keyword on same movie is very narrow.
# Broaden keyword.
FIXES["29c"] = [
    ("k.keyword = 'murder'", "k.keyword IN ('murder','fight','violence','blood','death')"),
]
# Also fix 29a and 29b keyword
FIXES["29a"].append(("k.keyword = 'murder'", "k.keyword IN ('murder','fight','violence','blood','death')"))
FIXES["29b"].append(("k.keyword = 'murder'", "k.keyword IN ('murder','fight','violence','blood','death')"))

# ── 31b: cn.name like 'Apex%' + mc.note + t.title combo. Broaden title. ──
FIXES["31b"] = [
    ("(t.title like '%Broken%'\n       OR t.title like '%Storm%'\n       OR t.title like 'Silent%')",
     "t.title IS NOT NULL"),
]

# ── 33a-c: movie_link + mi_idx on BOTH ends. movie_link has 1553 entries.
# With tv series kind, the overlap is narrow.
# Broaden: also include 'movie' kind, or 'episode'. ──
FIXES["33a"] = [
    ("kt1.kind in ('tv series')\n  AND kt2.kind in ('tv series')",
     "kt1.kind in ('tv series','movie')\n  AND kt2.kind in ('tv series','movie')"),
    ("lt.link in ('unknown link',\n                  'spin off')",
     "lt.link IN ('unknown link','spin off')"),
]
FIXES["33b"] = [
    ("kt1.kind in ('tv series')\n  AND kt2.kind in ('tv series')",
     "kt1.kind in ('tv series','movie')\n  AND kt2.kind in ('tv series','movie')"),
    ("t2.production_year = 2018", "t2.production_year > 2010"),
]
FIXES["33c"] = [
    ("kt1.kind in ('tv series',\n                   'episode')\n  AND kt2.kind in ('tv series',\n                   'episode')",
     "kt1.kind in ('tv series','movie','episode')\n  AND kt2.kind in ('tv series','movie','episode')"),
]

# ══════════════════════════════════════════════════════════════
# Apply all fixes
# ══════════════════════════════════════════════════════════════
fixed = 0
for qid, subs in FIXES.items():
    sql = read_q(qid)
    for old, new in subs:
        if old not in sql:
            print(f"  WARNING: {qid}: pattern not found: {old[:60]}")
        sql = sql.replace(old, new)
    write_q(qid, sql)
    fixed += 1

print(f"Fixed {fixed} queries in {ADAPTED}")
