"""Fix all 63 empty adapted JOB queries with data-driven predicate replacements.

This script reads each empty query from job_adapted/, applies targeted fixes
based on actual database values, and overwrites the file.

Key fixes needed:
1. mc.note IS NULL -> mc.note IS NOT NULL (all notes have year values)
2. mi.note like '%internet%' -> mi.note like '%Streaming%' 
3. mc.note like '%(worldwide)%' -> mc.note like '%(201%)%'
4. mc.note like '%(USA)%' / '%(France)%' / '%(Japan)%' -> mc.note like '%(201%)%'
5. lt.link LIKE '%follow%' -> lt.link = 'spin off' or lt.link = 'unknown link'
6. ci.note = '(producer)' -> ci.note = 'producer' (no parens in our data)
7. ci.note = '(voice: English version)' -> ci.note = '(voice)' 
8. ci.note = '(voice: Japanese version)' -> ci.note = '(voice)'
9. cn.name = 'YouTube' / 'DreamWorks Animation' / 'Lionsgate%' -> actual company names
10. t.title like '%Kung Fu Panda%' / '%Shrek%' etc -> actual titles
11. t.title like '%Freddy%' / '%Jason%' / 'Saw%' -> actual titles
12. t.title like '%Vampire%' -> actual title patterns
13. Year ranges too old -> fit our 1995-2025 range
14. episode_nr ranges 50-100 -> 5-24
15. n.name_pcode_cf patterns -> use actual codes
16. chn.name = 'Queen' -> actual char name
17. country_code '[nl]' / '[sm]' / '[ru]' -> existing codes
18. pi.note / bio patterns
19. mc.note like '%(Blu-ray)%' -> mc.note like '%(201%)%'
20. t.production_year = 1998 -> 2018
"""
import re
from pathlib import Path

ADAPTED = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\test19\job_adapted")

# Dict: query_id -> list of (old, new) string replacements
# Applied in order via str.replace()
FIXES = {}

# ── Query 2b: cn.country_code='[nl]', k.keyword='character-name-in-title'
# Both block. [nl] doesn't exist. character-name-in-title has only 22 movies.
# The combo of [nl] + that keyword is empty.
# Fix: [nl] -> [in] (667 companies), keep keyword but it was already changed in adapt
# Actually q2b was already adapted but the ORIGINAL script missed some subs.
# Let me re-read each query to see what's actually in the adapted version.

def read_q(qid):
    return (ADAPTED / f"{qid}.sql").read_text(encoding="utf-8")

def write_q(qid, sql):
    (ADAPTED / f"{qid}.sql").write_text(sql, encoding="utf-8")

# ══════════════════════════════════════════════════════════════
# SYSTEMATIC FIXES - Go through each empty query
# ══════════════════════════════════════════════════════════════

# --- 2b: Has cn.country_code='[nl]' which doesn't exist ---
# The original 11b.sql had lt.link LIKE '%follows%' and 
# t.production_year = 1998, t.title like '%Money%'
# The adapted version still has some unmatched predicates
FIXES["2b"] = [
    ("lt.link LIKE '%follows%'", "lt.link = 'spin off'"),
    ("lt.link LIKE '%follow%'", "lt.link = 'spin off'"),
    ("mc.note IS NULL", "mc.note IS NOT NULL"),
    ("t.production_year = 1998", "t.production_year = 2018"),
    ("t.production_year = 2018", "t.production_year = 2018"),  # keep
    ("t.title like '%Money%'", "t.title like '%Broken%'"),
]

# --- 2c: Has cn.country_code='[sm]' which doesn't exist, and old company names ---
FIXES["2c"] = [
    ("'20th Century Fox%'", "'Apex%'"),
    ("'Twentieth Century Fox%'", "'Mainframe%'"),
    ("'sequel'", "'murder'"),
    ("'revenge'", "'violence'"),
    ("'based-on-novel'", "'blood'"),
    ("mc.note IS NOT NULL", "mc.note IS NOT NULL"),
    ("cn.country_code !='[sm]'", "cn.country_code !='[ng]'"),
]

# --- 3b: Has k.keyword like '%sequel%', mi.info IN ('Bulgaria') ---
FIXES["3b"] = [
    ("'Birdemic%'", "'Broken%'"),
    ("'%Movie%'", "'%Storm%'"),
    ("mi.info IN ('Bulgaria')", "mi.info IN ('India')"),
    ("k.keyword like '%sequel%'", "k.keyword like '%murder%'"),
]

# --- 5a-5c: mc.note patterns that don't exist ---
# These are queries from family 15 (aka_title queries) that have mc.note 
# patterns like %(theatrical)%, %(France)%, %(VHS)%, %(USA)%
FIXES["5a"] = [
    ("mc.note like '%(theatrical)%'", "mc.note like '%(201%)%'"),
    ("mc.note like '%(France)%'", "mc.note like '%(201%)%'"),
    ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Denish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
     "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
]
FIXES["5b"] = [
    ("mc.note like '%(VHS)%'", "mc.note like '%(201%)%'"),
    ("mc.note like '%(USA)%'", "mc.note like '%(201%)%'"),
    ("mc.note like '%(1994)%'", "mc.note like '%(2014)%'"),
    ("mi.info IN ('USA',\n                  'America')",
     "mi.info IN ('USA',\n                  'UK')"),
]
FIXES["5c"] = [
    ("mc.note like '%(USA)%'", "mc.note like '%(201%)%'"),
    ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Denish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
     "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
    ("t.production_year > 1990", "t.production_year > 2005"),
]

# --- 6b-6e: n.name patterns that don't match ---
FIXES["6b"] = [
    ("n.name LIKE '%Downey%Robert%'", "n.name LIKE '%Tina%Feyen%'"),
    ("k.keyword in ('superhero',", "k.keyword in ('murder',"),
    ("'sequel',", "'fight',"),
    ("'second-part',", "'violence',"),
    ("'marvel-comics',", "'blood',"),
    ("'based-on-comic',", "'twist-ending',"),
    ("'tv-special',", "'tv-special',"),
    ("t.production_year > 2014", "t.production_year > 2010"),
]
FIXES["6c"] = [
    ("n.name LIKE '%Storm%Steel%'", "n.name LIKE '%Sofia%Uribe%'"),
    ("t.production_year > 2014", "t.production_year > 2005"),
]
FIXES["6d"] = [
    ("n.name LIKE '%Downey%Robert%'", "n.name LIKE '%Tina%Feyen%'"),
    ("k.keyword in ('superhero',", "k.keyword in ('murder',"),
    ("'sequel',", "'fight',"),
    ("'second-part',", "'violence',"),
    ("'marvel-comics',", "'blood',"),
    ("'based-on-comic',", "'twist-ending',"),
    ("'tv-special',", "'tv-special',"),
]
FIXES["6e"] = [
    ("n.name LIKE '%Downey%Robert%'", "n.name LIKE '%Sofia%'"),
]

# --- 7a-7c: lt.link = 'features', year ranges, n.name_pcode_cf ---
FIXES["7a"] = [
    ("t.production_year BETWEEN 1980 AND 1995", "t.production_year BETWEEN 2005 AND 2020"),
    ("n.name_pcode_cf BETWEEN 'A' AND 'F'", "n.name_pcode_cf IS NOT NULL"),
    ("n.name LIKE 'B%'))", "n.name LIKE 'S%'))"),
]
FIXES["7b"] = [
    ("t.production_year BETWEEN 1980 AND 1984", "t.production_year BETWEEN 2010 AND 2020"),
    ("n.name_pcode_cf LIKE 'D%'", "n.name_pcode_cf IS NOT NULL"),
]
FIXES["7c"] = [
    ("t.production_year BETWEEN 1980 AND 2010", "t.production_year BETWEEN 2005 AND 2025"),
    ("n.name_pcode_cf BETWEEN 'A' AND 'F'", "n.name_pcode_cf IS NOT NULL"),
    ("lt.link in ('references',\n                  'referenced in',\n                  'features',",
     "lt.link in ('unknown link',\n                  'spin off',"),
    ("n.name LIKE 'A%'))", "n.name LIKE 'S%'))"),
]

# --- 8a-8b: ci.note patterns, mc.note patterns, company patterns ---
FIXES["8a"] = [
    ("ci.note ='(voice: English version)'", "ci.note ='(voice)'"),
    ("cn.country_code ='[jp]'", "cn.country_code ='[in]'"),
    ("mc.note like '%(Japan)%'", "mc.note like '%(201%)%'"),
    ("mc.note not like '%(USA)%'", "mc.note IS NOT NULL"),
    ("n1.name like '%Yo%'", "n1.name like '%a%'"),
    ("n1.name not like '%Yu%'", "n1.name not like '%zzz%'"),
]
FIXES["8b"] = [
    ("ci.note ='(voice: English version)'", "ci.note ='(voice)'"),
    ("cn.country_code ='[jp]'", "cn.country_code ='[in]'"),
    ("mc.note like '%(Japan)%'", "mc.note like '%(201%)%'"),
    ("mc.note not like '%(USA)%'", "mc.note IS NOT NULL"),
    ("mc.note like '%(2006)%'\n       OR mc.note like '%(2007)%'",
     "mc.note like '%(2016)%'\n       OR mc.note like '%(2017)%'"),
    ("n.name like '%Yo%'", "n.name like '%a%'"),
    ("n.name not like '%Yu%'", "n.name not like '%zzz%'"),
    ("t.production_year BETWEEN 2006 AND 2007", "t.production_year BETWEEN 2016 AND 2020"),
    ("t.title like 'One Piece%'\n       OR t.title like 'Dragon Ball Z%'",
     "t.title like 'Broken%'\n       OR t.title like 'Silent%'"),
]

# --- 9a-9b: n.name like '%Chiaki%' blocks ---
FIXES["9a"] = [
    ("n.name like '%Chiaki%'", "n.name like '%a%'"),
    ("t.production_year BETWEEN 2005 AND 2015", "t.production_year BETWEEN 2010 AND 2025"),
    ("mi.info like 'Global:%200%'", "mi.info like 'Global:%201%'"),
]
FIXES["9b"] = [
    ("n.name like '%Chiaki%'", "n.name like '%a%'"),
    ("t.production_year BETWEEN 2007 AND 2010", "t.production_year BETWEEN 2015 AND 2022"),
    ("t.title like '%Raindrops%'", "t.title like '%Broken%'"),
    ("mc.note like '%(201%)%'", "mc.note like '%(201%)%'"),
    ("mi.info like 'Global:%2017%'\n       OR mi.info like 'Global:%2018%'",
     "mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%'"),
]

# --- 11a-11d: lt.link LIKE '%follow%' and mc.note IS NULL ---
for q in ["11a","11b","11c","11d","21a","21b","21c","27a","27b","27c"]:
    FIXES.setdefault(q, [])
    FIXES[q].extend([
        ("lt.link LIKE '%follow%'", "lt.link = 'spin off'"),
        ("lt.link LIKE '%follows%'", "lt.link = 'spin off'"),
        ("mc.note IS NULL", "mc.note IS NOT NULL"),
    ])

# 11b also has specific values
FIXES["11b"].extend([
    ("t.production_year = 1998", "t.production_year = 2018"),
    ("t.title like '%Money%'", "t.title like '%Broken%'"),
])

# 11c has old company names
FIXES["11c"].extend([
    ("cn.name like '20th Century Fox%'\n       OR cn.name like 'Twentieth Century Fox%'",
     "cn.name like 'Apex%'\n       OR cn.name like 'Mainframe%'"),
    ("k.keyword in ('sequel',\n                    'revenge',\n                    'based-on-novel')",
     "k.keyword in ('murder',\n                    'violence',\n                    'blood')"),
    ("t.production_year > 1950", "t.production_year > 2000"),
])

# 11d has old keywords and company kind
FIXES["11d"].extend([
    ("k.keyword in ('sequel',\n                    'revenge',\n                    'based-on-novel')",
     "k.keyword in ('murder',\n                    'violence',\n                    'blood')"),
    ("ct.kind != 'production companies'", "ct.kind = 'production companies'"),
    ("t.production_year > 1950", "t.production_year > 2000"),
])

# 21a-c fixes
for q in ["21a","21c"]:
    FIXES[q].extend([
        ("mi.info IN ('UK',\n                  'India',\n                  'Germany',\n                  'France',",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',"),
        ("t.production_year BETWEEN 2000 AND 2020", "t.production_year BETWEEN 2005 AND 2025"),
    ])
FIXES["21b"].extend([
    ("mi.info IN ('Germany',\n                  'German')",
     "mi.info IN ('USA',\n                  'UK')"),
    ("t.production_year BETWEEN 2000 AND 2010", "t.production_year BETWEEN 2005 AND 2025"),
])

# 27a-c fixes
for q in ["27a","27b","27c"]:
    FIXES[q].extend([
        ("t.production_year BETWEEN 2000 AND 2020", "t.production_year BETWEEN 2005 AND 2025"),
    ])
FIXES["27b"].extend([
    ("t.production_year = 1998", "t.production_year = 2018"),
])
# Add country fix to 27a-c
for q in ["27a","27b"]:
    FIXES[q].extend([
        ("mi.info IN ('UK',\n                  'Germany',\n                  'English',\n                  'German')",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan')"),
    ])
FIXES["27c"].extend([
    ("mi.info IN ('UK',\n                  'India',\n                  'Germany',\n                  'France',",
     "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',"),
])

# --- 12b: it2.info='bottom 10 rank' + title LIKE ---
FIXES["12b"] = [
    ("t.title LIKE 'Birdemic%'\n       OR t.title LIKE '%Movie%'",
     "t.title LIKE 'Broken%'\n       OR t.title LIKE '%Storm%'"),
]

# --- 13b, 13c: t.title LIKE '%Champion%' / '%Loser%' ---
FIXES["13b"] = [
    ("t.title LIKE '%Champion%'\n       OR t.title LIKE '%Loser%'",
     "t.title LIKE '%Broken%'\n       OR t.title LIKE '%Silent%'"),
]
FIXES["13c"] = [
    ("t.title LIKE 'Champion%'\n       OR t.title LIKE 'Loser%'",
     "t.title LIKE 'Broken%'\n       OR t.title LIKE 'Silent%'"),
]

# --- 14b: t.title LIKE '%murder%' etc ---
FIXES["14b"] = [
    ("t.title like '%murder%'\n       OR t.title like '%Murder%'\n       OR t.title like '%Mord%'",
     "t.title like '%Broken%'\n       OR t.title like '%Storm%'\n       OR t.title like '%Silent%'"),
    ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Denish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
     "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
    ("k.keyword in ('murder',\n                    'murder-in-title')",
     "k.keyword in ('murder',\n                    'violence')"),
]

# --- 15a-15d: mi.note like '%internet%', mc.note like '%(worldwide)%' etc ---
FIXES["15a"] = [
    ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
    ("mc.note like '%(worldwide)%'", "mc.note like '%(202%)%'"),
    ("mi.note like '%internet%'", "mi.note like '%Streaming%'"),
    ("mi.info like 'USA:% 200%'", "mi.info like 'Global:%201%'"),
]
FIXES["15b"] = [
    ("cn.name = 'YouTube'", "cn.name = 'Apex-Universal Pictures'"),
    ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
    ("mc.note like '%(worldwide)%'", "mc.note like '%(202%)%'"),
    ("mi.note like '%internet%'", "mi.note like '%Streaming%'"),
    ("mi.info like 'USA:% 200%'", "mi.info like 'Global:%201%'"),
    ("t.production_year BETWEEN 2005 AND 2010", "t.production_year BETWEEN 2015 AND 2025"),
]
FIXES["15c"] = [
    ("mi.note like '%internet%'", "mi.note like '%Streaming%'"),
    ("mi.info like 'USA:% 199%'\n       OR mi.info like 'USA:% 200%'",
     "mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%'"),
    ("t.production_year > 1990", "t.production_year > 2005"),
]
FIXES["15d"] = [
    ("mi.note like '%internet%'", "mi.note like '%Streaming%'"),
    ("t.production_year > 1990", "t.production_year > 2005"),
]

# --- 16a: k.keyword + episode_nr ranges ---
FIXES["16a"] = [
    ("k.keyword ='character-name-in-title'", "k.keyword ='murder'"),
    ("t.episode_nr >= 50", "t.episode_nr >= 5"),
    ("t.episode_nr < 100", "t.episode_nr < 24"),
]
FIXES["16c"] = [
    ("t.episode_nr < 100", "t.episode_nr < 24"),
]
FIXES["16d"] = [
    ("t.episode_nr >= 5", "t.episode_nr >= 1"),
    ("t.episode_nr < 100", "t.episode_nr < 24"),
]

# --- 17d: k.keyword='character-name-in-title' + n.name LIKE '%Bert%' ---
FIXES["17d"] = [
    ("k.keyword ='character-name-in-title'", "k.keyword ='murder'"),
    ("n.name LIKE '%Bert%'", "n.name LIKE '%a%'"),
]

# --- 18a: ci.note with old parens format ---
FIXES["18a"] = [
    ("ci.note in ('(producer)',\n                  '(executive producer)')",
     "ci.note in ('producer',\n                  '(voice)')"),
    ("n.name like '%Tim%'", "n.name like '%Storm%'"),
]

# --- 18b: ci.note old format ---
FIXES["18b"] = [
    ("ci.note in ('(writer)',\n                  '(head writer)',\n                  '(written by)',\n                  '(story)',\n                  '(story editor)')",
     "ci.note in ('writer',\n                  'producer',\n                  '(voice)',\n                  '(uncredited)')"),
    ("mi_idx.info > '8.0'", "mi_idx.info > '6.0'"),
    ("t.production_year BETWEEN 2008 AND 2014", "t.production_year BETWEEN 2005 AND 2025"),
]

# --- 18c: ci.note old format ---
FIXES["18c"] = [
    ("ci.note in ('(writer)',\n                  '(head writer)',\n                  '(written by)',\n                  '(story)',\n                  '(story editor)')",
     "ci.note in ('writer',\n                  'producer',\n                  '(voice)',\n                  '(uncredited)')"),
]

# --- 19a, 19b: ci.note patterns, n.name + mi.info patterns ---
FIXES["19a"] = [
    ("ci.note in ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')",
     "ci.note in ('(voice)',\n                  '(uncredited)')"),
    ("(mc.note like '%(presents)%'\n       OR mc.note like '%(presents)%')",
     "(mc.note like '%(co-production)%'\n       OR mc.note like '%(presents)%')"),
    ("(mi.info like 'Global:%200%'\n       OR mi.info like 'Global:%200%')",
     "(mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%')"),
    ("n.name like '%Chiaki%'", "n.name like '%a%'"),
    ("t.production_year BETWEEN 2005 AND 2009", "t.production_year BETWEEN 2010 AND 2025"),
]
FIXES["19b"] = [
    ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
    ("(mc.note like '%(presents)%'\n       OR mc.note like '%(presents)%')",
     "(mc.note like '%(co-production)%'\n       OR mc.note like '%(presents)%')"),
    ("(mi.info like 'Global:%2007%'\n       OR mi.info like 'Global:%2008%')",
     "(mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%')"),
    ("n.name like '%Chiaki%'", "n.name like '%a%'"),
    ("t.production_year BETWEEN 2007 AND 2008", "t.production_year BETWEEN 2015 AND 2025"),
    ("t.title like '%Raindrops%'", "t.title like '%Broken%'"),
]

# --- 20b: chn.name + n.name patterns ---
FIXES["20b"] = [
    ("chn.name like '%Shadow%Flame%'\n       OR chn.name like '%Colonel%'",
     "chn.name like '%Shadow%'\n       OR chn.name like '%Colonel%'"),
    ("n.name LIKE '%Storm%Steel%'", "n.name LIKE '%a%'"),
    ("k.keyword in ('superhero',\n                    'fight',\n                    'buddy-movie',",
     "k.keyword in ('murder',\n                    'fight',\n                    'violence',"),
]

# --- 22b: mc.note like '%(200%)%' + production_year > 2009 combo ---
FIXES["22b"] = [
    ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
    ("mc.note not like '%(co-production)%'", "mc.note IS NOT NULL"),
    ("mi.info IN ('Germany',\n                  'German',\n                  'USA',\n                  'Chinese')",
     "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan')"),
]

# --- 23a-23c: mi.note like '%internet%' ---
FIXES["23a"] = [
    ("mi.note like '%internet%'", "mi.note like '%Streaming%'"),
    ("mi.info like 'Global:% 199%'\n       OR mi.info like 'Global:% 200%'",
     "mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%'"),
]
FIXES["23b"] = [
    ("mi.note like '%internet%'", "mi.note like '%Streaming%'"),
    ("k.keyword in ('nerd',\n                    'loner',\n                    'alienation',",
     "k.keyword in ('murder',\n                    'fight',\n                    'violence',"),
    ("mi.info like 'Global:% 200%'", "mi.info like 'Global:%201%'"),
]
FIXES["23c"] = [
    ("mi.note like '%internet%'", "mi.note like '%Streaming%'"),
    ("kt.kind in ('movie',\n                  'tv movie',\n                  'video movie',",
     "kt.kind in ('movie',\n                  'tv series',\n                  'episode',"),
    ("mi.info like 'Global:% 199%'\n       OR mi.info like 'Global:% 200%'",
     "mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%'"),
]

# --- 24a, 24b: ci.note voice patterns + keywords ---
FIXES["24a"] = [
    ("ci.note in ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')",
     "ci.note in ('(voice)',\n                  '(uncredited)')"),
    ("k.keyword in ('hero',\n                    'martial-arts',\n                    'hand-to-hand-combat')",
     "k.keyword in ('murder',\n                    'fight',\n                    'violence')"),
    ("(mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%201%')",
     "(mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%')"),
]
FIXES["24b"] = [
    ("ci.note in ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')",
     "ci.note in ('(voice)',\n                  '(uncredited)')"),
    ("cn.name = 'DreamWorks Animation'", "cn.name = 'Apex-Universal Pictures'"),
    ("k.keyword in ('hero',\n                    'martial-arts',\n                    'hand-to-hand-combat',",
     "k.keyword in ('murder',\n                    'fight',\n                    'violence',"),
    ("(mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%201%')",
     "(mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%')"),
    ("t.title like 'Kung Fu Panda%'", "t.title like 'Broken%'"),
]

# --- 25b: t.title like 'Vampire%' ---
FIXES["25b"] = [
    ("t.title like 'Vampire%'", "t.title like 'Broken%'"),
    ("k.keyword in ('murder',\n                    'blood',\n                    'gore',",
     "k.keyword in ('murder',\n                    'blood',\n                    'violence',"),
]

# --- 28b: multiple blockers ---
FIXES["28b"] = [
    ("mc.note not like '%(co-production)%'", "mc.note IS NOT NULL"),
    ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
    ("mi.info IN ('UK',\n                  'Germany',\n                  'English',\n                  'German')",
     "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan')"),
]

# --- 29a-29c: very specific (chn.name='Queen', t.title='Shrek 2', k='computer-animation') ---
FIXES["29a"] = [
    ("chn.name = 'Queen'", "chn.name LIKE '%Shadow%'"),
    ("ci.note in ('(voice)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')",
     "ci.note in ('(voice)',\n                  '(uncredited)')"),
    ("k.keyword = 'computer-animation'", "k.keyword = 'murder'"),
    ("it3.info = 'trivia'", "it3.info = 'trivia'"),
    ("(mi.info like 'Global:%200%'\n       OR mi.info like 'Global:%200%')",
     "(mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%')"),
    ("t.title = 'Shrek 2'", "t.title LIKE 'Broken%'"),
    ("t.production_year BETWEEN 2000 AND 2010", "t.production_year BETWEEN 2010 AND 2025"),
]
FIXES["29b"] = [
    ("chn.name = 'Queen'", "chn.name LIKE '%Shadow%'"),
    ("ci.note in ('(voice)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')",
     "ci.note in ('(voice)',\n                  '(uncredited)')"),
    ("k.keyword = 'computer-animation'", "k.keyword = 'murder'"),
    ("it3.info = 'height'", "it3.info = 'trivia'"),
    ("mi.info like 'Global:%200%'", "mi.info like 'Global:%201%'"),
    ("t.title = 'Shrek 2'", "t.title LIKE 'Broken%'"),
    ("t.production_year BETWEEN 2000 AND 2005", "t.production_year BETWEEN 2010 AND 2025"),
]
FIXES["29c"] = [
    ("ci.note in ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')",
     "ci.note in ('(voice)',\n                  '(uncredited)')"),
    ("k.keyword = 'computer-animation'", "k.keyword = 'murder'"),
    ("it3.info = 'trivia'", "it3.info = 'trivia'"),
    ("(mi.info like 'Global:%200%'\n       OR mi.info like 'Global:%200%')",
     "(mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%')"),
    ("t.production_year BETWEEN 2000 AND 2010", "t.production_year BETWEEN 2010 AND 2025"),
]

# --- 30b: t.title like '%Freddy%' etc ---
FIXES["30b"] = [
    ("t.title like '%Freddy%'\n       OR t.title like '%Jason%'\n       OR t.title like 'Saw%'",
     "t.title like '%Broken%'\n       OR t.title like '%Storm%'\n       OR t.title like 'Silent%'"),
    ("ci.note in ('(writer)',\n                  '(uncredited)',\n                  'producer',",
     "ci.note in ('writer',\n                  '(uncredited)',\n                  'producer',"),
]

# --- 31a-31c: cn.name like 'Lionsgate%' ---
FIXES["31a"] = [
    ("cn.name like 'Lionsgate%'", "cn.name like 'Apex%'"),
    ("ci.note in ('(writer)',\n                  '(uncredited)',\n                  'producer',",
     "ci.note in ('writer',\n                  '(uncredited)',\n                  'producer',"),
]
FIXES["31b"] = [
    ("cn.name like 'Lionsgate%'", "cn.name like 'Apex%'"),
    ("ci.note in ('(writer)',\n                  '(uncredited)',\n                  'producer',",
     "ci.note in ('writer',\n                  '(uncredited)',\n                  'producer',"),
    ("mc.note like '%(Blu-ray)%'", "mc.note like '%(201%)%'"),
    ("t.title like '%Freddy%'\n       OR t.title like '%Jason%'\n       OR t.title like 'Saw%'",
     "t.title like '%Broken%'\n       OR t.title like '%Storm%'\n       OR t.title like 'Silent%'"),
]
FIXES["31c"] = [
    ("cn.name like 'Lionsgate%'", "cn.name like 'Apex%'"),
    ("ci.note in ('(writer)',\n                  '(uncredited)',\n                  'producer',",
     "ci.note in ('writer',\n                  '(uncredited)',\n                  'producer',"),
]

# --- 33a-33c: lt.link values, country codes, rating thresholds ---
FIXES["33a"] = [
    ("lt.link in ('fight',\n                  'follows',\n                  'followed by')",
     "lt.link in ('unknown link',\n                  'spin off')"),
    ("mi_idx2.info < '3.0'", "mi_idx2.info < '7.0'"),
    ("t2.production_year BETWEEN 2005 AND 2008", "t2.production_year BETWEEN 2010 AND 2025"),
]
FIXES["33b"] = [
    ("cn1.country_code = '[nl]'", "cn1.country_code = '[in]'"),
    ("lt.link LIKE '%follow%'", "lt.link = 'spin off'"),
    ("mi_idx2.info < '3.0'", "mi_idx2.info < '7.0'"),
    ("t2.production_year = 2007", "t2.production_year = 2018"),
]
FIXES["33c"] = [
    ("lt.link in ('fight',\n                  'follows',\n                  'followed by')",
     "lt.link in ('unknown link',\n                  'spin off')"),
    ("mi_idx2.info < '3.5'", "mi_idx2.info < '7.0'"),
    ("t2.production_year BETWEEN 2000 AND 2010", "t2.production_year BETWEEN 2010 AND 2025"),
]

# ══════════════════════════════════════════════════════════════
# Apply all fixes
# ══════════════════════════════════════════════════════════════
fixed = 0
for qid, subs in FIXES.items():
    sql = read_q(qid)
    for old, new in subs:
        sql = sql.replace(old, new)
    write_q(qid, sql)
    fixed += 1

print(f"Fixed {fixed} queries in {ADAPTED}")
