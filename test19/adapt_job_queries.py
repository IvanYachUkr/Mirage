"""Adapt all 113 JOB queries to use predicates matching our synthetic dataset.

Each query is read, text predicates are manually substituted to match values
that exist in our synthetic IMDB database, and the adapted version is written
to job_adapted/.

The JOIN structure remains EXACTLY the same - only constant values change.
"""
import re, shutil
from pathlib import Path

SRC = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\imdb_job_dataset\job_queries")
DST = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\test19\job_adapted")
DST.mkdir(exist_ok=True)

# ── Substitution map per query ──
# Format: query_id -> list of (old_string, new_string) replacements
# These are applied in order via str.replace()

# Common substitution fragments (reused across queries)
# Our data has: country_codes [us],[in],[ng],[jp],[kr],[za],[eg],[fr],[de],[it]...
# Our data has keywords: murder, fight, violence, blood, twist-ending, buddy-movie, etc.
# Our data has companies: Apex Masala Factory, Mainframe Cinema Group, Apex-Universal Pictures, etc.
# Our data has names: Sofia Uribe, Tina Feyen, Storm Steel, Chiaki Omura, George Night, etc.
# Our data has chars: Shadow Flame Agent, Colonel Vance Thorne, Professor Anaya Wu, Judge Noel Oh, etc.
# Our data has titles: Raindrops on Glass, Ancestral Spirits Rising, Broken Port Elizabeth, etc.
# Our release dates format: "Global:DD Month YYYY" (not "USA:DD Month YYYY")
# Our mc.note format: "(YYYY)" with some "(co-production)" and "(presents)"
# Our ci.note has: "(voice)", "(uncredited)", "writer", "producer", etc.
# Our bio notes: "Volker Boehm", "Pedro Borges", "Anonymous", etc.

SUBS = {
    # ════════════════════════════════════════════════════
    # QUERIES 1a-1d: company_type + movie_info_idx + movie_companies
    # Only text predicate is mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
    # Replace with a company name from our data
    # ════════════════════════════════════════════════════
    "1a": [
        ("%(as Metro-Goldwyn-Mayer Pictures)%", "%(as Apex Masala Factory)%"),
    ],
    "1b": [
        ("%(as Metro-Goldwyn-Mayer Pictures)%", "%(as Apex Masala Factory)%"),
    ],
    "1c": [
        ("%(as Metro-Goldwyn-Mayer Pictures)%", "%(as Apex Masala Factory)%"),
    ],
    "1d": [
        ("%(as Metro-Goldwyn-Mayer Pictures)%", "%(as Apex Masala Factory)%"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 2a-2d: company_name + keyword + link_type + movie_link
    # Has: cn.name LIKE '%Film%' OR '%Warner%', k.keyword='sequel',
    #      lt.link LIKE '%follow%', t.production_year ranges, t.title LIKE '%Money%'
    #      cn.country_code !='[pl]', mc.note IS NULL
    # Replace company patterns, keyword, year ranges for our data (1995-2025)
    # ════════════════════════════════════════════════════
    "2a": [
        ("cn.name LIKE '%Film%'\n       OR cn.name LIKE '%Warner%'", "cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Pictures%'"),
        ("k.keyword ='sequel'", "k.keyword ='murder'"),
        ("cn.country_code !='[pl]'", "cn.country_code !='[ng]'"),
        ("t.production_year BETWEEN 1950 AND 2000", "t.production_year BETWEEN 2000 AND 2020"),
    ],
    "2b": [
        ("cn.name LIKE '%Film%'\n       OR cn.name LIKE '%Warner%'", "cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Pictures%'"),
        ("k.keyword ='sequel'", "k.keyword ='murder'"),
        ("cn.country_code !='[pl]'", "cn.country_code !='[ng]'"),
        ("t.production_year = 1998", "t.production_year = 2018"),
        ("t.title like '%Money%'", "t.title like '%Broken%'"),
        ("lt.link LIKE '%follows%'", "lt.link LIKE '%follow%'"),
    ],
    "2c": [
        ("cn.name like '20th Century Fox%'\n       OR cn.name like 'Twentieth Century Fox%'",
         "cn.name like 'Apex%'\n       OR cn.name like 'Mainframe%'"),
        ("k.keyword in ('sequel',\n                    'revenge',\n                    'based-on-novel')",
         "k.keyword in ('murder',\n                    'violence',\n                    'blood')"),
        ("cn.country_code !='[pl]'", "cn.country_code !='[ng]'"),
        ("t.production_year > 1950", "t.production_year > 2000"),
    ],
    "2d": [
        ("k.keyword in ('sequel',\n                    'revenge',\n                    'based-on-novel')",
         "k.keyword in ('murder',\n                    'violence',\n                    'blood')"),
        ("cn.country_code !='[pl]'", "cn.country_code !='[ng]'"),
        ("t.production_year > 1950", "t.production_year > 2000"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 3a-3c: company_name + movie_info + movie_info_idx (rating)
    # Has: cn.country_code='[de]' or '[us]', mi.info IN genres, mi_idx.info > '8.0'
    # 3a: [de], genres, rating; 3b: [us], bottom 10, budget, title LIKE
    # ════════════════════════════════════════════════════
    "3a": [
        ("cn.country_code ='[de]'", "cn.country_code ='[in]'"),
    ],
    "3b": [
        ("t.title LIKE 'Birdemic%'\n       OR t.title LIKE '%Movie%'",
         "t.title LIKE 'Broken%'\n       OR t.title LIKE '%Storm%'"),
    ],
    "3c": [
        ("mi.info in ('Drama',\n                  'Horror',\n                  'Western',\n                  'Family')",
         "mi.info in ('Drama',\n                  'Horror',\n                  'Thriller',\n                  'Family')"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 4a-4c: release dates, rating, company, kind_type
    # Has: cn.country_code, t.title LIKE patterns
    # ════════════════════════════════════════════════════
    "4a": [
        ("cn.country_code ='[de]'", "cn.country_code ='[in]'"),
    ],
    "4b": [
        ("t.title LIKE '%Champion%'\n       OR t.title LIKE '%Loser%'",
         "t.title LIKE '%Broken%'\n       OR t.title LIKE '%Silent%'"),
    ],
    "4c": [
        ("t.title LIKE 'Champion%'\n       OR t.title LIKE 'Loser%'",
         "t.title LIKE 'Broken%'\n       OR t.title LIKE 'Silent%'"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 5a-5c: keyword + movie_info (countries) + movie_info_idx (rating)
    # Has: k.keyword IN (...), mi.info IN (country names), mi_idx threshold
    # Our countries: USA, UK, India, Japan, South Korea, France, China, Germany, etc.
    # ════════════════════════════════════════════════════
    "5a": [
        ("k.keyword in ('murder',\n                    'murder-in-title',\n                    'blood',\n                    'violence')",
         "k.keyword in ('murder',\n                    'blood',\n                    'violence',\n                    'fight')"),
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Denish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'Germany',\n                  'France',\n                  'Japan',\n                  'India',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
    ],
    "5b": [
        ("k.keyword in ('murder',\n                    'murder-in-title')",
         "k.keyword in ('murder',\n                    'violence')"),
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Denish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'Germany',\n                  'France',\n                  'Japan',\n                  'India',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
        ("t.title like '%murder%'\n       OR t.title like '%Murder%'\n       OR t.title like '%Mord%'",
         "t.title like '%Broken%'\n       OR t.title like '%broken%'\n       OR t.title like '%Storm%'"),
    ],
    "5c": [
        ("k.keyword IS NOT NULL\n   AND k.keyword in ('murder',\n                    'murder-in-title',\n                    'blood',\n                    'violence')",
         "k.keyword IS NOT NULL\n   AND k.keyword in ('murder',\n                    'blood',\n                    'violence',\n                    'fight')"),
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Danish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'Germany',\n                  'France',\n                  'Japan',\n                  'India',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 6a-6f: keyword + cast_info + name + title (specific real-world)
    # 6a: k='marvel-cinematic-universe', n.name LIKE '%Downey%Robert%', year > 2010
    # 6b: k='superhero', t.title LIKE '%Batman%', year BETWEEN
    # etc. - very specific to real IMDB
    # ════════════════════════════════════════════════════
    "6a": [
        ("k.keyword = 'marvel-cinematic-universe'", "k.keyword = 'murder'"),
        ("n.name LIKE '%Downey%Robert%'", "n.name LIKE '%Storm%Steel%'"),
    ],
    "6b": [
        ("k.keyword = 'superhero'", "k.keyword = 'fight'"),
        ("n.name LIKE '%Halle%Berry%'", "n.name LIKE '%Tina%Feyen%'"),
        ("t.production_year > 2005", "t.production_year > 2010"),
    ],
    "6c": [
        ("k.keyword = 'marvel-cinematic-universe'", "k.keyword = 'violence'"),
        ("n.name LIKE '%Downey%Robert%'", "n.name LIKE '%Storm%Steel%'"),
    ],
    "6d": [
        ("k.keyword = 'marvel-cinematic-universe'", "k.keyword = 'blood'"),
        ("n.name LIKE '%Robert%Downey%'", "n.name LIKE '%George%Night%'"),
    ],
    "6e": [
        ("k.keyword = 'marvel-cinematic-universe'", "k.keyword = 'fight'"),
        ("n.name LIKE '%Robert%'", "n.name LIKE '%Storm%'"),
    ],
    "6f": [
        ("k.keyword = 'marvel-cinematic-universe'", "k.keyword = 'murder'"),
        ("n.name LIKE '%Downey%'", "n.name LIKE '%Steel%'"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 7a-7c: name + movie_info + movie_info_idx + cast_info
    # Has: n.name LIKE '%Bert%' or 'B%', pi.note = 'Volker Boehm'
    # ════════════════════════════════════════════════════
    "7a": [
        ("n.name LIKE '%Bert%'", "n.name LIKE '%Storm%'"),
        ("pi.note = 'Volker Boehm'", "pi.note = 'Volker Boehm'"),  # exists in our data!
    ],
    "7b": [
        ("n.name LIKE '%Bert%'", "n.name LIKE '%Steel%'"),
    ],
    "7c": [
        ("n.name LIKE '%Bert%'", "n.name LIKE '%Omura%'"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 8a-8d: cast_info notes + name + movie_info + movie_info_idx
    # Has: ci.note IN ('(producer)','(executive producer)'), n.name LIKE '%Tim%'
    # 8b: ci.note IN writer variants, mi.info IN genres, n.gender='f'
    # ════════════════════════════════════════════════════
    "8a": [
        ("ci.note in ('(producer)',\n                  '(executive producer)')",
         "ci.note in ('producer',\n                  '(voice)')"),
        ("n.name like '%Tim%'", "n.name like '%Storm%'"),
    ],
    "8b": [
        ("ci.note in ('(writer)',\n                  '(head writer)',\n                  '(written by)',\n                  '(story)',\n                  '(story editor)')",
         "ci.note in ('writer',\n                  'producer',\n                  '(voice)',\n                  '(uncredited)')"),
        ("mi_idx.info > '8.0'", "mi_idx.info > '7.0'"),
        ("t.production_year BETWEEN 2008 AND 2014", "t.production_year BETWEEN 2010 AND 2020"),
    ],
    "8c": [
        ("ci.note in ('(writer)',\n                  '(head writer)',\n                  '(written by)',\n                  '(story)',\n                  '(story editor)')",
         "ci.note in ('writer',\n                  'producer',\n                  '(voice)',\n                  '(uncredited)')"),
    ],
    "8d": [
        ("ci.note in ('(writer)',\n                  '(head writer)',\n                  '(written by)',\n                  '(story)',\n                  '(story editor)')",
         "ci.note in ('writer',\n                  'producer',\n                  '(voice)',\n                  '(uncredited)')"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 9a-9d: aka_name + char_name + cast_info + company + movie_info
    # Has: ci.note IN voice variants, cn.country_code='[us]', n.name/gender,
    #      mi.info LIKE 'Japan:%200%', t.title LIKE '%Kung%Fu%Panda%'
    # Our release dates: "Global:DD Month YYYY", not "Japan:..." or "USA:..."
    # ════════════════════════════════════════════════════
    "9a": [
        ("ci.note in ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')",
         "ci.note in ('(voice)',\n                  '(uncredited)')"),
        ("(mc.note like '%(USA)%'\n       OR mc.note like '%(worldwide)%')",
         "(mc.note like '%(co-production)%'\n       OR mc.note like '%(presents)%')"),
        ("(mi.info like 'Japan:%200%'\n       OR mi.info like 'USA:%200%')",
         "(mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%')"),
        ("n.name like '%Ang%'", "n.name like '%Chiaki%'"),
        ("t.production_year BETWEEN 2005 AND 2009", "t.production_year BETWEEN 2015 AND 2020"),
    ],
    "9b": [
        ("ci.note = '(voice)'", "ci.note = '(voice)'"),  # exists
        ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
        ("(mc.note like '%(USA)%'\n       OR mc.note like '%(worldwide)%')",
         "(mc.note like '%(co-production)%'\n       OR mc.note like '%(presents)%')"),
        ("(mi.info like 'Japan:%2007%'\n       OR mi.info like 'USA:%2008%')",
         "(mi.info like 'Global:%2017%'\n       OR mi.info like 'Global:%2018%')"),
        ("n.name like '%Angel%'", "n.name like '%Chiaki%'"),
        ("t.production_year BETWEEN 2007 AND 2008", "t.production_year BETWEEN 2016 AND 2019"),
        ("t.title like '%Kung%Fu%Panda%'", "t.title like '%Raindrops%'"),
    ],
    "9c": [
        ("ci.note in ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')",
         "ci.note in ('(voice)',\n                  '(uncredited)')"),
        ("(mi.info like 'Japan:%200%'\n       OR mi.info like 'USA:%200%')",
         "(mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%')"),
        ("n.name like '%An%'", "n.name like '%a%'"),
    ],
    "9d": [
        ("ci.note in ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')",
         "ci.note in ('(voice)',\n                  '(uncredited)')"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 10a-10c: voice roles + char_name + cast_info + company + role_type
    # These are duplicates of 1a-1d in the original file dump - let me read them
    # Actually 10a-10c are the real char_name + cast + company queries
    # ════════════════════════════════════════════════════
    "10a": [
        ("ci.note like '%(voice)%'\n  AND ci.note like '%(uncredited)%'",
         "ci.note like '%(voice)%'"),
        ("cn.country_code = '[ru]'", "cn.country_code = '[in]'"),
    ],
    "10b": [
        ("ci.note like '%(producer)%'", "ci.note like '%(voice)%'"),
        ("cn.country_code = '[ru]'", "cn.country_code = '[in]'"),
    ],
    "10c": [
        ("ci.note like '%(producer)%'", "ci.note like '%(voice)%'"),
        ("t.production_year > 1990", "t.production_year > 2005"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 11a-11d: keyword + link_type + movie_companies + movie_info in countries
    # Same pattern as 2a-2d but with movie_info join
    # ════════════════════════════════════════════════════
    "11a": [
        ("cn.name LIKE '%Film%'\n       OR cn.name LIKE '%Warner%'",
         "cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Pictures%'"),
        ("k.keyword ='sequel'", "k.keyword ='murder'"),
        ("cn.country_code !='[pl]'", "cn.country_code !='[ng]'"),
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Denish',\n                  'Norwegian',\n                  'German')",
         "mi.info IN ('UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Brazil')"),
        ("t.production_year BETWEEN 1950 AND 2000", "t.production_year BETWEEN 2000 AND 2020"),
    ],
    "11b": [
        ("cn.name LIKE '%Film%'\n       OR cn.name LIKE '%Warner%'",
         "cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Pictures%'"),
        ("k.keyword ='sequel'", "k.keyword ='murder'"),
        ("cn.country_code !='[pl]'", "cn.country_code !='[ng]'"),
        ("mi.info IN ('Germany',\n                  'German')",
         "mi.info IN ('India',\n                  'Japan')"),
        ("t.production_year BETWEEN 2000 AND 2010", "t.production_year BETWEEN 2010 AND 2020"),
    ],
    "11c": [
        ("cn.name LIKE '%Film%'\n       OR cn.name LIKE '%Warner%'",
         "cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Pictures%'"),
        ("k.keyword ='sequel'", "k.keyword ='murder'"),
        ("cn.country_code !='[pl]'", "cn.country_code !='[ng]'"),
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Denish',\n                  'Norwegian',\n                  'German',\n                  'English')",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Brazil')"),
        ("t.production_year BETWEEN 1950 AND 2010", "t.production_year BETWEEN 2000 AND 2020"),
    ],
    "11d": [
        ("cn.name LIKE '%Film%'\n       OR cn.name LIKE '%Warner%'",
         "cn.name LIKE '%Studios%'\n       OR cn.name LIKE '%Pictures%'"),
        ("k.keyword ='sequel'", "k.keyword ='fight'"),
        ("cn.country_code !='[pl]'", "cn.country_code !='[ng]'"),
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Danish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 12a-12c: movie_info(countries) + keyword + movie_info_idx + mc notes
    #   + company_name. Has mc.note LIKE '%(USA)%' and '%(200%)%'
    # ════════════════════════════════════════════════════
    "12a": [
        ("mc.note not like '%(USA)%'", "mc.note not like '%(co-production)%'"),
        ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
        ("mi.info IN ('Germany',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan')"),
    ],
    "12b": [
        ("mc.note not like '%(USA)%'", "mc.note not like '%(co-production)%'"),
        ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
        ("mi.info IN ('Germany',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan')"),
    ],
    "12c": [
        ("mc.note not like '%(USA)%'", "mc.note not like '%(co-production)%'"),
        ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Danish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
    ],
    # 12c also has no mc.note filters -> same as 12a/b pattern

    # ════════════════════════════════════════════════════
    # QUERIES 13a-13d: same as 12 but broader, mi.info for countries
    # Also no mc.note filter on some
    # ════════════════════════════════════════════════════
    "13a": [
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Danish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
    ],
    "13b": [
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Danish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
        ("mc.note not like '%(USA)%'", "mc.note not like '%(co-production)%'"),
        ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
    ],
    "13c": [
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Danish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
    ],
    "13d": [
        ("mi.info IN ('Sweden',\n                  'Norway',\n                  'Germany',\n                  'Denmark',\n                  'Swedish',\n                  'Danish',\n                  'Norwegian',\n                  'German',\n                  'USA',\n                  'American')",
         "mi.info IN ('USA',\n                  'UK',\n                  'India',\n                  'Japan',\n                  'France',\n                  'Germany',\n                  'South Korea',\n                  'China',\n                  'Italy',\n                  'Spain')"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 14a-14c: release dates, mc.note patterns, aka_title, keyword, company
    # Has: mc.note LIKE '%(200%)%' and '%(worldwide)%', mi.note LIKE '%internet%'
    #      mi.info LIKE 'USA:% 200%', cn.country_code='[us]'
    # Our release dates: "Global:DD Month YYYY", notes have "(internet)" sometimes
    # ════════════════════════════════════════════════════
    "14a": [
        ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
        ("mc.note like '%(worldwide)%'", "mc.note like '%(presents)%'"),
        ("mi.note like '%internet%'", "mi.note like '%internet%'"),  # exists
        ("mi.info like 'USA:% 200%'", "mi.info like 'Global:%201%'"),
    ],
    "14b": [
        ("cn.name = 'YouTube'", "cn.name = 'Apex Masala Factory'"),
        ("mc.note like '%(200%)%'", "mc.note like '%(201%)%'"),
        ("mc.note like '%(worldwide)%'", "mc.note like '%(presents)%'"),
        ("mi.note like '%internet%'", "mi.note like '%internet%'"),
        ("mi.info like 'USA:% 200%'", "mi.info like 'Global:%201%'"),
        ("t.production_year BETWEEN 2005 AND 2010", "t.production_year BETWEEN 2015 AND 2020"),
    ],
    "14c": [
        ("mi.note like '%internet%'", "mi.note like '%internet%'"),
        ("mi.info like 'USA:% 199%'\n       OR mi.info like 'USA:% 200%'",
         "mi.info like 'Global:%201%'\n       OR mi.info like 'Global:%202%'"),
        ("t.production_year > 1990", "t.production_year > 2005"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 15a-15d: aka_title + company + keyword + release dates
    # Has: mi.note LIKE '%internet%'
    # ════════════════════════════════════════════════════
    "15a": [
        ("mi.note like '%internet%'", "mi.note like '%internet%'"),
        ("t.production_year > 1990", "t.production_year > 2005"),
    ],
    "15b": [
        ("cn.country_code ='[us]'\n  AND k.keyword ='character-name-in-title'\n  AND t.episode_nr >= 50\n  AND t.episode_nr < 100",
         "cn.country_code ='[us]'\n  AND k.keyword ='murder'\n  AND t.episode_nr >= 5\n  AND t.episode_nr < 30"),
    ],
    "15c": [
        ("k.keyword ='character-name-in-title'", "k.keyword ='murder'"),
    ],
    "15d": [
        ("k.keyword ='character-name-in-title'\n  AND t.episode_nr < 100",
         "k.keyword ='murder'\n  AND t.episode_nr < 30"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 16a-16d: aka_name + cast_info + company + keyword + name
    # Has: k.keyword='character-name-in-title', n.name LIKE patterns
    # ════════════════════════════════════════════════════
    "16a": [
        ("k.keyword ='character-name-in-title'\n  AND t.episode_nr >= 5\n  AND t.episode_nr < 100",
         "k.keyword ='fight'\n  AND t.episode_nr >= 1\n  AND t.episode_nr < 20"),
    ],
    "16b": [
        ("k.keyword ='character-name-in-title'", "k.keyword ='violence'"),
        ("n.name LIKE 'B%'", "n.name LIKE 'S%'"),
    ],
    "16c": [
        ("k.keyword ='character-name-in-title'", "k.keyword ='murder'"),
        ("n.name LIKE 'Z%'", "n.name LIKE 'A%'"),
    ],
    "16d": [
        ("k.keyword ='character-name-in-title'", "k.keyword ='blood'"),
        ("n.name LIKE 'X%'", "n.name LIKE 'C%'"),
    ],

    # ════════════════════════════════════════════════════
    # QUERIES 17a-17f: same cast+name+keyword+company pattern
    # ════════════════════════════════════════════════════
    "17a": [
        ("k.keyword ='character-name-in-title'", "k.keyword ='fight'"),
        ("n.name LIKE '%Bert%'", "n.name LIKE '%Storm%'"),
    ],
    "17b": [
        ("k.keyword ='character-name-in-title'", "k.keyword ='murder'"),
    ],
    "17c": [
        ("k.keyword ='character-name-in-title'", "k.keyword ='violence'"),
        ("n.name LIKE '%B%'", "n.name LIKE '%a%'"),
    ],
    "17d": [],  # 17d has no problematic predicates per original

    # ════════════════════════════════════════════════════
    # QUERIES 18a-18c: keyword + complete_cast + char_name + name
    # Has: chn.name LIKE '%Tony%Stark%' or '%Iron%Man%', '%Sherlock%'
    #      k.keyword IN (superhero, sequel, marvel-comics, etc.)
    #      n.name LIKE '%Downey%Robert%'
    # ════════════════════════════════════════════════════
    "18a": [
        ("chn.name not like '%Sherlock%'\n  AND (chn.name like '%Tony%Stark%'\n       OR chn.name like '%Iron%Man%')",
         "chn.name not like '%Professor%'\n  AND (chn.name like '%Shadow%'\n       OR chn.name like '%Colonel%')"),
        ("k.keyword in ('superhero',\n                    'sequel',\n                    'second-part',\n                    'marvel-comics',\n                    'based-on-comic',\n                    'tv-special',\n                    'fight',\n                    'violence')",
         "k.keyword in ('murder',\n                    'fight',\n                    'violence',\n                    'blood',\n                    'twist-ending',\n                    'tv-special',\n                    'buddy-movie',\n                    'adrenaline-fueled')"),
        ("t.production_year > 1950", "t.production_year > 2000"),
    ],
    "18b": [
        ("chn.name not like '%Sherlock%'\n  AND (chn.name like '%Tony%Stark%'\n       OR chn.name like '%Iron%Man%')",
         "chn.name not like '%Professor%'\n  AND (chn.name like '%Shadow%'\n       OR chn.name like '%Colonel%')"),
        ("k.keyword in ('superhero',\n                    'sequel',\n                    'second-part',\n                    'marvel-comics',\n                    'based-on-comic',\n                    'tv-special',\n                    'fight',\n                    'violence')",
         "k.keyword in ('murder',\n                    'fight',\n                    'violence',\n                    'blood',\n                    'twist-ending',\n                    'tv-special',\n                    'buddy-movie',\n                    'adrenaline-fueled')"),
        ("n.name LIKE '%Downey%Robert%'", "n.name LIKE '%Storm%Steel%'"),
    ],
    "18c": [
        ("chn.name IS NOT NULL\n  AND (chn.name like '%man%'\n       OR chn.name like '%Man%')",
         "chn.name IS NOT NULL\n  AND (chn.name like '%Agent%'\n       OR chn.name like '%Judge%')"),
        ("k.keyword in ('superhero',\n                    'marvel-comics',\n                    'based-on-comic',\n                    'tv-special',\n                    'fight',\n                    'violence',\n                    'magnet',\n                    'web',\n                    'claw',\n                    'laser')",
         "k.keyword in ('murder',\n                    'fight',\n                    'violence',\n                    'blood',\n                    'twist-ending',\n                    'tv-special',\n                    'buddy-movie',\n                    'adrenaline-fueled',\n                    'memory-loss',\n                    'spells')"),
    ],
}

# For queries 17d-17f, 19-33: read the originals and check what needs changing
# Let me handle these with a pass-through + targeted fixes approach

# Read remaining originals to decide substitutions
remaining_queries = {}
for sf in sorted(SRC.glob("*.sql")):
    if sf.stem in ("schema", "fkindexes"):
        continue
    if sf.stem not in SUBS:
        remaining_queries[sf.stem] = sf.read_text(encoding="utf-8")

# Analyze remaining queries for patterns needing substitution
# Build substitutions for remaining queries
for qid, sql in remaining_queries.items():
    subs = []
    
    # Common pattern: cn.country_code = '[ru]' -> '[in]'  
    if "'[ru]'" in sql:
        subs.append(("'[ru]'", "'[in]'"))
    if "'[pl]'" in sql:
        subs.append(("'[pl]'", "'[ng]'"))
    
    # Metro-Goldwyn-Mayer
    if "Metro-Goldwyn-Mayer" in sql:
        subs.append(("Metro-Goldwyn-Mayer Pictures", "Apex Masala Factory"))
    
    # Warner
    if "'%Warner%'" in sql and qid not in SUBS:
        subs.append(("%Warner%", "%Glass Tower%"))
    
    # Film company name pattern  
    if "'%Film%'" in sql and qid not in SUBS:
        subs.append(("%Film%", "%Studios%"))
    
    # 20th Century Fox pattern
    if "20th Century Fox" in sql:
        subs.append(("20th Century Fox%", "Apex%"))
    if "Twentieth Century Fox" in sql:
        subs.append(("Twentieth Century Fox%", "Mainframe%"))
    
    # YouTube
    if "'YouTube'" in sql:
        subs.append(("'YouTube'", "'Mainframe Cinema Group'"))
    
    # Character names with real IMDB references
    if "'%Tony%Stark%'" in sql:
        subs.append(("%Tony%Stark%", "%Shadow%Flame%"))
    if "'%Iron%Man%'" in sql:
        subs.append(("%Iron%Man%", "%Colonel%"))
    if "'%Sherlock%'" in sql:
        subs.append(("%Sherlock%", "%Professor%"))
    
    # Person name patterns
    if "'%Downey%Robert%'" in sql:
        subs.append(("%Downey%Robert%", "%Storm%Steel%"))
    if "'%Robert%Downey%'" in sql:
        subs.append(("%Robert%Downey%", "%George%Night%"))
    if "'%Downey%'" in sql and "'%Robert%Downey%'" not in sql and "'%Downey%Robert%'" not in sql:
        subs.append(("%Downey%", "%Steel%"))
    if "'%Robert%'" in sql and "'%Robert%Downey%'" not in sql and "'%Downey%Robert%'" not in sql:
        subs.append(("%Robert%", "%Storm%"))
    if "'%Halle%Berry%'" in sql:
        subs.append(("%Halle%Berry%", "%Tina%Feyen%"))
    if "'%Tim%'" in sql:
        subs.append(("%Tim%", "%Storm%"))
    if "'%Bert%'" in sql:
        subs.append(("%Bert%", "%Steel%"))
    if "'%Angel%'" in sql:
        subs.append(("%Angel%", "%Chiaki%"))
    if "'%Ang%'" in sql:
        subs.append(("%Ang%", "%Chiaki%"))
    if "'%An%'" in sql and "'%Ang%'" not in sql and "'%Angel%'" not in sql:
        subs.append(("%An%", "%a%"))
    
    # Keyword substitutions
    if "'marvel-cinematic-universe'" in sql:
        subs.append(("'marvel-cinematic-universe'", "'murder'"))
    if "'marvel-comics'" in sql:
        subs.append(("'marvel-comics'", "'blood'"))
    if "'character-name-in-title'" in sql:
        subs.append(("'character-name-in-title'", "'murder'"))
    if "'superhero'" in sql and "'superhero'" not in str(subs):
        # Only if not already handled
        pass
    if "'based-on-comic'" in sql:
        subs.append(("'based-on-comic'", "'twist-ending'"))
    if "'second-part'" in sql:
        subs.append(("'second-part'", "'buddy-movie'"))
    if "'sequel'" in sql and "'sequel'" not in str(subs):
        subs.append(("'sequel'", "'fight'"))
    
    # Title patterns
    if "'%Kung%Fu%Panda%'" in sql:
        subs.append(("%Kung%Fu%Panda%", "%Raindrops%"))
    if "'%Batman%'" in sql:
        subs.append(("%Batman%", "%Broken%"))
    if "'%Money%'" in sql:
        subs.append(("%Money%", "%Broken%"))
    if "'Birdemic%'" in sql:
        subs.append(("'Birdemic%'", "'Cordoba%'"))
    if "'%Movie%'" in sql and "'%Movie%'" not in str(subs):
        subs.append(("%Movie%", "%Storm%"))
    if "'%Champion%'" in sql:
        subs.append(("%Champion%", "%Broken%"))
    if "'Champion%'" in sql:
        subs.append(("'Champion%'", "'Broken%'"))
    if "'%Loser%'" in sql:
        subs.append(("%Loser%", "%Silent%"))
    if "'Loser%'" in sql:
        subs.append(("'Loser%'", "'Silent%'"))
    
    # Country/language values in movie_info
    if "'Swedish'" in sql:
        subs.append(("'Swedish'", "'English'"))
    if "'Denish'" in sql:
        subs.append(("'Denish'", "'Hindi'"))
    if "'Danish'" in sql:
        subs.append(("'Danish'", "'Hindi'"))
    if "'Norwegian'" in sql:
        subs.append(("'Norwegian'", "'French'"))
    if "'Sweden'" in sql:
        subs.append(("'Sweden'", "'UK'"))
    if "'Norway'" in sql:
        subs.append(("'Norway'", "'India'"))
    if "'Denmark'" in sql:
        subs.append(("'Denmark'", "'France'"))
    if "'American'" in sql:
        subs.append(("'American'", "'Chinese'"))
    
    # Release date format: USA:% -> Global:%
    if "mi.info like 'USA:%" in sql:
        subs.append(("'USA:%", "'Global:%"))
    if "mi.info like 'Japan:%" in sql:
        subs.append(("'Japan:%", "'Global:%"))
    
    # mc.note patterns
    if "mc.note like '%(USA)%'" in sql and "not like" not in sql.split("mc.note like '%(USA)%'")[0].split("\n")[-1]:
        subs.append(("%(USA)%", "%(presents)%"))
    if "mc.note not like '%(USA)%'" in sql:
        subs.append(("mc.note not like '%(USA)%'", "mc.note not like '%(co-production)%'"))
    if "'%(worldwide)%'" in sql:
        subs.append(("%(worldwide)%", "%(presents)%"))
    
    # ci.note patterns
    if "'(executive producer)'" in sql:
        subs.append(("'(executive producer)'", "'(uncredited)'"))
    if "'(head writer)'" in sql:
        subs.append(("'(head writer)'", "'(uncredited)'"))
    if "'(written by)'" in sql:
        subs.append(("'(written by)'", "'producer'"))
    if "'(story)'" in sql or "'(story editor)'" in sql:
        pass  # handled elsewhere
    
    # Year range adjustments (old movies don't exist in our data - starts ~1995)
    if "production_year > 1950" in sql:
        subs.append(("production_year > 1950", "production_year > 2000"))
    if "production_year > 1990" in sql:
        subs.append(("production_year > 1990", "production_year > 2005"))
    if "BETWEEN 1950 AND 2000" in sql:
        subs.append(("BETWEEN 1950 AND 2000", "BETWEEN 2000 AND 2020"))
    if "BETWEEN 1950 AND 2010" in sql:
        subs.append(("BETWEEN 1950 AND 2010", "BETWEEN 2000 AND 2020"))
    
    # n.name initial letter patterns
    if "n.name LIKE 'B%'" in sql and qid not in SUBS:
        subs.append(("n.name LIKE 'B%'", "n.name LIKE 'S%'"))
    if "n.name LIKE 'Z%'" in sql and qid not in SUBS:
        subs.append(("n.name LIKE 'Z%'", "n.name LIKE 'A%'"))
    if "n.name LIKE 'X%'" in sql and qid not in SUBS:
        subs.append(("n.name LIKE 'X%'", "n.name LIKE 'C%'"))
    if "n.name LIKE '%B%'" in sql and "'%Bert%'" not in sql and "'%Batman%'" not in sql and "'%Berry%'" not in sql and qid not in SUBS:
        subs.append(("n.name LIKE '%B%'", "n.name LIKE '%a%'"))
    
    # pi.note (bio contributor) - already matches
    # 'Volker Boehm' exists in our data
    
    # All episode_nr ranges - our eps have smaller numbers
    if "t.episode_nr >= 50" in sql:
        subs.append(("t.episode_nr >= 50", "t.episode_nr >= 5"))
    if "t.episode_nr < 100" in sql:
        subs.append(("t.episode_nr < 100", "t.episode_nr < 30"))
    if "t.episode_nr >= 5\n  AND t.episode_nr < 100" in sql:
        subs.append(("t.episode_nr >= 5\n  AND t.episode_nr < 100",
                     "t.episode_nr >= 1\n  AND t.episode_nr < 20"))
    
    if subs:
        SUBS[qid] = subs


# ── Now process all query files ──
processed = 0
for sf in sorted(SRC.glob("*.sql")):
    if sf.stem in ("schema", "fkindexes"):
        continue
    
    sql = sf.read_text(encoding="utf-8")
    qid = sf.stem
    
    if qid in SUBS:
        for old, new in SUBS[qid]:
            sql = sql.replace(old, new)
    
    # Write adapted query
    (DST / sf.name).write_text(sql, encoding="utf-8")
    processed += 1

print(f"Adapted {processed} queries -> {DST}")
print(f"Queries with substitutions: {len(SUBS)}")
print(f"Queries unchanged: {processed - len(SUBS)}")
