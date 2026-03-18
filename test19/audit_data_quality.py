"""
V12 -- Exhaustive Data Quality Audit
====================================
Checks EVERY CSV, JSON, and graph file before pipeline execution.
Reports: column completeness, null rates, value distributions,
string quality, referential integrity, and WorldState compatibility.
"""
import json, os, sys, csv
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent
ENTITY_DIR = BASE_DIR / "entities"
GRAPH_DIR = BASE_DIR / "graph"

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

results = []

def check(category, name, status, detail=""):
    results.append((category, name, status, detail))
    icon = {"PASS": "+", "WARN": "!", "FAIL": "X"}[status]
    print(f"  [{icon}] {category} | {name}: {detail}" if detail else f"  [{icon}] {category} | {name}")


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_nulls(df, table_name, critical_cols, warn_cols=None):
    """Check null rates for a DataFrame."""
    for col in critical_cols:
        if col not in df.columns:
            check(table_name, f"Missing column: {col}", FAIL)
            continue
        null_count = df[col].isna().sum()
        null_rate = null_count / len(df) * 100 if len(df) > 0 else 0
        if null_count == 0:
            check(table_name, f"{col}: no nulls", PASS)
        elif null_rate < 1:
            check(table_name, f"{col}: {null_count} nulls ({null_rate:.2f}%)", WARN)
        else:
            check(table_name, f"{col}: {null_count} nulls ({null_rate:.1f}%)", FAIL)

    for col in (warn_cols or []):
        if col not in df.columns:
            check(table_name, f"Optional column missing: {col}", WARN)
            continue
        null_count = df[col].isna().sum()
        null_rate = null_count / len(df) * 100 if len(df) > 0 else 0
        if null_rate > 50:
            check(table_name, f"{col}: {null_count} nulls ({null_rate:.1f}%) - high null rate", WARN)


def check_string_quality(df, table_name, col, min_len=1, max_len=500):
    """Check string column quality: empty strings, too short, too long."""
    if col not in df.columns:
        return
    vals = df[col].dropna().astype(str)
    empty = (vals.str.strip() == "").sum()
    too_short = (vals.str.len() < min_len).sum()
    too_long = (vals.str.len() > max_len).sum()
    
    if empty > 0:
        check(table_name, f"{col}: {empty} empty strings", WARN if empty < len(df) * 0.05 else FAIL)
    
    if too_long > 0:
        max_found = vals.str.len().max()
        check(table_name, f"{col}: {too_long} strings > {max_len} chars (max: {max_found})", WARN)
    
    # Check for obvious garbage
    garbage_patterns = vals[vals.str.contains(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', regex=True, na=False)]
    if len(garbage_patterns) > 0:
        check(table_name, f"{col}: {len(garbage_patterns)} strings with control characters", FAIL,
              f"Sample: {garbage_patterns.iloc[0][:50]}")


def check_id_uniqueness(df, table_name, id_col):
    """Check ID column uniqueness and validity."""
    if id_col not in df.columns:
        check(table_name, f"Missing ID column: {id_col}", FAIL)
        return
    vals = pd.to_numeric(df[id_col], errors='coerce')
    nulls = vals.isna().sum()
    if nulls > 0:
        check(table_name, f"{id_col}: {nulls} non-numeric values", FAIL)
    zeros = (vals == 0).sum()
    if zeros > 0:
        check(table_name, f"{id_col}: {zeros} zero values", WARN)
    dupes = vals.duplicated().sum()
    if dupes > 0:
        check(table_name, f"{id_col}: {dupes} duplicates", FAIL)
    else:
        check(table_name, f"{id_col}: unique ({int(vals.min())}-{int(vals.max())})", PASS)


def check_vocab(df, table_name, col, allowed_values, is_list_col=False):
    """Check if column values are within allowed vocabulary."""
    if col not in df.columns:
        return
    if is_list_col:
        all_vals = []
        for v in df[col].dropna().astype(str):
            all_vals.extend([x.strip() for x in v.split(";") if x.strip()])
        oov = [v for v in all_vals if v not in allowed_values]
    else:
        oov = [v for v in df[col].dropna().unique() if v not in allowed_values]
    
    if oov:
        sample = oov[:5]
        check(table_name, f"{col}: {len(oov)} out-of-vocabulary values", WARN, f"Sample: {sample}")
    else:
        check(table_name, f"{col}: all values in vocabulary", PASS)


# ═══════════════════════════════════════════════════════════════════════
# AUDIT IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════

def audit_all():
    sys.path.insert(0, str(BASE_DIR))
    from contracts import (
        GENRES, CAREER_STAGES, NATIONALITIES, MARKETS,
        STYLE_TAGS, DIRECTOR_STYLES, COMPANY_TIERS,
        ARCHETYPES, EDGE_TYPES,
    )

    # ─── 1. FILE EXISTENCE ────────────────────────────────────────────
    section("1. FILE EXISTENCE CHECK")
    
    required_files = {
        "entities/person.csv": "Core entity",
        "entities/person_roles.csv": "Role assignments",
        "entities/company.csv": "Core entity",
        "entities/keyword.csv": "Core entity",
        "entities/title_bank.csv": "Title bank",
        "entities/character_bank.csv": "Character names",
        "entities/persons_latent.json": "Person latent variables",
        "entities/companies_latent.json": "Company latent variables",
        "graph/edge_graph.csv": "Relationship graph",
        "graph/communities.csv": "Community assignments",
    }
    
    for rel_path, desc in required_files.items():
        full_path = BASE_DIR / rel_path
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            check("Files", f"{rel_path} ({desc})", PASS, f"{size_kb:.0f} KB")
        else:
            check("Files", f"{rel_path} ({desc})", FAIL, "MISSING")

    # ─── 2. PERSON.CSV ────────────────────────────────────────────────
    section("2. PERSON.CSV")
    persons = pd.read_csv(ENTITY_DIR / "person.csv")
    print(f"  Rows: {len(persons)}, Columns: {list(persons.columns)}")
    
    check_id_uniqueness(persons, "person.csv", "person_id")
    check_nulls(persons, "person.csv", 
                ["person_id", "name", "nationality", "gender", "bio", "career_stage", "roles"],
                ["style_tags", "genre_affinity", "market_fit"])
    
    # String quality
    check_string_quality(persons, "person.csv", "name", min_len=3, max_len=80)
    check_string_quality(persons, "person.csv", "bio", min_len=10, max_len=1000)
    
    # Value distributions
    check_vocab(persons, "person.csv", "gender", {"M", "F", "NB"})
    check_vocab(persons, "person.csv", "career_stage", set(CAREER_STAGES))
    check_vocab(persons, "person.csv", "nationality", set(NATIONALITIES))
    check_vocab(persons, "person.csv", "genre_affinity", set(GENRES), is_list_col=True)
    check_vocab(persons, "person.csv", "style_tags", set(STYLE_TAGS) | set(DIRECTOR_STYLES), is_list_col=True)
    check_vocab(persons, "person.csv", "market_fit", set(MARKETS), is_list_col=True)
    
    # Distribution checks
    gender_dist = persons['gender'].value_counts(normalize=True)
    print(f"  Gender distribution: {dict(gender_dist.round(3))}")
    if gender_dist.get('M', 0) > 0.75 or gender_dist.get('F', 0) > 0.75:
        check("person.csv", "Gender balance", WARN, "Heavily skewed gender distribution")
    
    stage_dist = persons['career_stage'].value_counts()
    print(f"  Career stage distribution: {dict(stage_dist)}")
    
    nat_top10 = persons['nationality'].value_counts().head(10)
    print(f"  Top 10 nationalities: {dict(nat_top10)}")
    
    # Role distribution
    all_roles = []
    for r in persons['roles'].dropna().astype(str):
        all_roles.extend([x.strip() for x in r.split(";") if x.strip()])
    role_counts = Counter(all_roles)
    print(f"  Role distribution: {dict(role_counts)}")
    n_directors = role_counts.get("director", 0)
    if n_directors < 50:
        check("person.csv", f"Only {n_directors} directors", FAIL, "Need at least 50+ directors for ~5k movies")
    else:
        check("person.csv", f"{n_directors} directors", PASS)

    # ─── 3. PERSON_ROLES.CSV ─────────────────────────────────────────
    section("3. PERSON_ROLES.CSV")
    roles_df = pd.read_csv(ENTITY_DIR / "person_roles.csv")
    print(f"  Rows: {len(roles_df)}, Columns: {list(roles_df.columns)}")
    
    check_nulls(roles_df, "person_roles.csv", ["person_id", "role_type"])
    
    # Referential integrity: all person_ids exist in person.csv
    person_ids = set(persons['person_id'].astype(int))
    role_pids = set(roles_df['person_id'].astype(int))
    orphan_roles = role_pids - person_ids
    if orphan_roles:
        check("person_roles.csv", f"Orphan person_ids", FAIL, f"{len(orphan_roles)} IDs not in person.csv")
    else:
        check("person_roles.csv", "Referential integrity", PASS, "All person_ids exist in person.csv")
    
    role_type_dist = roles_df['role_type'].value_counts()
    print(f"  Role types: {dict(role_type_dist)}")

    # ─── 4. COMPANY.CSV ──────────────────────────────────────────────
    section("4. COMPANY.CSV")
    companies = pd.read_csv(ENTITY_DIR / "company.csv")
    print(f"  Rows: {len(companies)}, Columns: {list(companies.columns)}")
    
    check_id_uniqueness(companies, "company.csv", "company_id")
    check_nulls(companies, "company.csv",
                ["company_id", "name", "country", "description", "specialty_genres", "tier"],
                ["preferred_actor_styles", "preferred_director_styles", "pop_weight"])
    
    check_string_quality(companies, "company.csv", "name", min_len=3, max_len=100)
    check_string_quality(companies, "company.csv", "description", min_len=10, max_len=1000)
    check_vocab(companies, "company.csv", "tier", set(COMPANY_TIERS))
    check_vocab(companies, "company.csv", "specialty_genres", set(GENRES), is_list_col=True)
    
    tier_dist = companies['tier'].value_counts()
    print(f"  Tier distribution: {dict(tier_dist)}")
    
    country_dist = companies['country'].value_counts().head(10)
    print(f"  Top 10 countries: {dict(country_dist)}")
    
    # Pop weight distribution
    if 'pop_weight' in companies.columns:
        pw = companies['pop_weight'].dropna()
        print(f"  Pop weight: min={pw.min():.3f}, max={pw.max():.3f}, mean={pw.mean():.3f}, std={pw.std():.3f}")

    # ─── 5. KEYWORD.CSV ──────────────────────────────────────────────
    section("5. KEYWORD.CSV")
    keywords = pd.read_csv(ENTITY_DIR / "keyword.csv")
    print(f"  Rows: {len(keywords)}, Columns: {list(keywords.columns)}")
    
    check_id_uniqueness(keywords, "keyword.csv", "keyword_id")
    check_nulls(keywords, "keyword.csv", ["keyword_id", "keyword", "topic_genre"],
                ["pop_weight"])
    check_string_quality(keywords, "keyword.csv", "keyword", min_len=2, max_len=100)
    check_vocab(keywords, "keyword.csv", "topic_genre", set(GENRES))
    
    kw_dupes = keywords['keyword'].duplicated().sum()
    if kw_dupes > 0:
        check("keyword.csv", f"Duplicate keywords", WARN, f"{kw_dupes} duplicates")
    else:
        check("keyword.csv", "Keyword uniqueness", PASS)
    
    genre_dist = keywords['topic_genre'].value_counts().head(10)
    print(f"  Top 10 topic genres: {dict(genre_dist)}")

    # ─── 6. TITLE_BANK.CSV ───────────────────────────────────────────
    section("6. TITLE_BANK.CSV")
    titles = pd.read_csv(ENTITY_DIR / "title_bank.csv")
    print(f"  Rows: {len(titles)}, Columns: {list(titles.columns)}")
    
    check_nulls(titles, "title_bank.csv", ["title"], ["tagline", "genre_hint"])
    check_string_quality(titles, "title_bank.csv", "title", min_len=1, max_len=150)
    
    title_dupes = titles['title'].duplicated().sum()
    if title_dupes > 0:
        check("title_bank.csv", f"Duplicate titles", FAIL, f"{title_dupes} duplicates")
    else:
        check("title_bank.csv", "Title uniqueness", PASS)
    
    # Genre hint distribution
    if 'genre_hint' in titles.columns:
        gh_dist = titles['genre_hint'].value_counts().head(10)
        print(f"  Top 10 genre hints: {dict(gh_dist)}")
    
    # Check tagline quality
    if 'tagline' in titles.columns:
        empty_taglines = (titles['tagline'].fillna("").str.strip() == "").sum()
        tagline_rate = (1 - empty_taglines / len(titles)) * 100
        check("title_bank.csv", f"Tagline fill rate: {tagline_rate:.1f}%",
              PASS if tagline_rate > 80 else WARN)
    
    # Sample titles for quality
    print(f"  Sample titles: {list(titles['title'].head(5))}")

    # ─── 7. CHARACTER_BANK.CSV ────────────────────────────────────────
    section("7. CHARACTER_BANK.CSV")
    chars = pd.read_csv(ENTITY_DIR / "character_bank.csv")
    print(f"  Rows: {len(chars)}, Columns: {list(chars.columns)}")
    
    check_nulls(chars, "character_bank.csv", ["character_name", "archetype"])
    check_string_quality(chars, "character_bank.csv", "character_name", min_len=2, max_len=100)
    check_vocab(chars, "character_bank.csv", "archetype", set(ARCHETYPES))
    
    archetype_dist = chars['archetype'].value_counts()
    print(f"  Archetype distribution: {dict(archetype_dist)}")
    
    char_dupes = chars['character_name'].duplicated().sum()
    if char_dupes > 0:
        pct = char_dupes / len(chars) * 100
        check("character_bank.csv", f"Duplicate character names", 
              WARN if pct < 20 else FAIL, f"{char_dupes} ({pct:.1f}%)")
    else:
        check("character_bank.csv", "Character name uniqueness", PASS)

    # ─── 8. EDGE_GRAPH.CSV ───────────────────────────────────────────
    section("8. EDGE_GRAPH.CSV")
    edges = pd.read_csv(GRAPH_DIR / "edge_graph.csv")
    print(f"  Rows: {len(edges)}, Columns: {list(edges.columns)}")
    
    check_nulls(edges, "edge_graph.csv",
                ["src_id", "dst_id", "edge_type", "sign", "weight", "valid_from", "valid_to"],
                ["src_name", "dst_name", "reason", "source_kind"])
    
    # Edge type distribution
    et_dist = edges['edge_type'].value_counts()
    print(f"  Edge type distribution:")
    for et, count in et_dist.items():
        pct = count / len(edges) * 100
        print(f"    {et}: {count} ({pct:.1f}%)")
    
    # Sign distribution
    sign_dist = edges['sign'].value_counts()
    print(f"  Sign distribution: {dict(sign_dist)}")
    positive = sign_dist.get('+', 0)
    negative = sign_dist.get('-', 0)
    pos_pct = positive / len(edges) * 100
    check("edge_graph.csv", f"Sign balance: {pos_pct:.1f}% positive",
          PASS if 50 < pos_pct < 95 else WARN)
    
    # Weight distribution
    w = edges['weight'].dropna()
    print(f"  Weight: min={w.min():.3f}, max={w.max():.3f}, mean={w.mean():.3f}, std={w.std():.3f}")
    oob = ((w < 0) | (w > 1)).sum()
    if oob > 0:
        check("edge_graph.csv", f"Weight out of [0,1]", FAIL, f"{oob} edges")
    else:
        check("edge_graph.csv", "Weight range [0,1]", PASS)
    
    # Temporal validity
    vf = edges['valid_from'].dropna()
    vt = edges['valid_to'].dropna()
    print(f"  Valid_from: min={int(vf.min())}, max={int(vf.max())}, mean={vf.mean():.0f}")
    print(f"  Valid_to: min={int(vt.min())}, max={int(vt.max())}, mean={vt.mean():.0f}")
    
    invalid_periods = (edges['valid_from'] > edges['valid_to']).sum()
    if invalid_periods > 0:
        check("edge_graph.csv", f"valid_from > valid_to", FAIL, f"{invalid_periods} edges")
    else:
        check("edge_graph.csv", "Temporal validity", PASS, "All valid_from <= valid_to")
    
    # Self-loops
    self_loops = (edges['src_id'] == edges['dst_id']).sum()
    if self_loops > 0:
        check("edge_graph.csv", f"Self-loops", FAIL, f"{self_loops} edges")
    else:
        check("edge_graph.csv", "No self-loops", PASS)
    
    # Duplicate edges
    edge_keys = edges[['src_id', 'dst_id', 'edge_type']].apply(
        lambda r: (min(r['src_id'], r['dst_id']), max(r['src_id'], r['dst_id']), r['edge_type']), axis=1)
    edge_dupes = edge_keys.duplicated().sum()
    if edge_dupes > 0:
        check("edge_graph.csv", f"Duplicate edges (same src, dst, type)", WARN, f"{edge_dupes}")
    else:
        check("edge_graph.csv", "Edge uniqueness", PASS)
    
    # Referential integrity: edge IDs vs person/company IDs
    company_ids = set(companies['company_id'].astype(int))
    
    # Person-person edges
    pp_edges = edges[edges['edge_type'].isin(['collaboration', 'mentorship', 'chemistry', 'rivalry', 'avoid'])]
    pp_src_missing = set(pp_edges['src_id'].astype(int)) - person_ids
    pp_dst_missing = set(pp_edges['dst_id'].astype(int)) - person_ids
    if pp_src_missing or pp_dst_missing:
        check("edge_graph.csv", "Person-person ref integrity", FAIL,
              f"Missing src_ids: {len(pp_src_missing)}, dst_ids: {len(pp_dst_missing)}")
    else:
        check("edge_graph.csv", "Person-person ref integrity", PASS, "All IDs in person.csv")
    
    # Company-director edges
    cd_edges = edges[edges['edge_type'].isin(['preferred', 'blacklist'])]
    cd_src_missing = set(cd_edges['src_id'].astype(int)) - company_ids
    cd_dst_missing = set(cd_edges['dst_id'].astype(int)) - person_ids
    if cd_src_missing or cd_dst_missing:
        check("edge_graph.csv", "Company-director ref integrity", FAIL,
              f"Missing company_ids: {len(cd_src_missing)}, person_ids: {len(cd_dst_missing)}")
    else:
        check("edge_graph.csv", "Company-director ref integrity", PASS)
    
    # Company-company edges
    cc_edges = edges[edges['edge_type'].isin(['co_production', 'market_rival'])]
    cc_missing = (set(cc_edges['src_id'].astype(int)) | set(cc_edges['dst_id'].astype(int))) - company_ids
    if cc_missing:
        check("edge_graph.csv", "Company-company ref integrity", FAIL, f"Missing: {len(cc_missing)}")
    else:
        check("edge_graph.csv", "Company-company ref integrity", PASS)
    
    # Reason quality
    if 'reason' in edges.columns:
        empty_reasons = (edges['reason'].fillna("").str.strip() == "").sum()
        reason_rate = (1 - empty_reasons / len(edges)) * 100
        check("edge_graph.csv", f"Reason fill rate: {reason_rate:.1f}%",
              PASS if reason_rate > 90 else WARN)
        
        # Sample reasons
        sample_reasons = edges['reason'].dropna().sample(min(5, len(edges)), random_state=42).tolist()
        print(f"  Sample reasons: {sample_reasons[:3]}")
    
    # Graph connectivity: how many unique persons appear?
    pp_unique = set(pp_edges['src_id'].astype(int)) | set(pp_edges['dst_id'].astype(int))
    coverage = len(pp_unique) / len(persons) * 100
    check("edge_graph.csv", f"Person coverage: {len(pp_unique)}/{len(persons)} ({coverage:.1f}%)",
          PASS if coverage > 80 else WARN if coverage > 50 else FAIL)
    
    # Degree distribution
    degree = Counter()
    for _, r in pp_edges.iterrows():
        degree[int(r['src_id'])] += 1
        degree[int(r['dst_id'])] += 1
    if degree:
        degs = list(degree.values())
        print(f"  Person degree: min={min(degs)}, max={max(degs)}, mean={np.mean(degs):.1f}, median={np.median(degs):.0f}")
        isolates = len(person_ids - set(degree.keys()))
        check("edge_graph.csv", f"Isolated persons (no edges): {isolates}", 
              PASS if isolates < len(persons) * 0.2 else WARN)

    # ─── 9. COMMUNITIES.CSV ──────────────────────────────────────────
    section("9. COMMUNITIES.CSV")
    comms = pd.read_csv(GRAPH_DIR / "communities.csv")
    print(f"  Rows: {len(comms)}, Columns: {list(comms.columns)}")
    
    check_nulls(comms, "communities.csv", ["person_id", "community"])
    
    # Coverage
    comm_pids = set(comms['person_id'].astype(int))
    missing_from_comms = person_ids - comm_pids
    if missing_from_comms:
        pct = len(missing_from_comms) / len(persons) * 100
        check("communities.csv", f"Missing persons", WARN if pct < 5 else FAIL,
              f"{len(missing_from_comms)} persons ({pct:.1f}%) not assigned to communities")
    else:
        check("communities.csv", "Full person coverage", PASS)
    
    # Orphan check
    orphan_comms = comm_pids - person_ids
    if orphan_comms:
        check("communities.csv", f"Orphan person_ids", WARN, f"{len(orphan_comms)} IDs not in person.csv")
    
    # Distribution
    comm_dist = comms['community'].value_counts().sort_index()
    n_comms = len(comm_dist)
    print(f"  Communities: {n_comms}")
    for cid, count in comm_dist.items():
        pct = count / len(comms) * 100
        print(f"    Community {cid}: {count} ({pct:.1f}%)")
    
    if n_comms < 5 or n_comms > 30:
        check("communities.csv", f"Community count: {n_comms}", WARN, "Expected 8-14")
    else:
        check("communities.csv", f"Community count: {n_comms}", PASS)

    # ─── 10. PERSONS_LATENT.JSON ─────────────────────────────────────
    section("10. PERSONS_LATENT.JSON")
    plv = json.loads((ENTITY_DIR / "persons_latent.json").read_text(encoding="utf-8"))
    print(f"  Records: {len(plv)}")
    
    # Coverage
    plv_ids = {int(lv["person_id"]) for lv in plv}
    missing_latent = person_ids - plv_ids
    if missing_latent:
        pct = len(missing_latent) / len(persons) * 100
        check("persons_latent.json", f"Missing persons", WARN if pct < 5 else FAIL,
              f"{len(missing_latent)} persons ({pct:.1f}%) missing latent vars")
    else:
        check("persons_latent.json", "Full person coverage", PASS)
    
    # Key completeness
    expected_keys = [
        "person_id", "name", "public_reputation", "controversy_score",
        "risk_tolerance", "artistic_ambition", "creative_style_vector",
        "budget_band_pref",
    ]
    key_counts = Counter()
    for lv in plv:
        for k in expected_keys:
            if k in lv:
                key_counts[k] += 1
    
    for k in expected_keys:
        count = key_counts.get(k, 0)
        rate = count / len(plv) * 100
        if rate == 100:
            check("persons_latent.json", f"Key '{k}': 100%", PASS)
        elif rate > 90:
            check("persons_latent.json", f"Key '{k}': {rate:.1f}%", WARN)
        else:
            check("persons_latent.json", f"Key '{k}': {rate:.1f}%", FAIL)
    
    # Value range checks
    for float_key in ["public_reputation", "controversy_score", "risk_tolerance", "artistic_ambition"]:
        vals = [float(lv.get(float_key, 0)) for lv in plv if float_key in lv]
        if vals:
            print(f"  {float_key}: min={min(vals):.3f}, max={max(vals):.3f}, mean={np.mean(vals):.3f}")
            oob = sum(1 for v in vals if v < 0 or v > 1)
            if oob > 0:
                check("persons_latent.json", f"{float_key} out of [0,1]", WARN, f"{oob} values")
    
    # creative_style_vector length check
    csv_lens = [len(lv.get("creative_style_vector", [])) for lv in plv if "creative_style_vector" in lv]
    if csv_lens:
        common_len = Counter(csv_lens).most_common(1)[0]
        print(f"  creative_style_vector: most common length={common_len[0]} ({common_len[1]} records)")
        wrong_len = sum(1 for l in csv_lens if l != common_len[0])
        if wrong_len > 0:
            check("persons_latent.json", f"creative_style_vector inconsistent lengths", WARN, f"{wrong_len} records")
    
    # budget_band_pref length check
    bbp_lens = [len(lv.get("budget_band_pref", [])) for lv in plv if "budget_band_pref" in lv]
    if bbp_lens:
        common_len = Counter(bbp_lens).most_common(1)[0]
        print(f"  budget_band_pref: most common length={common_len[0]} ({common_len[1]} records)")
        if common_len[0] < 5:
            check("persons_latent.json", f"budget_band_pref too short (need 5)", FAIL)
    
    # Sample record
    print(f"  Sample record keys: {list(plv[0].keys())}")

    # ─── 11. COMPANIES_LATENT.JSON ───────────────────────────────────
    section("11. COMPANIES_LATENT.JSON")
    clv = json.loads((ENTITY_DIR / "companies_latent.json").read_text(encoding="utf-8"))
    print(f"  Records: {len(clv)}")
    
    clv_ids = {int(lv["company_id"]) for lv in clv}
    missing_c_latent = company_ids - clv_ids
    if missing_c_latent:
        pct = len(missing_c_latent) / len(companies) * 100
        check("companies_latent.json", f"Missing companies", WARN if pct < 5 else FAIL,
              f"{len(missing_c_latent)} companies ({pct:.1f}%) missing latent vars")
    else:
        check("companies_latent.json", "Full company coverage", PASS)
    
    expected_c_keys = [
        "company_id", "name", "prestige_score", "risk_appetite",
        "controversy_tolerance", "budget_tier_focus",
    ]
    key_counts = Counter()
    for lv in clv:
        for k in expected_c_keys:
            if k in lv:
                key_counts[k] += 1
    
    for k in expected_c_keys:
        count = key_counts.get(k, 0)
        rate = count / len(clv) * 100
        if rate == 100:
            check("companies_latent.json", f"Key '{k}': 100%", PASS)
        elif rate > 90:
            check("companies_latent.json", f"Key '{k}': {rate:.1f}%", WARN)
        else:
            check("companies_latent.json", f"Key '{k}': {rate:.1f}%", FAIL)
    
    for float_key in ["prestige_score", "risk_appetite", "controversy_tolerance"]:
        vals = [float(lv.get(float_key, 0)) for lv in clv if float_key in lv]
        if vals:
            print(f"  {float_key}: min={min(vals):.3f}, max={max(vals):.3f}, mean={np.mean(vals):.3f}")
    
    # budget_tier_focus length check
    btf_lens = [len(lv.get("budget_tier_focus", [])) for lv in clv if "budget_tier_focus" in lv]
    if btf_lens:
        common_len = Counter(btf_lens).most_common(1)[0]
        print(f"  budget_tier_focus: most common length={common_len[0]} ({common_len[1]} records)")
        if common_len[0] < 5:
            check("companies_latent.json", f"budget_tier_focus too short (need 5)", FAIL)
    
    print(f"  Sample record keys: {list(clv[0].keys())}")

    # ─── 12. CROSS-TABLE CONSISTENCY ─────────────────────────────────
    section("12. CROSS-TABLE CONSISTENCY")
    
    # Title count matches what generate_movies.py will use
    check("Cross-table", f"Title bank: {len(titles)} titles available for assembly", PASS)
    
    # Character bank vs expected movie count
    # ~5k movies * ~5 cast avg = ~25k characters needed
    chars_needed = len(titles) * 5
    if len(chars) < chars_needed:
        check("Cross-table", f"Character bank might be small", WARN,
              f"{len(chars)} chars for {len(titles)} movies (need ~{chars_needed})")
    else:
        check("Cross-table", f"Character bank size adequate", PASS,
              f"{len(chars)} chars for {len(titles)} movies")
    
    # Directors per movie ratio
    movies_per_dir = len(titles) / max(1, n_directors)
    check("Cross-table", f"Movies/director ratio: {movies_per_dir:.1f}",
          PASS if 5 < movies_per_dir < 30 else WARN,
          f"{len(titles)} movies / {n_directors} directors")
    
    # Actors per movie ratio
    n_actors = role_counts.get("actor", 0)
    movies_per_actor = len(titles) / max(1, n_actors)
    check("Cross-table", f"Movies/actor ratio: {movies_per_actor:.1f}",
          PASS if 0.5 < movies_per_actor < 5 else WARN,
          f"{len(titles)} movies / {n_actors} actors")

    # ─── SUMMARY ──────────────────────────────────────────────────────
    section("SUMMARY")
    passes = sum(1 for _, _, s, _ in results if s == PASS)
    warns = sum(1 for _, _, s, _ in results if s == WARN)
    fails = sum(1 for _, _, s, _ in results if s == FAIL)
    total = len(results)
    
    print(f"\n  Total checks: {total}")
    print(f"  PASS: {passes} ({passes/total*100:.0f}%)")
    print(f"  WARN: {warns} ({warns/total*100:.0f}%)")
    print(f"  FAIL: {fails} ({fails/total*100:.0f}%)")
    
    if fails > 0:
        print(f"\n  FAILURES:")
        for cat, name, status, detail in results:
            if status == FAIL:
                print(f"    [X] {cat} | {name}: {detail}")
    
    if warns > 0:
        print(f"\n  WARNINGS:")
        for cat, name, status, detail in results:
            if status == WARN:
                print(f"    [!] {cat} | {name}: {detail}")
    
    return fails, warns


if __name__ == "__main__":
    fails, warns = audit_all()
    sys.exit(1 if fails > 0 else 0)
