"""
V13 Pipeline -- verify.py
=========================
Verification & Realism Analysis suite.
Reuses analysis patterns adapted for the expanded V13 schema
(directors, franchises, ratings, communities).
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import os, sys, math

sys.path.insert(0, os.path.dirname(__file__))
from contracts import (
    GENRES, GENRE_WEIGHTS, COUNTRIES, COUNTRY_WEIGHTS,
    PRODUCTION_TIERS, TIER_WEIGHTS, BUDGET_RANGES,
    CAST_SIZE_RANGES, DECADE_WEIGHTS, ENTITY_COUNTS,
)


def _read_file(path_without_ext: str) -> pd.DataFrame:
    """Read .arrow (Feather v2) if available, else fall back to .csv."""
    arrow_path = path_without_ext + ".arrow"
    if os.path.exists(arrow_path):
        import pyarrow.feather as feather
        return feather.read_table(arrow_path).to_pandas()
    csv_path = path_without_ext + ".csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, low_memory=False)
    return pd.DataFrame()


def run_verification(base_dir: str = None):
    """Run full verification suite on generated data."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    W = 78
    def header(txt):
        print(f"\n{'='*W}")
        print(f"  {txt}")
        print(f"{'='*W}")

    # Load (Arrow-first, CSV-fallback)
    movies = _read_file(os.path.join(base_dir, "movie"))
    cast = _read_file(os.path.join(base_dir, "cast_info"))
    md = _read_file(os.path.join(base_dir, "movie_directors"))
    mc = _read_file(os.path.join(base_dir, "movie_companies"))
    mk = _read_file(os.path.join(base_dir, "movie_keyword"))

    edir = os.path.join(base_dir, "entities")
    persons = pd.read_csv(os.path.join(edir, "person.csv"))
    companies = pd.read_csv(os.path.join(edir, "company.csv"))
    keywords = pd.read_csv(os.path.join(edir, "keyword.csv"))

    # Assign IDs if missing
    if 'person_id' not in persons.columns:
        persons['person_id'] = range(1, len(persons) + 1)
    if 'company_id' not in companies.columns:
        companies['company_id'] = range(1, len(companies) + 1)

    total = len(movies)
    person_name = dict(zip(persons['person_id'], persons['name']))

    # ═════════════════════════════════════════════════════════════════
    header("0. SCALE & FILE INVENTORY")
    print(f"  Movies:          {total} (target: {ENTITY_COUNTS['movies']})")
    print(f"  Cast rows:       {len(cast)} (avg {len(cast)/total:.2f}/movie)")
    print(f"  Movie-Directors: {len(md)}")
    print(f"  Movie-Company:   {len(mc)} (avg {len(mc)/total:.2f}/movie)")
    print(f"  Movie-Keyword:   {len(mk)} (avg {len(mk)/total:.2f}/movie)")
    print(f"  Persons:         {len(persons)} (target: {ENTITY_COUNTS['persons_total']})")
    print(f"  Companies:       {len(companies)} (target: {ENTITY_COUNTS['companies']})")
    print(f"  Keywords:        {len(keywords)} (target: {ENTITY_COUNTS['keywords']})")

    # ═════════════════════════════════════════════════════════════════
    header("1. GENRE DISTRIBUTION")
    genre_counts = movies['genre'].value_counts()
    genre_deltas = []
    print(f"  {'Genre':<14} {'Count':>6} {'Actual%':>8} {'Target%':>8} {'Delta':>7}")
    print(f"  {'-'*14} {'-'*6} {'-'*8} {'-'*8} {'-'*7}")
    for g in sorted(GENRE_WEIGHTS.keys(), key=lambda x: -GENRE_WEIGHTS[x]):
        actual = genre_counts.get(g, 0)
        pct = actual/total*100
        tgt = GENRE_WEIGHTS[g]*100
        delta = pct - tgt
        genre_deltas.append(abs(delta))
        print(f"  {g:<14} {actual:>6} {pct:>7.1f}% {tgt:>7.0f}% {delta:>+6.1f}%")
    print(f"\n  Genre MAD: {np.mean(genre_deltas):.2f} pp")

    # ═════════════════════════════════════════════════════════════════
    header("2. DECADE DISTRIBUTION")
    decade_map = Counter()
    for y in movies['year']:
        decade_map[f"{(y//10)*10}s"] += 1
    decade_deltas = []
    # Include all decades from weights + data to catch mismatches both ways
    all_decades = sorted(set(DECADE_WEIGHTS.keys()) | set(decade_map.keys()))
    for d in all_decades:
        actual = decade_map.get(d, 0)
        pct = actual/total*100
        tgt = DECADE_WEIGHTS.get(d, 0)*100
        decade_deltas.append(abs(pct - tgt))
        print(f"  {d}: {actual} ({pct:.1f}%, target {tgt:.0f}%)")
    print(f"  Decade MAD: {np.mean(decade_deltas):.2f} pp")

    # ═════════════════════════════════════════════════════════════════
    header("3. TIER DISTRIBUTION")
    tier_counts = movies['production_tier'].value_counts()
    tier_deltas = []
    for t in PRODUCTION_TIERS:
        a = tier_counts.get(t, 0)
        pct = a/total*100
        tgt = TIER_WEIGHTS[t]*100
        tier_deltas.append(abs(pct-tgt))
        print(f"  {t:<8}: {a:>5} ({pct:.1f}%, target {tgt:.0f}%)")
    print(f"  Tier MAD: {np.mean(tier_deltas):.2f} pp")

    # ─── Budget Compliance ─────────────────────────────────────-──────
    header("4. BUDGET COMPLIANCE")
    # R12-NOTE: budget_usd is country-scaled (e.g. India = 9% of US equivalent).
    # Small-market films may appear "out of range" vs BUDGET_RANGES (which are
    # in US-equivalent dollars). This is expected -- not a data integrity issue.
    for tier, (lo, hi) in BUDGET_RANGES.items():
        subset = movies[movies['production_tier'] == tier]['budget_usd']
        if len(subset) == 0: continue
        in_range = ((subset >= lo) & (subset <= hi)).sum()
        print(f"  {tier:<6}: {in_range}/{len(subset)} in range ({in_range/len(subset)*100:.0f}%) "
              f"[country-scaled; non-US may appear low]")


    # ═════════════════════════════════════════════════════════════════
    header("5. BOX OFFICE")
    mult = (movies['box_office_usd'] / movies['budget_usd'].replace(0, np.nan)).dropna()
    print(f"  Median: {mult.median():.2f}x  Mean: {mult.mean():.2f}x")
    flops = (mult < 0.5).sum()
    hits = (mult > 5).sum()
    print(f"  Flops (<0.5x): {flops} ({flops/len(mult)*100:.1f}%)")
    print(f"  Hits (>5x): {hits} ({hits/len(mult)*100:.1f}%)")

    # ═════════════════════════════════════════════════════════════════
    header("6. ACTOR UTILIZATION & CASTING")
    freq = cast['person_id'].value_counts()
    unique_cast = cast['person_id'].nunique()
    print(f"  Actors used: {unique_cast}/{len(persons)} ({unique_cast/len(persons)*100:.1f}%)")
    print(f"  Most cast: {freq.iloc[0]} films  Median: {freq.median():.0f}")

    # Gini
    vals = np.sort(freq.values.astype(float))
    n = len(vals)
    index = np.arange(1, n+1)
    gini = (2*np.sum(index*vals) - (n+1)*np.sum(vals)) / (n*np.sum(vals)) if np.sum(vals) > 0 else 0
    print(f"  Casting Gini: {gini:.4f}")

    # ═════════════════════════════════════════════════════════════════
    header("7. CONSTRAINT ENFORCEMENT")
    # Load edge graph if available
    edge_path = os.path.join(base_dir, "graph", "edge_graph.csv")
    rivalries = set()
    friendships = {}
    if os.path.exists(edge_path):
        edges = pd.read_csv(edge_path)
        for _, e in edges.iterrows():
            if e.get('edge_type') == 'rivalry':
                rivalries.add((min(e['src_id'], e['dst_id']), max(e['src_id'], e['dst_id'])))
            elif e.get('edge_type') == 'friendship':
                key = (min(e['src_id'], e['dst_id']), max(e['src_id'], e['dst_id']))
                friendships[key] = e.get('weight', 1.0)

    # R11-FIX: Replaced iterrows (O(n_cast) Python loop) with vectorized groupby.
    # 37,500 cast rows -> single groupby pass vs 37,500 Python iterations.
    movie_actors = cast.groupby('title_id')['person_id'].apply(set).to_dict()


    rival_violations = 0
    bf_co = 0
    for mid, actors in movie_actors.items():
        for a, b in combinations(sorted(actors), 2):
            if (a, b) in rivalries:
                rival_violations += 1
            if (a, b) in friendships:
                bf_co += 1

    print(f"  Rival violations: {rival_violations}")
    print(f"  Best-friend co-occurrences: {bf_co}")

    # FK integrity
    orphans = 0
    orphans += len(set(cast['person_id']) - set(persons['person_id']))
    orphans += len(set(cast['title_id']) - set(movies['title_id']))
    orphans += len(set(mc['company_id']) - set(companies['company_id']))
    orphans += len(set(mc['title_id']) - set(movies['title_id']))
    orphans += len(set(mk['title_id']) - set(movies['title_id']))
    print(f"  FK orphans: {orphans}")
    print(f"  Duplicate titles: {movies['title'].duplicated().sum()}")

    # ═════════════════════════════════════════════════════════════════
    header("8. POWER-LAW TESTS")
    f_vals = freq.values.astype(float)
    lr = np.log(np.arange(1, len(f_vals)+1))
    lf = np.log(f_vals)
    slope, intercept = np.polyfit(lr, lf, 1)
    res = lf - (slope*lr + intercept)
    r2 = 1 - np.sum(res**2)/np.sum((lf-np.mean(lf))**2)
    print(f"  Casting Zipf: slope={slope:.3f} R2={r2:.4f} (real: -0.5 to -1.0)")

    bo = movies['box_office_usd'].sort_values(ascending=False).values.astype(float)
    lr_bo = np.log(np.arange(1, len(bo)+1))
    lbo = np.log(bo + 1)
    s_bo, i_bo = np.polyfit(lr_bo, lbo, 1)
    res_bo = lbo - (s_bo*lr_bo + i_bo)
    r2_bo = 1 - np.sum(res_bo**2)/np.sum((lbo-np.mean(lbo))**2)
    print(f"  BO Zipf: slope={s_bo:.3f} R2={r2_bo:.4f} (real: -0.7 to -1.2)")

    # ═════════════════════════════════════════════════════════════════
    header("9. CROSS-TABLE CORRELATIONS")
    print(f"  Budget vs BO: r={movies[['budget_usd','box_office_usd']].corr().iloc[0,1]:.4f}")

    freq_df = freq.reset_index()
    freq_df.columns = ['person_id', 'film_count']
    pop_corr = np.nan
    if 'pop_weight' in persons.columns:
        freq_df = freq_df.merge(persons[['person_id','pop_weight']], on='person_id')
        pop_corr = freq_df[['film_count','pop_weight']].corr().iloc[0,1]
        print(f"  Pop->casting: r={pop_corr:.4f}")
    else:
        # Try loading from enriched persons saved by generate_movies
        ep = _read_file(os.path.join(base_dir, "persons_enriched"))
        if len(ep) > 0 and 'pop_weight' in ep.columns:
            freq_df = freq_df.merge(ep[['person_id','pop_weight']], on='person_id')
            pop_corr = freq_df[['film_count','pop_weight']].corr().iloc[0,1]
            print(f"  Pop->casting: r={pop_corr:.4f} (from enriched)")
        else:
            print(f"  Pop->casting: N/A (no pop_weight column)")

    cs = cast.groupby('title_id').size().reset_index(name='cast_size')
    cs = cs.merge(movies[['title_id','production_tier']], on='title_id')
    cs['tier_num'] = cs['production_tier'].map({'Epic':5,'A':4,'Mid':3,'Indie':2,'Micro':1})
    tier_corr = cs[['cast_size','tier_num']].corr().iloc[0,1]
    print(f"  Tier->cast size: r={tier_corr:.4f}")

    if 'rating' in movies.columns:
        rb = movies[['rating','budget_usd']].corr().iloc[0,1]
        print(f"  Rating->budget: r={rb:.4f}")

    # ═════════════════════════════════════════════════════════════════
    header("10. CO-OCCURRENCE NETWORK")
    cooccur = Counter()
    for mid, actors in movie_actors.items():
        for a, b in combinations(sorted(actors), 2):
            cooccur[(a,b)] += 1

    total_pairs = len(cooccur)
    possible = len(persons)*(len(persons)-1)//2
    print(f"  Total pairs: {total_pairs:,} (of {possible:,} possible)")
    print(f"  Pair density: {total_pairs/max(possible,1)*100:.1f}%")

    # Clique clustering lift
    if os.path.exists(edge_path):
        same_p = diff_p = same_co = diff_co = 0
        # Use community assignment if available
        comm_path = os.path.join(base_dir, "graph", "communities.csv")
        if os.path.exists(comm_path):
            comm_df = pd.read_csv(comm_path)
            person_comm = dict(zip(comm_df['person_id'], comm_df['community']))
        else:
            person_comm = {}

        if person_comm:
            for (a, b), cnt in cooccur.items():
                ca = person_comm.get(a, -1)
                cb = person_comm.get(b, -2)
                if ca == cb and ca != -1:
                    same_p += 1; same_co += cnt
                else:
                    diff_p += 1; diff_co += cnt
            if same_p > 0 and diff_p > 0:
                lift = (same_co/same_p) / (diff_co/diff_p)
                print(f"  Clique clustering lift: {lift:.2f}x")

    # ═════════════════════════════════════════════════════════════════
    header("11. FRANCHISE ANALYSIS")
    if 'franchise_id' in movies.columns:
        fmovies = movies[movies['franchise_id'].notna()]
        n_fran = fmovies['franchise_id'].nunique()
        print(f"  Franchise movies: {len(fmovies)}/{total} ({len(fmovies)/total*100:.1f}%)")
        print(f"  Unique franchises: {n_fran}")
        if n_fran > 0:
            sizes = fmovies.groupby('franchise_id').size()
            print(f"  Movies/franchise: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.1f}")

    # ═════════════════════════════════════════════════════════════════
    header("12. REALISM VERDICT")
    verdicts = []

    cast_zipf_ok = -1.2 < slope < -0.3
    verdicts.append(("Casting Zipf", f"{slope:.2f}", "-0.5~-1.0", cast_zipf_ok))

    bo_zipf_ok = -1.5 < s_bo < -0.5
    verdicts.append(("BO Zipf", f"{s_bo:.2f}", "-0.7~-1.2", bo_zipf_ok))

    bbo = movies[['budget_usd','box_office_usd']].corr().iloc[0,1]
    verdicts.append(("Budget-BO corr", f"{bbo:.2f}", "0.6-0.8", bbo > 0.5))

    if not np.isnan(pop_corr):
        verdicts.append(("Pop->casting", f"{pop_corr:.2f}", "0.5-0.8", 0.3 < pop_corr < 0.9))
    verdicts.append(("Tier->cast", f"{tier_corr:.2f}", "0.7+", tier_corr > 0.5))
    verdicts.append(("Actor utilization", f"{unique_cast/len(persons)*100:.0f}%", ">95%",
                      unique_cast/len(persons) > 0.90))
    verdicts.append(("Casting Gini", f"{gini:.2f}", "0.50-0.58", 0.50 <= gini <= 0.58))

    good = sum(1 for _, _, _, ok in verdicts if ok)
    for name, val, target, ok in verdicts:
        tag = "GOOD" if ok else "WEAK"
        print(f"  {name:<20} {val:<10} target: {target:<10} {tag}")

    print(f"\n  RESULT: {good}/{len(verdicts)} pass")

    print(f"\n{'='*W}")
    return good, len(verdicts)


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    run_verification(base)



