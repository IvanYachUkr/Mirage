"""
V13 Pipeline -- generate_movies.py (orchestrator)
==================================================
Main assembly loop + CLI entry point.
Imports all components from the modular split:
  utils, world_state, financials, assembly, secondary_tables.

No business logic lives here -- only orchestration & I/O.
"""
import pandas as pd
import numpy as np
import hashlib
import os, sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, os.path.dirname(__file__))

# === Contracts (constants & config) ====================================
from contracts import (
    SNAPSHOT_CONFIG, ENTITY_COUNTS, FRANCHISE_CONFIG,
)

# === Modular imports ==================================================
from utils import normalize_weights, _safe_float, TONE_STYLE_HINTS
from world_state import WorldState, get_person_latent, get_company_latent, latent_similarity
from financials import compute_financials, edge_is_active, record_financial_outcome
from assembly import (
    sample_movie_concept, pick_director, pick_co_director, pick_companies,
    pick_cast, pick_title, pick_keywords, pick_crew,
)
from secondary_tables import (
    generate_release_dates, generate_box_office_weekly,
    generate_box_office_daily, generate_box_office_by_territory,
    generate_reviews, generate_awards, generate_locations,
    generate_alternate_titles, generate_ratings_breakdown,
    generate_movie_links, generate_company_links,
    generate_person_demographics, generate_tv_series,
    generate_user_ratings, generate_episode_cast,
    generate_media_links,
)
from schema import (
    TABLE_DEFS, build_secondary_generators, get_auto_pk_tables,
)
from temporal_evolution_api import procedural_year_step, llm_year_step
from big_history_events import generate_and_apply_events
from generation_critic import run_post_generation_critic
from feather_sink import (
    ArrowSink, PA_SCHEMAS, STREAMABLE_TABLES, POST_LOOP_STREAMABLE,
    df_to_arrow, read_table,
)

YEARLY_SNAPSHOT_ENABLED = bool(SNAPSHOT_CONFIG.get("write_yearly_snapshots", False))


def _scaled_tv_series_target(n_movies: int, world: WorldState | None = None) -> int:
    """V17: Dynamic TV series count based on movie count.

    Uses sqrt scaling for a natural sub-linear relationship:
    - 100 movies -> ~60 series
    - 1000 movies -> ~190 series
    - 10000 movies -> ~600 series
    - 100000 movies -> ~6000 series
    """
    base = int(ENTITY_COUNTS.get("tv_series", 150))
    runtime_cfg = getattr(getattr(getattr(world, 'workspace', None), 'config', None), 'runtime', None)
    floor = int(getattr(runtime_cfg, 'tv_series_floor_small', 8))
    sqrt_scale = float(getattr(runtime_cfg, 'tv_series_sqrt_scale', 6.0))
    large_ratio = float(getattr(runtime_cfg, 'tv_series_large_ratio', 0.06))
    max_series = int(getattr(runtime_cfg, 'tv_series_max', 8000))
    sqrt_target = int(round(max(float(floor), sqrt_scale * np.sqrt(max(1, n_movies)))))
    if n_movies <= base:
        return max(floor, min(base, sqrt_target))
    scaled = max(base, sqrt_target, int(round(max(0.0, large_ratio) * max(1, n_movies))))
    if max_series > 0:
        scaled = min(max_series, scaled)
    return max(base, scaled)


# -----
# MAIN ASSEMBLY LOOP
# -

def assemble_movies(world: WorldState, n_movies: int = 5000,
                    enable_llm_evolution: bool = False,
                    llm_model: str = None,
                    evolution_log_dir: str = None,
                    checkpoint_dir: str = None) -> dict:
    """Assemble N movies with temporal evolution.

    Movies are pre-sampled by year and assembled chronologically.
    At each year boundary, procedural_year_step() nudges the world state
    (reputation, edges). If enable_llm_evolution is True, llm_year_step()
    is also called for richer graph mutations.
    """
    # Auto-initialize accumulators for in-memory tables only.
    # Streamable tables go through ArrowSink (written to disk incrementally).
    accum = {name: [] for name in TABLE_DEFS if name not in STREAMABLE_TABLES}

    # Open Arrow IPC sinks for streamable secondary tables
    base_dir = str(world.base_dir)
    sinks: dict[str, ArrowSink] = {}
    for tname in STREAMABLE_TABLES:
        schema = PA_SCHEMAS.get(tname)
        if schema is None:
            continue
        sink_path = os.path.join(base_dir, f"{tname}.arrow")
        sinks[tname] = ArrowSink(sink_path, schema)
    print(f"  Opened {len(sinks)} Arrow sinks for streaming tables")

    # Internal state for movie_links generator
    previous_movies_for_links = []

    # Global tables (generated once, not per-movie)
    accum["company_links"] = generate_company_links(
        world.companies if hasattr(world, 'companies') else None,
        world.rng
    )
    # D28: Build company family index so pick_companies can give a genre-correlation
    # boost to related companies (parent/subsidiary/co-production partners).
    # Without this, company_links records exist but have no effect -- linked companies
    # produce *less* similar movies than random (0.81x anti-correlation).
    world.company_family = {}  # company_id -> set of related company_ids
    for lnk in accum["company_links"]:
        c1 = int(lnk.get("company_id_1", 0))
        c2 = int(lnk.get("company_id_2", 0))
        if c1 and c2:
            world.company_family.setdefault(c1, set()).add(c2)
            world.company_family.setdefault(c2, set()).add(c1)
    # V18-FIX #13: Merge company_links data into _merge_families so
    # pick_companies sees both LLM merger data AND company_links.
    for cid, related in world.company_family.items():
        world._merge_families.setdefault(cid, set()).update(related)

    accum["person_demographics"] = generate_person_demographics(
        world.persons, world.rng
    )

    # TV series hierarchy (series -> seasons -> episodes)
    tv_data = generate_tv_series(
        world.persons, world.companies, world.rng,
        n_series=_scaled_tv_series_target(n_movies, world)
    )
    accum["tv_series"] = tv_data["tv_series"]
    accum["seasons"] = tv_data["seasons"]
    accum["episodes"] = tv_data["episodes"]
    # V15 FIX: generate_episode_cast was never called -- accum["episode_cast"] was
    # always [] which caused generate_media_links to find zero actor overlaps and
    # produce an empty media_links.csv with no header.
    accum["episode_cast"] = generate_episode_cast(tv_data, world.actors, world.rng)
    print(f"  TV series: {len(tv_data['tv_series'])} series, "
          f"{len(tv_data['seasons'])} seasons, "
          f"{len(tv_data['episodes'])} episodes, "
          f"{len(accum['episode_cast'])} episode cast rows")

    # NOTE: user_ratings is generated AFTER the movie loop (needs movie rows)
    # See post-processing section below

    # Build secondary generator registry
    secondary_generators = build_secondary_generators()
    # v16: no hard cap by title bank size.
    # Curated titles are used first; overflow titles are generated compositionally.

    print(f"\nAssembling {n_movies} movies (temporal mode)...")

    # === Pre-assign titles and use their LLM-generated years ==========
    # Each title in the bank has a year assigned by the LLM.
    # We shuffle the bank, assign one title per movie, and use that
    # title's year so every movie keeps its LLM-assigned time period.
    title_assignments = {}  # movie_id -> {title, tagline, year, genre_hint}
    if len(world.title_bank) > 0 and 'year' in world.title_bank.columns:
        tb = world.title_bank.sample(frac=1, random_state=world.rng.randint(0, 2**31)).reset_index(drop=True)
        for i in range(min(n_movies, len(tb))):
            mid = i + 1
            row = tb.iloc[i]
            title_assignments[mid] = {
                "title": str(row["title"]),
                "tagline": str(row.get("tagline", "")) if pd.notna(row.get("tagline")) else "",
                "year": int(row["year"]) if pd.notna(row.get("year")) else 2020,
                "genre_hint": str(row.get("genre_hint", "")) if pd.notna(row.get("genre_hint")) else "",
                # D23: propagate award_contender flag from title bank
                "award_contender": bool(row.get("award_contender", False))
                    if not (isinstance(row.get("award_contender"), float)
                            and row.get("award_contender") != row.get("award_contender")) else False,
            }
        print(f"  Pre-assigned {len(title_assignments)} titles from bank (years {tb['year'].min()}-{tb['year'].max()})")
    else:
        print(f"  WARNING: No title bank years found, falling back to YEAR_RANGE")

    # === D5 (v13): Fix franchise chronological ordering ==================
    # _setup_franchises() assigns movie IDs to franchise slots randomly (no
    # knowledge of years). Now that title_assignments gives us each movie's
    # year, we can sort each franchise's movie IDs by year so installment
    # numbers match chronological release order.
    if world.movie_franchise_map:
        # Determine max allowed year from the title bank or YEAR_RANGE
        _all_years = [ta["year"] for ta in title_assignments.values()]
        max_year = max(_all_years) if _all_years else 2025

        # Group movie IDs by franchise
        franchise_movie_ids: dict = {}
        for mid, franchise in world.movie_franchise_map.items():
            fid = franchise["franchise_id"]
            franchise_movie_ids.setdefault(fid, []).append(mid)

        # Rebuild the map: sort each group by year, assign to installment slots
        new_franchise_map = {}
        dropped_overflow = 0
        for fid, mids in franchise_movie_ids.items():
            franchise = world.movie_franchise_map[mids[0]]  # any entry, same dict
            # Sort movie IDs by their assigned year (title bank year or YEAR_RANGE fallback)
            mids_sorted = sorted(
                mids,
                key=lambda m: title_assignments[m]["year"] if m in title_assignments else 9999
            )
            # Reassign in year order -- installment 1 = earliest, etc.
            for rank, m in enumerate(mids_sorted, start=1):
                new_franchise_map[m] = franchise
            # Also override the year of earlier installments if they're out of min-gap order.
            # Guarantee at least 1 year between consecutive installments.
            # V17-FIX: Clamp to max_year to prevent sequels exceeding configured year range.
            if len(mids_sorted) > 1:
                prev_year = (title_assignments[mids_sorted[0]]["year"]
                             if mids_sorted[0] in title_assignments else None)
                for m in mids_sorted[1:]:
                    if m in title_assignments and prev_year is not None:
                        y = title_assignments[m]["year"]
                        if y <= prev_year:
                            title_assignments[m] = dict(title_assignments[m])
                            title_assignments[m]["year"] = min(prev_year + 2, max_year)
                        prev_year = title_assignments[m]["year"]
                        # If we hit the ceiling, remaining installments get clamped too
                        if prev_year >= max_year:
                            prev_year = max_year

        world.movie_franchise_map = new_franchise_map
        n_fmovies = len(new_franchise_map)
        print(f"  D5: Franchise chronological ordering applied ({n_fmovies} franchise movie slots, max_year={max_year})")



    # === Pre-sample concepts and sort chronologically ====================
    year_list = []
    for mid in range(1, n_movies + 1):
        # Use LLM-assigned year from title bank if available
        forced_year = title_assignments[mid]["year"] if mid in title_assignments else None
        concept = sample_movie_concept(world, mid, forced_year=forced_year)
        year_list.append((concept["year"], mid, concept))
    year_list.sort(key=lambda x: (x[0], x[1]))

    print(f"  Year range: {year_list[0][0]}-{year_list[-1][0]}")

    current_year = None
    year_bucket = []  # movies produced in current_year (for evolution)
    evo_stats = {"procedural_ops": 0, "llm_ops": 0, "years_evolved": 0}

    # Market Competition -- demand pool per (year, genre)
    demand_pool = {}  # (year, genre) -> remaining_demand (starts at 1.0)
    DEPLETION_BY_TIER = {"Epic": 0.25, "A": 0.15, "Mid": 0.08, "Indie": 0.04, "Micro": 0.02}

    for seq_idx, (year, mid, concept) in enumerate(year_list):
        # === Year boundary: evolve world state ========================
        if current_year is not None and year != current_year:
            # Run evolution for the completed year
            if year_bucket:
                rep = procedural_year_step(world, from_year=current_year,
                                          to_year=current_year + 1,
                                          year_bucket=year_bucket)
                evo_stats["procedural_ops"] += rep.applied
                evo_stats["years_evolved"] += 1
                print(f"  [Year {current_year}->{current_year+1}] Procedural: {rep.applied} ops applied", flush=True)

                if enable_llm_evolution:
                    print(f"  [Year {current_year}->{current_year+1}] Calling LLM evolution...", flush=True)
                    llm_rep = llm_year_step(world, from_year=current_year,
                                           to_year=current_year + 1,
                                           year_bucket=year_bucket,
                                           model=llm_model,
                                           log_dir=evolution_log_dir)
                    evo_stats["llm_ops"] += llm_rep.applied
                    print(f"  [Year {current_year}->{current_year+1}] LLM: {llm_rep.applied} ops, {llm_rep.skipped} skipped, {llm_rep.errors} errors", flush=True)
                    if llm_rep.messages:
                        for log_line in llm_rep.messages[:3]:
                            print(f"    LLM log: {log_line}", flush=True)

                # V19: BHE decoupled from LLM flag -- always fire (has own probabilistic triggers + procedural fallback)
                _bhe_result = generate_and_apply_events(
                    world, year=current_year, year_bucket=year_bucket,
                    log_dir=evolution_log_dir,
                )
                if _bhe_result.get("events"):
                    evo_stats["big_history_events"] = evo_stats.get("big_history_events", 0) + len(_bhe_result["events"])

            # === CHECKPOINT: pickle accum after each year boundary ================
            # If next year's LLM call hangs and process dies, accumulated work is
            # preserved. Overwritten each year -- only the latest checkpoint is kept.
            if checkpoint_dir and accum.get("movie"):
                try:
                    import pickle as _pickle
                    _ckpt_path = os.path.join(checkpoint_dir, "movies_checkpoint.pkl")
                    with open(_ckpt_path, "wb") as _f:
                        _pickle.dump(accum, _f)
                    n_ckpt = len(accum.get("movie", []))
                    print(f"  [Checkpoint] {n_ckpt} movies saved after year {current_year}", flush=True)
                except Exception as _ce:
                    print(f"  [Checkpoint] Warning: {_ce}", flush=True)

            # === TEMPORAL SNAPSHOT: year-level CSV exports for time-series queries ==
            # Writes 3 lightweight CSVs per year to snapshots/{year}/:
            #   edges_active.csv   - all edges with valid_from<=year and valid_to>=year (or null)
            #   persons_state.csv  - person_id, pop_weight, career_stage at this year
            #   companies_state.csv - company_id, tier, pop_weight at this year
            # These allow SQL queries like:
            #   "Show me the friendship graph in 1998"
            #   "How did career stages shift from 1990 to 2010?"
            #   "Which companies changed tier between 1995 and 2005?"
            _final_year = year_list[-1][0] if year_list else 2025
            _is_snapshot_year = (current_year % 10 == 0) or (current_year == _final_year)
            if checkpoint_dir and YEARLY_SNAPSHOT_ENABLED and _is_snapshot_year:
                try:
                    _snap_dir = os.path.join(checkpoint_dir, "snapshots", str(current_year))
                    os.makedirs(_snap_dir, exist_ok=True)

                    # 1. Active edges at this year
                    if getattr(world, "edge_graph", None) is not None:
                        _active_edges = []
                        _all_edges = []
                        for _e in world.edge_graph.edges:
                            try:
                                _vf = _e.get("valid_from")
                                _vt = _e.get("valid_to")
                                _edge_row = {
                                    "src_id": _e.get("src_id"),
                                    "dst_id": _e.get("dst_id"),
                                    "edge_type": _e.get("edge_type"),
                                    "sign": _e.get("sign"),
                                    "weight": round(float(_e.get("weight", 0.0) or 0.0), 4),
                                    "source_kind": _e.get("source_kind", ""),
                                    "valid_from": _vf,
                                    "valid_to": _vt,
                                }
                                # V17: ALL edges go into edges_all.csv (full temporal history)
                                _all_edges.append(_edge_row)
                                # Active-only filter for edges_active.csv
                                _vf_ok = (_vf is None or int(_vf) <= current_year)
                                _vt_ok = (_vt is None or int(_vt) >= current_year)
                                if _vf_ok and _vt_ok:
                                    _active_edges.append(_edge_row)
                            except Exception:
                                continue
                        if _active_edges:
                            pd.DataFrame(_active_edges).to_csv(
                                os.path.join(_snap_dir, "edges_active.csv"), index=False
                            )
                        # V17: Save complete edge history (active + expired) for temporal studies.
                        # This preserves edges that existed in the past but have since expired,
                        # enabling queries like "show all friendships that ended before 2005".
                        if _all_edges:
                            pd.DataFrame(_all_edges).to_csv(
                                os.path.join(_snap_dir, "edges_all.csv"), index=False
                            )

                    # 2. Person state
                    if world.persons is not None:
                        _cols = ["person_id", "name", "career_stage", "pop_weight",
                                 "debut_year", "retirement_year", "peak_start", "peak_end"]
                        _pcols = [c for c in _cols if c in world.persons.columns]
                        world.persons[_pcols].to_csv(
                            os.path.join(_snap_dir, "persons_state.csv"), index=False
                        )

                    # 3. Company state
                    if world.companies is not None:
                        _ccols_want = ["company_id", "name", "tier", "pop_weight",
                                       "founded_year", "defunct_year"]
                        _ccols = [c for c in _ccols_want if c in world.companies.columns]
                        world.companies[_ccols].to_csv(
                            os.path.join(_snap_dir, "companies_state.csv"), index=False
                        )

                    _n_edges = len(_active_edges) if "_active_edges" in locals() else 0
                    _n_all = len(_all_edges) if "_all_edges" in locals() else 0
                    print(f"  [Snapshot] Year {current_year}: "
                          f"{_n_edges} active / {_n_all} total edges / "
                          f"{len(world.persons) if world.persons is not None else 0} persons / "
                          f"{len(world.companies) if world.companies is not None else 0} companies", flush=True)
                except Exception as _se:
                    print(f"  [Snapshot] Warning: {_se}", flush=True)


            year_bucket = []
            # G4-FIX: Carry 50% of demand depletion into the next year.
            # Real market saturation is sticky -- if Horror was saturated in 2019,
            # studios reduce Horror output in 2020 too. A hard reset was unrealistic.
            demand_pool = {k: 0.5 + 0.5 * v for k, v in demand_pool.items()}

            # E3-FIX: index_add_edge() / index_expire_edge() keep the affinity index
            # current incrementally (O(1) per edge). No full rebuild needed here.
            # _pending_affinity_rebuild flag retained for safety but never set.
            pass  # no-op: index stays current via incremental updates

        current_year = year

        if (seq_idx + 1) % 500 == 0:
            used_actors = len(world.person_film_count)
            print(f"  Movie {seq_idx + 1}/{n_movies} (year {year})... "
                  f"({used_actors} actors, {evo_stats['years_evolved']} years evolved)")

        # Step 2: Director (+ optional co-director, D12)
        director_id = pick_director(world, concept)
        co_director_id = pick_co_director(world, concept, director_id) if director_id else None

        # Step 3: Companies
        companies = pick_companies(world, concept, director_id)
        # Step 4: Cast (shortlist + rescore + retry)
        cast, competition_pairs = pick_cast(world, concept, director_id)

        # Step 4b: Crew (below-the-line)
        crew_rows = pick_crew(world, concept, director_id, cast)

        # Step 5: Title (use pre-assigned from bank, or pick dynamically)
        if mid in title_assignments:
            ta = title_assignments[mid]
            title = ta["title"]
            tagline = ta["tagline"] if ta["tagline"] and ta["tagline"] != "nan" else ""
            award_contender = ta.get("award_contender", False)  # D23
            world.used_titles.add(title)
        else:
            title, tagline, award_contender = pick_title(world, concept)  # D23: unpack 3-tuple

        # Guard against empty/null titles (pick_title compositional fallback can return "")
        if not title or title == "nan":
            title = f"Untitled-{mid}"

        # Step 6: Financials (latent-driven correlated model)
        genre = concept["genre"]
        demand_key = (year, genre)
        remaining_demand = demand_pool.get(demand_key, 1.0)
        fin = compute_financials(world, concept, cast, director_id, companies,
                                 demand_factor=remaining_demand)
        # Deplete demand pool after this movie
        depletion = DEPLETION_BY_TIER.get(concept["tier"], 0.08)
        demand_pool[demand_key] = max(0.1, remaining_demand * (1.0 - depletion))

        # Step 7: Keywords (D25: pass company_ids for cluster routing)
        kw_ids = pick_keywords(world, concept,
                               company_ids=[c["company_id"] for c in companies])

        # === Build rows ==============================================

        # Franchise tracking
        # C2-FIX: installment is now the order in which we ENCOUNTER this franchise
        # while iterating year_list (which is sorted by year). So the first movie
        # processed for a franchise = installment 1, guaranteed chronologically correct.
        # Old code read concept["installment"] which was set during random ID shuffle,
        # before year assignment -- so installment 3 could be in 1980, installment 1 in 2005.
        franchise = concept.get("franchise")
        fid = None
        inst = None
        if franchise:
            fid = franchise["franchise_id"]
            franchise["movies_generated"] += 1
            inst = franchise["movies_generated"]  # C2-FIX: counter-based, chronological
            # R1-FIX: sample_movie_concept read movies_generated BEFORE this increment,
            # so concept["installment"] was always off by one from movie 2 onwards.
            # compute_financials reads concept.get("installment") for sequel budget growth
            # and rating decay -- correct it here so it always sees the canonical inst.
            concept["installment"] = inst
            if inst == 1:
                franchise["director_id"] = director_id
                franchise["company_ids"] = [c["company_id"] for c in companies]
                franchise["cast_pool"] = [c["person_id"] for c in cast]

        # G2-FIX: Formula-generated plot_summary -- previously always "".
        # Non-empty values are essential for text analysis tests (review CV, keyword density).
        # This is a light deterministic description; LLM enrichment can overwrite it later.
        _tier_adj = {"Epic": "sweeping", "A": "acclaimed", "Mid": "gripping",
                     "Indie": "intimate", "Micro": "raw"}.get(concept["tier"], "compelling")
        _genre = concept["genre"]
        _country = concept["country"]
        _year = concept["year"]
        _franchise_note = f" (installment {inst} of the franchise)" if inst and inst > 1 else ""
        plot_summary = (
            f"A {_tier_adj} {_genre} film from {_country} ({_year}){_franchise_note}. "
            f"{'Rated ' + fin['certification'] + '.' if fin.get('certification') else ''}"
        ).strip()

        # A1: language ISO code (from full language name)
        _LANG_CODE = {
            "English": "en", "French": "fr", "Spanish": "es", "German": "de",
            "Italian": "it", "Japanese": "ja", "Korean": "ko", "Chinese": "zh",
            "Hindi": "hi", "Portuguese": "pt", "Arabic": "ar", "Russian": "ru",
            "Turkish": "tr", "Polish": "pl", "Dutch": "nl",
        }
        original_language = _LANG_CODE.get(concept.get("language", "English"), "en")

        # A1: aspect ratio (genre + era weighted)
        _yr = concept["year"]
        if _yr < 1960:
            aspect_ratio = world.rng.choice(["1.33:1", "1.37:1"], p=[0.60, 0.40])
        elif _yr < 1980:
            aspect_ratio = world.rng.choice(["1.33:1", "1.85:1", "2.39:1"], p=[0.15, 0.55, 0.30])
        elif concept["genre"] in ("Action", "Sci-Fi", "Fantasy") or concept["tier"] == "Epic":
            aspect_ratio = world.rng.choice(["1.85:1", "2.39:1"], p=[0.35, 0.65])
        else:
            aspect_ratio = world.rng.choice(["1.85:1", "2.39:1", "1.78:1"], p=[0.65, 0.25, 0.10])

        # A1: color format (era-based)
        if _yr < 1966:
            color_format = world.rng.choice(["B&W", "Color"], p=[0.75, 0.25])
        elif _yr < 1975:
            color_format = world.rng.choice(["B&W", "Color", "Colorized"], p=[0.20, 0.78, 0.02])
        else:
            color_format = "Color"

        movie_row = {
            "title_id": mid,
            "title": title,
            "year": concept["year"],
            "country": concept["country"],
            "language": concept["language"],
            "original_language": original_language,
            "aspect_ratio": aspect_ratio,
            "color_format": color_format,
            "genre": concept["genre"],
            "production_tier": concept["tier"],
            "budget_usd": fin["budget_usd"],
            "box_office_usd": fin["box_office_usd"],
            "runtime_minutes": fin["runtime_minutes"],
            "rating": fin["rating"],
            "num_votes": fin["num_votes"],
            "certification": fin["certification"],
            "tagline": tagline,
            "plot_summary": plot_summary,
            "franchise_id": fid,
            "installment_no": inst,
            "award_campaign_strength": fin.get("award_campaign_strength", 0.0),
            "seed": world.seed,
            "snapshot_id": SNAPSHOT_CONFIG["snapshot_id"],
        }

        accum["movie"].append(movie_row)

        # === Secondary tables (auto-wired via registry) ===============
        # Compute base_release_date for downstream generators
        # Generate release dates first (needed by others)
        _rd_rows = generate_release_dates(concept, mid, world.rng)
        base_release_date = f"{int(concept['year']):04d}-01-01"
        if _rd_rows:
            if "release_dates" in sinks:
                sinks["release_dates"].write_rows(_rd_rows)
            else:
                accum["release_dates"].extend(_rd_rows)
            base_release_date = next(
                (r.get("release_date") for r in _rd_rows if r.get("release_type") == "Theatrical"),
                base_release_date
            )

        # Build context dict for all secondary generators
        movie_context = {
            "mid": mid,
            "concept": concept,
            "fin": fin,
            "director_id": director_id,
            "cast": cast,
            "crew_rows": crew_rows,
            "title": title,
            "rng": world.rng,
            "world": world,
            "base_release_date": base_release_date,
            "previous_movies_for_links": previous_movies_for_links,
            "award_contender": award_contender,  # D23: boosts award nomination probability
        }

        # D29: Generate daily first so weekly can derive from it (eliminates 98.1% mismatch)
        _daily_rows = generate_box_office_daily(
            title_id=mid,
            total_box_office_usd=fin["box_office_usd"],
            base_release_date=base_release_date,
            rng=world.rng,
        )
        if _daily_rows:
            if "box_office_daily" in sinks:
                sinks["box_office_daily"].write_rows(_daily_rows)
            else:
                accum["box_office_daily"].extend(_daily_rows)
        movie_context["_daily_rows"] = _daily_rows  # weekly will derive from these

        # Run all secondary generators (except release_dates and box_office_daily, already done)
        award_rows = []
        for gen in secondary_generators:
            if gen.table_name in ("release_dates", "box_office_daily"):
                continue
            kwargs = gen.build_args(movie_context)
            rows = gen.generate_fn(**kwargs)
            if rows:
                # Route to sink (streaming) or accum (in-memory)
                if gen.table_name in sinks:
                    sinks[gen.table_name].write_rows(rows)
                else:
                    accum[gen.table_name].extend(rows)
                if gen.table_name == "awards":
                    award_rows = rows
                if gen.post_hook:
                    gen.post_hook(rows, world)

        # V17: Feed the financial momentum system so subsequent movies
        # benefit from lagged performance memory (regime/momentum integration).
        record_financial_outcome(
            world,
            concept,
            fin,
            director_id=director_id,
            companies=companies,
            award_rows=award_rows,
        )

        previous_movies_for_links.append({"title_id": mid, "genre": concept["genre"]})

        # Track for year-boundary evolution
        year_bucket.append({
            "movie_id": mid,
            "director_id": director_id,
            "cast_ids": [c["person_id"] for c in cast],
            "company_ids": [c["company_id"] for c in companies],
            "genre": concept["genre"],
            "tier": concept["tier"],
            "rating": fin["rating"],
            "budget_usd": fin["budget_usd"],
            "box_office_usd": fin["box_office_usd"],
            "performance_ratio": float(fin.get("performance_ratio", float(fin["box_office_usd"]) / max(1.0, float(fin["budget_usd"])))),
            "market_regime_score": float(fin.get("market_regime_score", 0.0)),
            "company_momentum": float(fin.get("company_momentum", 0.0)),
            "director_momentum": float(fin.get("director_momentum", 0.0)),
            "genre_heat": float(fin.get("genre_heat", 0.0)),
            "slate_pressure": float(fin.get("slate_pressure", 1.0)),
            "competition_pairs": list(competition_pairs),
        })

        # === Dynamic edge spawning =================================
        if getattr(world, "edge_graph", None) is not None:
            spawn_rng = np.random.RandomState(
                int(hashlib.blake2b(f"spawn|{world.seed}|{mid}|{concept['year']}".encode(), digest_size=4).hexdigest(), 16))
            edges = world.edge_graph.edges
            cast_pids = [c["person_id"] for c in cast]
            spawned = 0

            if not hasattr(world, '_yearly_friendship_spawns'):
                world._yearly_friendship_spawns = {}  # type: ignore[attr-defined]
            yr_spawns = world._yearly_friendship_spawns.get(year, 0)
            # V18-SCALE: All caps proportional to graph size
            _n_persons = len(world.persons) if world.persons is not None else 10000
            _per_movie_cap = max(10, _n_persons // 2000)   # 20 @ 40K, 100 @ 200K
            _yearly_cap    = max(500, n_movies // 2)        # scales with movie count

            # Actor-actor: new friendships from co-starring
            for ci in range(len(cast_pids)):
                if spawned >= _per_movie_cap or yr_spawns >= _yearly_cap:
                    break
                for cj in range(ci + 1, len(cast_pids)):
                    if spawned >= _per_movie_cap:
                        break
                    a, b = int(cast_pids[ci]), int(cast_pids[cj])
                    key = (min(a, b), max(a, b))
                    if key in (world.affinity_index or {}).get("friendships", {}):
                        continue
                    sim = latent_similarity(world, a, b)
                    spawn_prob = 0.15 * sim  # V18-SCALE: tripled from 0.05
                    if spawn_rng.random() < spawn_prob:
                        # B07 fix: varied reason text so duplicate rate < 30%
                        _friendship_reasons = [
                            f"Co-starred in {title!r} ({concept['year']}); style sim={sim:.2f}",
                            f"Worked together on film {mid} in {concept['year']}; creative compatibility={sim:.2f}",
                            f"Developed on-set rapport during {concept['genre']} production ({concept['year']})",
                            f"Recurring collaboration since {concept['year']} ({concept['tier']} tier)",
                            f"Chemistry discovered filming {title!r}; latent sim={sim:.2f}",
                        ]
                        _reason = _friendship_reasons[int(spawn_rng.randint(0, len(_friendship_reasons)))]
                        _new_edge = {
                            "src_id": key[0], "dst_id": key[1],
                            "src_type": "person", "dst_type": "person",
                            "edge_type": "friendship", "sign": "+",
                            "weight": round(0.15 + 0.10 * sim, 3),
                            "reason": _reason,
                            "source_kind": "spawned_costar",
                            "valid_from": int(concept["year"]),
                            "valid_to": None,
                        }
                        edges.append(_new_edge)
                        # V18-FIX #12: Index immediately so pick_cast sees it
                        world.edge_graph.index_add_edge(world.affinity_index, _new_edge)
                        spawned += 1
                        yr_spawns += 1

            # Director-actor: new mentorship edges
            if director_id and spawned < _per_movie_cap:
                for pid in cast_pids[:5]:  # V18-SCALE: top-5 billed (was top-3)
                    if spawned >= _per_movie_cap:
                        break
                    existing_prefs = (world.affinity_index or {}).get("director_prefs", {}).get(int(director_id), [])
                    existing_pref_ids = set()
                    for ep in existing_prefs:
                        if isinstance(ep, dict):
                            existing_pref_ids.add(int(ep.get("actor_id", -1)))
                        elif isinstance(ep, (list, tuple)):
                            existing_pref_ids.add(int(ep[0]))
                    if int(pid) in existing_pref_ids:
                        continue
                    if spawn_rng.random() < 0.20:  # V18-SCALE: raised from 0.08
                        # B07 fix: varied mentorship reason text
                        _mentor_reasons = [
                            f"Director guided actor on {title!r} ({concept['year']}); mentorship developed",
                            f"Intensive collaboration on {concept['genre']} film {mid}; director took mentorial role",
                            f"Actor first major role under this director ({concept['year']})",
                            f"Creative mentorship formed during {concept['tier']}-tier production in {concept['year']}",
                        ]
                        _mreason = _mentor_reasons[int(spawn_rng.randint(0, len(_mentor_reasons)))]
                        _new_edge = {
                            "src_id": int(director_id), "dst_id": int(pid),
                            "src_type": "person", "dst_type": "person",
                            "edge_type": "mentorship", "sign": "+",
                            "weight": 0.30,
                            "reason": _mreason,
                            "source_kind": "spawned_mentorship",
                            "valid_from": int(concept["year"]),
                            "valid_to": None,
                        }
                        edges.append(_new_edge)
                        # V18-FIX #12: Index immediately
                        world.edge_graph.index_add_edge(world.affinity_index, _new_edge)
                        spawned += 1

            # == Chemistry edges: emerge from successful co-starring (temporal) ==
            # Chemistry is DISCOVERED by a hit film, not pre-existing.
            # Only spawn if this movie clears both quality AND commercial thresholds.
            # valid_from = release year -> feeds back into casting via D19 weight boost.
            rating_val = float(fin.get("rating", 0.0))
            bo_val = float(fin.get("box_office_usd", 0.0))
            budget_val = float(fin.get("budget_usd", 1.0))
            is_successful_film = (rating_val >= 7.0 and bo_val >= budget_val * 0.8)  # V18-SCALE: relaxed thresholds

            if is_successful_film and len(cast_pids) >= 2:
                if not hasattr(world, "_chemistry_pairs"):
                    world._chemistry_pairs = set()  # type: ignore[attr-defined]
                chemistry_spawned = 0
                for ci in range(len(cast_pids)):
                    for cj in range(ci + 1, len(cast_pids)):
                        _chem_cap = max(5, len(cast_pids) // 4)  # V18-SCALE: proportional to cast
                        if chemistry_spawned >= _chem_cap:
                            break
                        a, b = int(cast_pids[ci]), int(cast_pids[cj])
                        pair_key = (min(a, b), max(a, b))
                        if pair_key in world._chemistry_pairs:
                            continue  # already have chemistry from earlier film
                        sim = latent_similarity(world, a, b)
                        if sim >= 0.50:  # V18-SCALE: lowered from 0.70
                            _new_edge = {
                                "src_id": pair_key[0], "dst_id": pair_key[1],
                                "src_type": "person", "dst_type": "person",
                                "edge_type": "chemistry", "sign": "+",
                                "weight": round(0.50 + 0.45 * sim, 3),  # 0.84-0.95 range
                                "source_kind": "latent_hybrid",
                                "reason": f"co-star sim={sim:.2f} rating={rating_val:.1f} film_id={mid}",
                                "valid_from": int(concept["year"]),
                                "valid_to": None,
                            }
                            edges.append(_new_edge)
                            # V18-FIX #12: Index immediately
                            world.edge_graph.index_add_edge(world.affinity_index, _new_edge)
                            world._chemistry_pairs.add(pair_key)
                            chemistry_spawned += 1
                            spawned += 1
                            yr_spawns += 1
                if chemistry_spawned > 0:
                    evo_stats["chemistry_edges"] = evo_stats.get("chemistry_edges", 0) + chemistry_spawned

            # V18-FIX #14: Removed dead _pending_affinity_rebuild flag.
            # Edges are now incrementally indexed via index_add_edge (fix #12).
            if spawned > 0:
                evo_stats["spawned_edges"] = evo_stats.get("spawned_edges", 0) + spawned
                world._yearly_friendship_spawns[year] = yr_spawns

        # Cast (A1: screen_time_minutes + salary_usd)
        _SCREEN_TIME = {1: (65, 100), 2: (35, 65), 3: (20, 40)}
        # V18-FIX #15: Extended salary fractions for deep billing orders.
        # Previously only 1-3 were covered; billing 4-44+ all got identical
        # default (0.001, 0.005), creating unnaturally flat salary distributions.
        _SALARY_FRAC = {
            1: (0.03, 0.10),   # Lead star
            2: (0.01, 0.04),   # Second lead
            3: (0.005, 0.02),  # Third billing
            4: (0.003, 0.012), # Featured supporting
            5: (0.002, 0.008), # Supporting
            6: (0.001, 0.005), # Ensemble supporting
            7: (0.001, 0.004), # Ensemble supporting
            8: (0.0008, 0.003),# Minor supporting
        }
        for c in cast:
            bo = c["billing_order"]
            st_lo, st_hi = _SCREEN_TIME.get(bo, (5, 20))
            sl_lo, sl_hi = _SALARY_FRAC.get(bo, (0.001, 0.005))
            accum["cast_info"].append({
                "title_id": mid,
                "person_id": c["person_id"],
                "character_name": c["character_name"],
                "character_description": "",
                "billing_order": bo,
                "archetype": c["archetype"],
                "screen_time_minutes": int(world.rng.randint(st_lo, st_hi + 1)),
                "salary_usd": int(fin["budget_usd"] * world.rng.uniform(sl_lo, sl_hi)),
            })

        # Crew (A1: department from pick_crew) -- streamed to sink
        _crew_batch = []
        for cr in crew_rows:
            _crew_batch.append({
                "title_id": mid,
                "person_id": int(cr["person_id"]),
                "crew_role": cr["crew_role"],
                "credit_order": int(cr.get("credit_order", 1)),
                "department": str(cr.get("department", "Production")),
            })
        if _crew_batch:
            if "movie_crew" in sinks:
                sinks["movie_crew"].write_rows(_crew_batch)
            else:
                accum["movie_crew"].extend(_crew_batch)

        # Director(s)
        accum["movie_directors"].append({
            "title_id": mid,
            "director_id": director_id,
        })
        if co_director_id:  # D12: co-director row
            accum["movie_directors"].append({
                "title_id": mid,
                "director_id": co_director_id,
            })

        # Companies
        for c in companies:
            accum["movie_companies"].append({
                "title_id": mid,
                "company_id": c["company_id"],
                "role": c["role"],
            })

        # Keywords
        for kid in kw_ids:
            accum["movie_keyword"].append({
                "title_id": mid,
                "keyword_id": kid,
            })

    # ─── Final year-boundary evolution for the last year ───────────────
    if year_bucket and current_year is not None:
        rep = procedural_year_step(world, from_year=current_year,
                                  to_year=current_year + 1,
                                  year_bucket=year_bucket)
        evo_stats["procedural_ops"] += rep.applied
        evo_stats["years_evolved"] += 1
        if enable_llm_evolution:
            llm_rep = llm_year_step(world, from_year=current_year,
                                   to_year=current_year + 1,
                                   year_bucket=year_bucket,
                                   model=llm_model,
                                   log_dir=evolution_log_dir)
            evo_stats["llm_ops"] += llm_rep.applied
        # V19: BHE decoupled from LLM flag -- always fire for the final year too
        generate_and_apply_events(
            world, year=current_year, year_bucket=year_bucket,
            log_dir=evolution_log_dir,
        )

    # ─── Post-loop global tables (need completed movie rows) ───────────
    # Stream user_ratings directly to Arrow sink (largest table -- 60-400M rows at 200K).
    # The sink is passed INTO generate_user_ratings so rows stream per-movie,
    # never holding more than ~2K rows in memory at once.
    ur_schema = PA_SCHEMAS.get("user_ratings")
    ur_sink_path = os.path.join(base_dir, "user_ratings.arrow")
    ur_sink = ArrowSink(ur_sink_path, ur_schema) if ur_schema else None
    if ur_sink:
        ur_count = generate_user_ratings(accum["movie"], world.rng, sink=ur_sink)
        ur_sink.close()
        sinks["user_ratings"] = ur_sink  # track for stats
        print(f"  User ratings: {ur_count:,} ratings (streamed to {ur_sink_path})")
    else:
        ur_rows = generate_user_ratings(accum["movie"], world.rng)
        accum["user_ratings"] = ur_rows
        print(f"  User ratings: {len(ur_rows):,} ratings")

    # V14: Flush world_events (populated by LLMMasterClass during Big History Events)
    _world_events = list(getattr(world, "world_events", []))
    if _world_events:
        print(f"  World events: {len(_world_events)} events logged")

    # V14 A2/A3: Interval + cross-entity tables (post-loop; need full movie/cast data)
    # These are generated and immediately streamed to Arrow sinks.
    from secondary_tables import (
        generate_production_timeline, generate_streaming_windows,
        generate_person_contracts, generate_movie_sequence,
        generate_person_collaborations,
    )
    _post_loop = {
        "production_timeline": generate_production_timeline(accum["movie"], world.rng),
        "streaming_windows":   generate_streaming_windows(accum["movie"], world.rng),
        "person_contracts":    generate_person_contracts(world.persons, world.companies, accum["cast_info"], world.rng),
        "movie_sequence":      generate_movie_sequence(accum["movie"]),
        "person_collaborations": generate_person_collaborations(accum["cast_info"], accum["movie"]),
        "media_links":         generate_media_links(
            accum["movie"], accum["tv_series"], accum["cast_info"],
            accum["movie_companies"], accum["episode_cast"], world.rng,
        ),
        "world_events":        _world_events,
    }
    # Stream each post-loop table to its own Arrow file
    for tname, rows in _post_loop.items():
        schema = PA_SCHEMAS.get(tname)
        if schema and rows:
            sink = ArrowSink(os.path.join(base_dir, f"{tname}.arrow"), schema)
            sink.write_rows(rows)
            sink.close()
            sinks[tname] = sink  # track for stats
        elif rows:
            accum[tname] = rows
    print(
        f"  A2/A3: {len(_post_loop.get('production_timeline', []))} timeline phases, "
        f"{len(_post_loop.get('streaming_windows', []))} streaming windows, "
        f"{len(_post_loop.get('person_contracts', []))} contracts, "
        f"{len(_post_loop.get('movie_sequence', []))} seq links, "
        f"{len(_post_loop.get('person_collaborations', []))} collab pairs, "
        f"{len(_post_loop.get('media_links', []))} media links"
    )
    del _post_loop  # free memory

    # Close all in-loop sinks
    for tname, sink in sinks.items():
        if not sink._closed:
            sink.close()
            print(f"  Sink closed: {tname} ({sink.total_rows:,} rows)")

    # === Build output DataFrames (registry-driven) ===================
    # Build output DataFrames (only for in-memory tables; streamed tables are on disk)
    result = {name: pd.DataFrame(accum[name]) for name in TABLE_DEFS
              if name in accum and name not in sinks}

    # === Post-assembly: clamp birth_year in person_demographics ======
    # Demographics are generated BEFORE movies (L90), so birth_year is
    # based on debut_year from persons_enriched -- which may not match
    # the actual movie years a person gets assigned to during assembly.
    # Fix: find each person's earliest/latest movie year from the actual
    # cast_info + movie_directors tables, then clamp birth_year so that
    # age is always in [15, 100] at every assigned movie.
    demo_df = result.get("person_demographics")
    movie_df = result.get("movie")
    md_df = result.get("movie_directors")
    ci_df = result.get("cast_info")
    if demo_df is not None and len(demo_df) > 0 and movie_df is not None and len(movie_df) > 0:
        year_by_movie = movie_df.set_index("title_id")["year"]

        # V18-SCALE: Vectorized earliest/latest movie year computation
        # replaces iterrows() on 1.6M+ rows
        person_years = []
        if ci_df is not None and len(ci_df) > 0:
            ci_years = ci_df[["person_id", "title_id"]].copy()
            ci_years["year"] = ci_years["title_id"].map(year_by_movie)
            ci_years = ci_years.dropna(subset=["year"])
            ci_years["person_id"] = ci_years["person_id"].astype(int)
            ci_years["year"] = ci_years["year"].astype(int)
            person_years.append(ci_years[["person_id", "year"]])
        if md_df is not None and len(md_df) > 0:
            md_years = md_df[["director_id", "title_id"]].copy()
            md_years["year"] = md_years["title_id"].map(year_by_movie)
            md_years = md_years.dropna(subset=["year"])
            md_years.rename(columns={"director_id": "person_id"}, inplace=True)
            md_years["person_id"] = md_years["person_id"].astype(int)
            md_years["year"] = md_years["year"].astype(int)
            person_years.append(md_years[["person_id", "year"]])

        if person_years:
            all_py = pd.concat(person_years, ignore_index=True)
            year_bounds = all_py.groupby("person_id")["year"].agg(["min", "max"])
            year_bounds.columns = ["earliest", "latest"]

            # Vectorized clamping
            if "birth_date" in demo_df.columns:
                demo_df = demo_df.copy()
                demo_df["_birth_year"] = pd.to_datetime(demo_df["birth_date"], errors="coerce").dt.year
                demo_df["_pid"] = demo_df["person_id"].astype(int)
                demo_df = demo_df.merge(year_bounds, left_on="_pid", right_index=True, how="left")

                has_bounds = demo_df["earliest"].notna() & demo_df["_birth_year"].notna()
                if has_bounds.any():
                    by = demo_df.loc[has_bounds, "_birth_year"].values.astype(int)
                    max_by = demo_df.loc[has_bounds, "earliest"].values.astype(int) - 15
                    min_by = demo_df.loc[has_bounds, "latest"].values.astype(int) - 100
                    # Fix tiny windows
                    tiny = min_by > max_by
                    min_by[tiny] = max_by[tiny] - 5
                    new_by = np.clip(by, min_by, max_by)
                    changed = new_by != by

                    if changed.any():
                        # Reconstruct birth_date strings for clamped rows
                        c_idx = demo_df.index[has_bounds][changed]
                        old_dates = demo_df.loc[c_idx, "birth_date"].astype(str)
                        new_years = new_by[changed]
                        new_dates = [
                            f"{yr:04d}{d[4:]}" if isinstance(d, str) and len(d) >= 10 else f"{yr:04d}-06-15"
                            for yr, d in zip(new_years, old_dates)
                        ]
                        demo_df.loc[c_idx, "birth_date"] = new_dates
                        print(f"  Birth-year clamped for {int(changed.sum())} persons (post-assembly age fix)")

                demo_df.drop(columns=["_birth_year", "_pid", "earliest", "latest"], inplace=True, errors="ignore")
                result["person_demographics"] = demo_df

    # Auto-PK assignment for tables that need it
    for table_name, pk_col in get_auto_pk_tables().items():
        df = result[table_name]
        if len(df) > 0 and pk_col not in df.columns:
            df.insert(0, pk_col, range(1, len(df) + 1))

    # Stats
    used_actors = len(world.person_film_count)
    n_movies = len(accum["movie"])
    print(f"\n=== Assembly complete ===")
    print(f"  Movies:     {n_movies}")
    for tname in TABLE_DEFS:
        if tname == "movie":
            continue
        # Check sinks first, then accum, then result
        if tname in sinks:
            n = sinks[tname].total_rows
        elif tname in accum:
            n = len(accum[tname])
        elif tname in result:
            n = len(result[tname])
        else:
            n = 0
        if n > 0:
            avg = f" (avg {n/n_movies:.1f}/movie)" if n_movies > 0 else ""
            src = " [streamed]" if tname in sinks else ""
            print(f"  {tname:20s}: {n:>8}{avg}{src}")
    print(f"  Actors used: {used_actors}/{len(world.actors)} ({used_actors/len(world.actors)*100:.1f}%)")
    print(f"  Evolution: {evo_stats['years_evolved']} years, "
          f"{evo_stats['procedural_ops']} procedural ops, "
          f"{evo_stats['llm_ops']} LLM ops, "
          f"{evo_stats.get('spawned_edges', 0)} spawned edges")

    freq = Counter(world.person_film_count)
    top = world.person_film_count.most_common(5)
    print(f"  Top actors: {top}")

    return result


# -----------------------------------------------------------------------
# FLAT TABLE BUILDER
# -----------------------------------------------------------------------

def build_flat_table(result: dict, world: 'WorldState') -> pd.DataFrame:
    """Build the unnormalized deliverable A quickly (vectorized string assembly)."""
    movies_df = result["movie"]
    if movies_df is None or len(movies_df) == 0:
        return pd.DataFrame(columns=[
            "title_id", "title", "year", "genre", "country", "director",
            "actors", "companies", "budget_usd", "box_office_usd",
            "rating", "description", "keywords",
        ])

    cast_df = result["cast_info"]
    mc_df = result["movie_companies"]
    mk_df = result["movie_keyword"]
    md_df = result["movie_directors"]

    person_name = dict(zip(world.persons["person_id"], world.persons["name"]))
    company_name = dict(zip(world.companies["company_id"], world.companies["name"]))
    keyword_text = dict(zip(
        world.keywords["keyword_id"],
        world.keywords.get("keyword", world.keywords.get("name", world.keywords.iloc[:, 1]))
    ))
    director_name = dict(zip(world.directors["person_id"], world.directors["name"]))

    # V17: Vectorized string assembly using .map() + .agg() instead of per-row loops
    cast_by_mid = {}
    if len(cast_df) > 0 and "title_id" in cast_df.columns:
        cast_work = cast_df.copy()
        cast_work["person_label"] = cast_work["person_id"].map(person_name).fillna("?")
        if "archetype" in cast_work.columns:
            cast_work["actor_label"] = cast_work["person_label"] + " (" + cast_work["archetype"].fillna("").astype(str) + ")"
        else:
            cast_work["actor_label"] = cast_work["person_label"]
        sort_cols = [c for c in ["title_id", "billing_order"] if c in cast_work.columns]
        if sort_cols:
            cast_work = cast_work.sort_values(sort_cols)
        cast_by_mid = cast_work.groupby("title_id")["actor_label"].agg("; ".join).to_dict()

    comp_by_mid = {}
    if len(mc_df) > 0 and "title_id" in mc_df.columns:
        comp_work = mc_df.copy()
        comp_work["company_label"] = comp_work["company_id"].map(company_name).fillna("?")
        if "role" in comp_work.columns:
            comp_work["company_label"] = comp_work["company_label"] + " (" + comp_work["role"].fillna("").astype(str) + ")"
        comp_by_mid = comp_work.groupby("title_id")["company_label"].agg("; ".join).to_dict()

    dir_by_mid = {}
    if len(md_df) > 0 and {"title_id", "director_id"}.issubset(md_df.columns):
        dir_first = md_df.groupby("title_id")["director_id"].first()
        dir_by_mid = dir_first.map(lambda pid: director_name.get(pid, "?")).to_dict()

    kw_by_mid = {}
    if len(mk_df) > 0 and {"title_id", "keyword_id"}.issubset(mk_df.columns):
        kw_work = mk_df.copy()
        kw_work["keyword_label"] = kw_work["keyword_id"].map(keyword_text).fillna("?").astype(str)
        kw_by_mid = kw_work.groupby("title_id")["keyword_label"].agg(", ".join).to_dict()

    rows = []
    append_row = rows.append
    for row in movies_df.itertuples(index=False):
        mid = int(getattr(row, "title_id"))
        append_row({
            "title_id": mid,
            "title": getattr(row, "title", ""),
            "year": getattr(row, "year", None),
            "genre": getattr(row, "genre", ""),
            "country": getattr(row, "country", ""),
            "director": dir_by_mid.get(mid, "?"),
            "actors": cast_by_mid.get(mid, ""),
            "companies": comp_by_mid.get(mid, ""),
            "budget_usd": getattr(row, "budget_usd", None),
            "box_office_usd": getattr(row, "box_office_usd", None),
            "rating": getattr(row, "rating", None),
            "description": getattr(row, "plot_summary", ""),
            "keywords": kw_by_mid.get(mid, ""),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def _count_available_titles(base_dir: str) -> int:
    """Choose default movie count with overflow support.

    Uses title bank size as a floor and ENTITY_COUNTS['movies'] as the target.
    If target is None (auto-detect), generates exactly one movie per curated title.
    If target exceeds curated titles, generation falls back to compositional titles.
    """
    import json as _json
    edir = Path(base_dir) / "entities"

    configured_target = ENTITY_COUNTS.get("movies")
    if configured_target is not None:
        configured_target = int(configured_target)

    # Try CSV first (post-conversion)
    csv_path = edir / "title_bank.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        curated = int(len(df))
        if configured_target is None:
            print(f"  Auto-detect: will generate {curated} movies (1 per curated title)")
            return curated
        if configured_target > curated:
            print(f"  Title bank has {curated} curated titles; will generate {configured_target - curated} compositional overflow titles.")
        return max(curated, configured_target)

    # Try JSON (pre-conversion)
    json_path = edir / "movie_titlebank.json"
    if json_path.exists():
        data = _json.loads(json_path.read_text(encoding="utf-8"))
        curated = int(len(data))
        if configured_target is None:
            print(f"  Auto-detect: will generate {curated} movies (1 per curated title)")
            return curated
        if configured_target > curated:
            print(f"  Title bank has {curated} curated titles; will generate {configured_target - curated} compositional overflow titles.")
        return max(curated, configured_target)

    # Fallback
    fallback = configured_target or 5000
    print(f"  WARNING: No title bank found -- using fallback {fallback}")
    return fallback


def main(base_dir: str = None, n_movies: int = None,
         enable_llm_evolution: bool = False,
         enable_llm_critic: bool = False,
         llm_model: str = None,
         llm_critic_model: str | None = None,
         evolution_log_dir: str = None,
         critic_log_dir: str | None = None):
    """Run full movie assembly pipeline."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Dynamic movie count: use title bank size unless explicitly overridden
    if n_movies is None:
        n_movies = _count_available_titles(base_dir)
        print(f"  Auto-detected {n_movies} titles in title bank")
    
    # Ensure dependent components (e.g., franchise planner) see the requested scale
    ENTITY_COUNTS["movies"] = int(n_movies)

    world = WorldState(base_dir, seed=SNAPSHOT_CONFIG["seed"])
    world.load()

    # Save enriched persons with pop_weight for verify.py
    df_to_arrow(world.persons, os.path.join(base_dir, "persons_enriched.arrow"))

    # V15 FIX: Save enriched companies so defunct_year mutations from llm_master.py
    # (e.g. merge_companies marking company B defunct) are persisted to disk.
    df_to_arrow(world.companies, os.path.join(base_dir, "companies_enriched.arrow"))

    # Default evolution log dir
    if evolution_log_dir is None and enable_llm_evolution:
        evolution_log_dir = os.path.join(base_dir, "graph", "temporal_patches")
    if critic_log_dir is None and enable_llm_critic:
        critic_log_dir = os.path.join(base_dir, "critic")

    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    result = assemble_movies(world, n_movies,
                             enable_llm_evolution=enable_llm_evolution,
                             llm_model=llm_model,
                             evolution_log_dir=evolution_log_dir,
                             checkpoint_dir=checkpoint_dir)

    # V19-FIX: Re-save enriched data AFTER assembly.  Temporal evolution
    # (procedural_year_step, llm_year_step) mutates world.persons pop_weight,
    # career_stage, retirement_year etc. and world.companies defunct_year
    # during the loop.  The pre-assembly save captured the initial state;
    # this captures the post-evolution final state.
    df_to_arrow(world.persons, os.path.join(base_dir, "persons_enriched.arrow"))
    df_to_arrow(world.companies, os.path.join(base_dir, "companies_enriched.arrow"))

    # V17: Post-generation LLM critic -- samples riskiest movies, proposes
    # bounded safe repairs (plot summaries, taglines, keywords, alt titles).
    critic_report = {"status": "disabled", "applied": 0, "sampled_titles": []}
    if enable_llm_critic:
        result, critic_report = run_post_generation_critic(
            result,
            world,
            enabled=True,
            model=llm_critic_model,
            log_dir=critic_log_dir,
        )
        print(
            f"  Post-generation critic: {critic_report.get('status', 'unknown')} "
            f"(applied={critic_report.get('applied', 0)}, sampled={len(critic_report.get('sampled_titles', []))}, "
            f"cache_hit={critic_report.get('cache_hit', False)})"
        )
        try:
            import json as _json_cr
            with open(os.path.join(base_dir, "critic_report.json"), "w", encoding="utf-8") as _f:
                _json_cr.dump(critic_report, _f, ensure_ascii=False, indent=2)
        except Exception as _critic_exc:
            print(f"  Critic report warning: {_critic_exc}")

    # Save in-memory tables as Arrow IPC (.arrow)
    for name, df in result.items():
        path = os.path.join(base_dir, f"{name}.arrow")
        df_to_arrow(df, path, table_name=name)
        print(f"Saved {path} ({len(df):,} rows)")
    # Streamed tables were already written by their sinks during assembly.
    # List them for completeness.
    for tname in STREAMABLE_TABLES | POST_LOOP_STREAMABLE:
        arrow_path = os.path.join(base_dir, f"{tname}.arrow")
        if os.path.exists(arrow_path):
            sz_mb = os.path.getsize(arrow_path) / 1e6
            print(f"Saved {arrow_path} (streamed, {sz_mb:.1f} MB)")

    # Save flat table (Deliverable A)
    flat = build_flat_table(result, world)
    df_to_arrow(flat, os.path.join(base_dir, "movies_flat.arrow"))
    print(f"Saved movies_flat.arrow (Deliverable A, {len(flat)} rows x {len(flat.columns)} cols)")

    # Save analysis table (Deliverable B = A + debug cols)
    analysis = flat.copy()
    movie_df = result["movie"]
    for col in ["title_id", "production_tier", "runtime_minutes", "certification",
                 "num_votes", "franchise_id", "installment_no", "seed", "snapshot_id"]:
        if col in movie_df.columns:
            analysis[col] = movie_df[col].values
    df_to_arrow(analysis, os.path.join(base_dir, "movies_analysis.arrow"))
    print(f"Saved movies_analysis.arrow (Deliverable B)")

    # Save complete temporal edge graph (all versions, including expired SCD2 rows).
    # This is the definitive output for temporal graph analysis.
    edge_graph = getattr(world, "edge_graph", None)
    if edge_graph is not None and getattr(edge_graph, "edges", None):
        all_edges = []
        active_edges = []
        final_year = max((m.get("year", 2025) for m in result.get("movie", pd.DataFrame()).to_dict("records")), default=2025) if "movie" in result else 2025
        for e in edge_graph.edges:
            try:
                vf = e.get("valid_from")
                vt = e.get("valid_to")
                row = {
                    "src_id": e.get("src_id"),
                    "dst_id": e.get("dst_id"),
                    "src_type": e.get("src_type", "person"),
                    "dst_type": e.get("dst_type", "person"),
                    "edge_type": e.get("edge_type"),
                    "sign": e.get("sign"),
                    "weight": round(float(e.get("weight", 0.0) or 0.0), 4),
                    "source_kind": e.get("source_kind", ""),
                    "reason": e.get("reason", ""),
                    "valid_from": vf,
                    "valid_to": vt,
                }
                all_edges.append(row)
                # Active-only: not retired and valid_from/valid_to bracket includes final year
                if not e.get("_scd2_retired", False):
                    vf_ok = (vf is None or int(vf) <= final_year)
                    vt_ok = (vt is None or int(vt) >= final_year)
                    if vf_ok and vt_ok:
                        active_edges.append(row)
            except Exception:
                continue
        if all_edges:
            df_to_arrow(pd.DataFrame(all_edges), os.path.join(base_dir, "edges_temporal.arrow"))
            print(f"Saved edges_temporal.arrow (full SCD2 history, {len(all_edges)} rows)")
        if active_edges:
            df_to_arrow(pd.DataFrame(active_edges), os.path.join(base_dir, "edges_final.arrow"))
            print(f"Saved edges_final.arrow (active edges at final year, {len(active_edges)} rows)")

    return result, world


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test 12: generate movies + relational tables")
    parser.add_argument("--base_dir", default=None, help="Directory containing entities/ and graph/")
    parser.add_argument("--n_movies", type=int, default=None,
                        help="Number of movies to generate")
    parser.add_argument("--enable_llm_evolution", action="store_true", default=None,
                        help="Enable LLM-based world evolution at year boundaries (OFF by default; requires GEMINI_API_KEY)")
    parser.add_argument("--disable_llm_evolution", action="store_true", default=None,
                        help="Disable LLM-based world evolution at year boundaries")
    parser.add_argument("--llm_model", default=None,
                        help="Gemini model for LLM evolution (default: gemini-2.5-flash-lite)")
    parser.add_argument("--enable_llm_critic", action="store_true", default=None,
                        help="Enable the bounded post-generation LLM critic/repair pass")
    parser.add_argument("--disable_llm_critic", action="store_true", default=None,
                        help="Disable the post-generation LLM critic/repair pass")
    parser.add_argument("--llm_critic_model", default=None,
                        help="Model override for the post-generation LLM critic")
    parser.add_argument("--evolution_log_dir", default=None,
                        help="Directory for evolution patch logs (default: graph/temporal_patches/)")
    parser.add_argument("--critic_log_dir", default=None,
                        help="Directory for post-generation critic logs (default: critic/)")
    args = parser.parse_args()

    # v16: LLM evolution is OFF by default; explicit opt-in required.
    enable_llm_evolution = bool(getattr(args, 'enable_llm_evolution', False))
    if getattr(args, 'disable_llm_evolution', False):
        enable_llm_evolution = False
    enable_llm_critic = bool(getattr(args, 'enable_llm_critic', False))
    if getattr(args, 'disable_llm_critic', False):
        enable_llm_critic = False


    main(base_dir=args.base_dir, n_movies=args.n_movies,
         enable_llm_evolution=enable_llm_evolution,
         enable_llm_critic=enable_llm_critic,
         llm_model=args.llm_model,
         llm_critic_model=args.llm_critic_model,
         evolution_log_dir=args.evolution_log_dir,
         critic_log_dir=args.critic_log_dir)
