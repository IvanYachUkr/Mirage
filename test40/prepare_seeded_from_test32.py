from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generate_persons_procedural import assign_career_timelines


ROOT_ARTIFACTS = [
    "modeling_priors.json",
    "identity_bank.json",
    "company_lexicon.json",
    "keyword_seed_bank.json",
    "character_identity_bank.json",
    "temporal_regime_plan.json",
    "title_grammar_bank.json",
    "world_policy.json",
    "year_slate_plan.json",
    "concept_packs.json",
    "keyword_motif_bank.json",
    "franchise_bibles.json",
]

ENTITY_COPY_ALL = [
    "companies.json",
    "companies_latent.json",
    "company.csv",
    "company_financial_profile.csv",
    "keywords.json",
    "keyword.csv",
    "genres.json",
    "title_bank.csv",
    "character_bank.csv",
]

ROOT_GENERATED_FILES = [
    ".pipeline_checkpoint.json",
    "pipeline_timing.jsonl",
    "critic_report.json",
    "recovery_report.json",
    "signoff_report.json",
    "signoff_report.md",
    "movie.csv",
    "movies_flat.csv",
    "movies_analysis.csv",
    "tv_series.csv",
    "seasons.csv",
    "episodes.csv",
    "episode_cast.csv",
    "persons_enriched.csv",
    "companies_enriched.csv",
]

ROOT_GENERATED_DIRS = [
    "graph",
    "checkpoints",
    "critic",
    "imdb_schema",
    "decision_logs",
]

ROOT_ARROW_TABLES = [
    "movie",
    "cast_info",
    "movie_directors",
    "movie_companies",
    "movie_keyword",
    "movie_crew",
    "release_dates",
    "box_office_weekly",
    "box_office_by_territory",
    "box_office_daily",
    "reviews",
    "awards",
    "locations",
    "alternate_titles",
    "ratings_breakdown",
    "movie_links",
    "person_demographics",
    "tv_series",
    "seasons",
    "episodes",
    "episode_cast",
    "company_links",
    "user_ratings",
    "production_timeline",
    "media_links",
    "person_contracts",
    "world_events",
    "movies_flat",
    "movies_analysis",
    "persons_enriched",
    "companies_enriched",
    "edges_temporal",
    "edges_final",
]

AWARD_PRIOR_PATCH = {
    "prestige_by_tier": {
        "Epic": 0.42,
        "A": 0.26,
        "A-List": 0.26,
        "Mid": 0.13,
        "Mid-Budget": 0.13,
        "Indie": 0.08,
        "Micro": 0.03,
        "Micro-Budget": 0.03,
    },
    "history_bonus_scale": 0.04,
    "history_bonus_cap": 0.10,
    "entry_probability": {
        "base": 0.0,
        "scale": 0.18,
        "min": 0.0,
        "max": 0.28,
    },
    "lambda_scale": 0.55,
    "lambda_cap": 1.6,
    "max_nominations": 3,
    "won_probability": {
        "base": 0.0,
        "scale": 0.06,
        "min": 0.0,
        "max": 0.16,
    },
}

POST_ACTOR_ROLE_PRIORITY = [
    "director",
    "writer",
    "producer",
    "editor",
    "cinematographer",
    "composer",
    "production_designer",
]

STAGE_ORDER = ["rising", "prime", "veteran", "legend", "retired"]
STAGE_RANK = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
STAGE_ALIASES = {
    "veterant": "veteran",
    "vetaran": "veteran",
    "legand": "legend",
    "legendary": "legend",
    "priime": "prime",
    "emerging": "rising",
    "established": "prime",
}


def _stable_int(*parts: object) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _normalize_stage(value: object) -> str:
    key = str(value or "").strip().lower()
    key = STAGE_ALIASES.get(key, key)
    return key if key in STAGE_RANK else "prime"


def _needs_timeline_repair(person: dict[str, Any]) -> bool:
    required = ("debut_year", "peak_start", "peak_end", "retirement_year", "yearly_max")
    return any(person.get(key) is None for key in required)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _keyword_json_from_csv(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = dict(row)
        for key in ("keyword_id", "specificity_tier"):
            if key in normalized:
                try:
                    normalized[key] = int(float(normalized[key]))
                except Exception:
                    pass
        for key in ("pop_weight", "franchise_affinity", "recurrence_strength"):
            if key in normalized:
                try:
                    normalized[key] = float(normalized[key])
                except Exception:
                    pass
        out.append(normalized)
    return out


def _cleanup_destination(dest_dir: Path) -> None:
    for dirname in ROOT_GENERATED_DIRS:
        target = dest_dir / dirname
        if target.exists():
            shutil.rmtree(target)
    for filename in ROOT_GENERATED_FILES:
        target = dest_dir / filename
        if target.exists():
            target.unlink()
    for name in ROOT_ARROW_TABLES:
        target = dest_dir / f"{name}.arrow"
        if target.exists():
            target.unlink()
    entities_dir = dest_dir / "entities"
    if entities_dir.exists():
        shutil.rmtree(entities_dir)


def _copy_root_artifacts(source_dir: Path, dest_dir: Path) -> None:
    if source_dir.resolve() == dest_dir.resolve():
        return
    for filename in ROOT_ARTIFACTS:
        src = source_dir / filename
        dst = dest_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing source artifact: {src}")
        shutil.copy2(src, dst)


def _patch_awards(dest_dir: Path) -> dict[str, Any]:
    path = dest_dir / "modeling_priors.json"
    payload = _read_json(path)
    secondary = payload.setdefault("secondary_table_priors", {})
    awards = secondary.setdefault("awards", {})
    awards.update(AWARD_PRIOR_PATCH)
    _write_json(path, payload)
    return awards


def _role_targets(total_target: int, available_by_role: Counter[str]) -> dict[str, int]:
    requested = {
        "actor": max(3600, int(round(total_target * 0.66))),
        "director": max(900, int(round(total_target * 0.15))),
        "writer": max(540, int(round(total_target * 0.09))),
        "producer": max(360, int(round(total_target * 0.06))),
        "editor": max(300, int(round(total_target * 0.05))),
        "cinematographer": max(300, int(round(total_target * 0.05))),
        "composer": max(240, int(round(total_target * 0.04))),
        "production_designer": max(180, int(round(total_target * 0.03))),
    }
    return {role: min(int(available_by_role.get(role, 0)), int(target)) for role, target in requested.items()}


def _stage_targets(person_records: list[dict[str, Any]], total_target: int) -> dict[str, int]:
    source_counts = Counter(str(record["stage"]) for record in person_records)
    total_source = max(1, sum(source_counts.values()))
    targets: dict[str, int] = {}
    assigned = 0
    ordered = STAGE_ORDER + [stage for stage in sorted(source_counts) if stage not in STAGE_RANK]
    for stage in ordered:
        share = float(source_counts.get(stage, 0)) / float(total_source)
        count = int(round(total_target * share))
        targets[stage] = count
        assigned += count
    if assigned != total_target:
        drift = total_target - assigned
        preferred = [stage for stage in ordered if stage in targets]
        index = 0
        while drift != 0 and preferred:
            stage = preferred[index % len(preferred)]
            if drift > 0:
                targets[stage] += 1
                drift -= 1
            elif targets[stage] > 0:
                targets[stage] -= 1
                drift += 1
            index += 1
    return targets


def _sorted_people(records: list[dict[str, Any]], *, role_hint: str | None = None) -> list[dict[str, Any]]:
    def _key(record: dict[str, Any]) -> tuple[int, int, int, int]:
        roles = record["roles"]
        stage = str(record["stage"])
        stage_rank = STAGE_RANK.get(stage, len(STAGE_RANK))
        multi_role_bonus = -len(roles)
        actor_bonus = 0 if "actor" in roles else 1
        role_bias = 0 if role_hint and role_hint in roles else 1
        noise = _stable_int(record["person_id"], role_hint or "fill")
        return (role_bias, actor_bonus + stage_rank, multi_role_bonus, noise)

    return sorted(records, key=_key)


def _pick_stage_balanced(
    candidate_records: list[dict[str, Any]],
    *,
    limit: int,
    selected: set[int],
    add_person,
    prefer_actor: bool = False,
    key_hint: str = "fill",
) -> int:
    if limit <= 0:
        return 0
    remaining = [record for record in candidate_records if int(record["person_id"]) not in selected]
    if not remaining:
        return 0

    target = min(int(limit), len(remaining))
    stage_targets = _stage_targets(remaining, target)
    ordered_stages = STAGE_ORDER + [stage for stage in sorted({str(record["stage"]) for record in remaining}) if stage not in STAGE_RANK]

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in remaining:
        buckets[str(record["stage"])].append(record)
    for stage, bucket in buckets.items():
        bucket.sort(
            key=lambda record: (
                0 if (prefer_actor and "actor" in record["roles"]) else 1,
                -len(record["roles"]),
                _stable_int(record["person_id"], stage, key_hint),
            )
        )

    positions = {stage: 0 for stage in buckets}
    stage_added: Counter[str] = Counter()
    added = 0

    def next_record(stage: str) -> dict[str, Any] | None:
        pos = positions.get(stage, 0)
        bucket = buckets.get(stage, [])
        while pos < len(bucket):
            record = bucket[pos]
            pos += 1
            positions[stage] = pos
            if int(record["person_id"]) not in selected:
                return record
        positions[stage] = pos
        return None

    progressed = True
    while added < target and progressed:
        progressed = False
        for stage in ordered_stages:
            if added >= target:
                break
            if stage_added[stage] >= int(stage_targets.get(stage, 0)):
                continue
            record = next_record(stage)
            if record is None:
                continue
            if add_person(int(record["person_id"])):
                stage_added[stage] += 1
                added += 1
                progressed = True

    if added >= target:
        return added

    fallback = sorted(
        remaining,
        key=lambda record: (
            0 if (prefer_actor and "actor" in record["roles"]) else 1,
            -len(record["roles"]),
            _stable_int(record["person_id"], key_hint, "fallback"),
        ),
    )
    for record in fallback:
        if added >= target:
            break
        if add_person(int(record["person_id"])):
            added += 1
    return added


def _select_people(
    persons: list[dict[str, Any]],
    roles_by_person: dict[int, set[str]],
    *,
    target_persons: int,
) -> tuple[list[int], dict[str, Any]]:
    person_records: list[dict[str, Any]] = []
    for row in persons:
        person_id = int(row["person_id"])
        roles = set(roles_by_person.get(person_id, set(row.get("roles") or [])))
        person_records.append(
            {
                "person_id": person_id,
                "roles": roles,
                "stage": _normalize_stage(row.get("career_stage")),
            }
        )

    available_by_role = Counter()
    for record in person_records:
        for role in record["roles"]:
            available_by_role[role] += 1

    desired_by_role = _role_targets(target_persons, available_by_role)
    desired_by_stage = _stage_targets(person_records, target_persons)

    person_index = {int(record["person_id"]): record for record in person_records}
    selected: set[int] = set()
    stage_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()

    def add_person(person_id: int) -> bool:
        if person_id in selected or len(selected) >= target_persons:
            return False
        record = person_index[person_id]
        selected.add(person_id)
        stage_counts[str(record["stage"])] += 1
        for role in record["roles"]:
            role_counts[role] += 1
        return True

    actor_target = desired_by_role.get("actor", 0)
    actor_candidates = [record for record in person_records if "actor" in record["roles"]]
    _pick_stage_balanced(
        actor_candidates,
        limit=actor_target,
        selected=selected,
        add_person=add_person,
        prefer_actor=True,
        key_hint="actor",
    )

    for role in POST_ACTOR_ROLE_PRIORITY:
        needed = max(0, int(desired_by_role.get(role, 0)) - int(role_counts.get(role, 0)))
        if needed <= 0:
            continue
        candidates = [record for record in person_records if role in record["roles"]]
        _pick_stage_balanced(
            candidates,
            limit=needed,
            selected=selected,
            add_person=add_person,
            prefer_actor=True,
            key_hint=role,
        )

    remaining_slots = max(0, target_persons - len(selected))
    if remaining_slots > 0:
        stage_fill_candidates = [record for record in person_records if int(record["person_id"]) not in selected]
        ordered_stages = STAGE_ORDER + [stage for stage in sorted(desired_by_stage) if stage not in STAGE_RANK]
        for stage in ordered_stages:
            if remaining_slots <= 0:
                break
            need = max(0, int(desired_by_stage.get(stage, 0)) - int(stage_counts.get(stage, 0)))
            if need <= 0:
                continue
            stage_candidates = [record for record in stage_fill_candidates if str(record["stage"]) == stage]
            added = _pick_stage_balanced(
                stage_candidates,
                limit=min(need, remaining_slots),
                selected=selected,
                add_person=add_person,
                prefer_actor=True,
                key_hint=f"stage-{stage}",
            )
            remaining_slots -= added

    remaining = [record for record in person_records if int(record["person_id"]) not in selected]
    _pick_stage_balanced(
        remaining,
        limit=max(0, target_persons - len(selected)),
        selected=selected,
        add_person=add_person,
        prefer_actor=True,
        key_hint="final-fill",
    )

    selected_ids = sorted(selected)
    summary = {
        "target_persons": int(target_persons),
        "selected_persons": int(len(selected_ids)),
        "available_by_role": {role: int(available_by_role[role]) for role in sorted(available_by_role)},
        "target_by_role": {role: int(desired_by_role[role]) for role in sorted(desired_by_role)},
        "selected_by_role": {role: int(role_counts[role]) for role in sorted(role_counts)},
        "target_by_stage": {stage: int(desired_by_stage[stage]) for stage in sorted(desired_by_stage)},
        "selected_by_stage": {stage: int(stage_counts[stage]) for stage in sorted(stage_counts)},
    }
    return selected_ids, summary


def _copy_entities(
    source_dir: Path,
    dest_dir: Path,
    *,
    selected_person_ids: list[int],
    seed: int,
    extra_title_source_dir: Path | None = None,
) -> dict[str, int]:
    source_entities = source_dir / "entities"
    dest_entities = dest_dir / "entities"
    dest_entities.mkdir(parents=True, exist_ok=True)

    selected_ids = set(int(person_id) for person_id in selected_person_ids)

    persons = _read_json(source_entities / "persons.json")
    persons_latent = _read_json(source_entities / "persons_latent.json")
    person_rows = _read_csv_rows(source_entities / "person.csv")
    person_role_rows = _read_csv_rows(source_entities / "person_roles.csv")

    filtered_persons = [row for row in persons if int(row["person_id"]) in selected_ids]
    filtered_persons_latent = [row for row in persons_latent if int(row["person_id"]) in selected_ids]
    filtered_person_rows = [row for row in person_rows if int(row["person_id"]) in selected_ids]
    filtered_person_role_rows = [row for row in person_role_rows if int(row["person_id"]) in selected_ids]
    for row in filtered_persons:
        row["career_stage"] = _normalize_stage(row.get("career_stage"))
    for row in filtered_person_rows:
        row["career_stage"] = _normalize_stage(row.get("career_stage"))
    if any(_needs_timeline_repair(row) for row in filtered_persons):
        assign_career_timelines(filtered_persons, seed=int(seed))

    _write_json(dest_entities / "persons.json", filtered_persons)
    _write_json(dest_entities / "persons_latent.json", filtered_persons_latent)
    _write_csv_rows(dest_entities / "person.csv", filtered_person_rows, list(person_rows[0].keys()))
    _write_csv_rows(dest_entities / "person_roles.csv", filtered_person_role_rows, list(person_role_rows[0].keys()))

    for filename in ENTITY_COPY_ALL:
        src = source_entities / filename
        dst = dest_entities / filename
        if src.exists():
            shutil.copy2(src, dst)
        elif filename == "keywords.json" and (source_entities / "keyword.csv").exists():
            _write_json(dst, _keyword_json_from_csv(source_entities / "keyword.csv"))
        else:
            raise FileNotFoundError(f"Missing source entity artifact: {src}")

    title_merge_summary: dict[str, int] = {
        "primary_title_bank_rows": 0,
        "extra_title_bank_rows": 0,
        "merged_title_bank_rows": 0,
        "extra_title_bank_added": 0,
    }
    if extra_title_source_dir is not None:
        primary_title_bank_path = dest_entities / "title_bank.csv"
        extra_title_bank_path = extra_title_source_dir / "entities" / "title_bank.csv"
        if not extra_title_bank_path.exists():
            raise FileNotFoundError(f"Missing extra title bank: {extra_title_bank_path}")

        primary_rows = _read_csv_rows(primary_title_bank_path)
        extra_rows = _read_csv_rows(extra_title_bank_path)
        fieldnames = list(primary_rows[0].keys()) if primary_rows else list(extra_rows[0].keys())

        def _title_key(row: dict[str, Any]) -> tuple[str, str, str]:
            return (
                str(row.get("title") or "").strip().casefold(),
                str(row.get("year") or "").strip(),
                str(row.get("genre_hint") or "").strip().casefold(),
            )

        merged_rows = list(primary_rows)
        seen = {_title_key(row) for row in primary_rows}
        added = 0
        for row in extra_rows:
            key = _title_key(row)
            if not key[0] or key in seen:
                continue
            normalized = {name: row.get(name, "") for name in fieldnames}
            merged_rows.append(normalized)
            seen.add(key)
            added += 1
        _write_csv_rows(primary_title_bank_path, merged_rows, fieldnames)
        title_merge_summary = {
            "primary_title_bank_rows": len(primary_rows),
            "extra_title_bank_rows": len(extra_rows),
            "merged_title_bank_rows": len(merged_rows),
            "extra_title_bank_added": added,
        }

    counts = {
        "persons": len(filtered_persons),
        "persons_latent": len(filtered_persons_latent),
        "person_rows": len(filtered_person_rows),
        "person_roles": len(filtered_person_role_rows),
    }
    for filename in ENTITY_COPY_ALL:
        path = dest_entities / filename
        if filename.endswith(".json"):
            payload = _read_json(path)
            if isinstance(payload, list):
                counts[filename] = len(payload)
        elif filename.endswith(".csv"):
            counts[filename] = max(0, sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1)
    counts.update(title_merge_summary)
    return counts


def _build_roles_by_person(source_dir: Path) -> dict[int, set[str]]:
    rows = _read_csv_rows(source_dir / "entities" / "person_roles.csv")
    roles_by_person: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        person_id = int(row["person_id"])
        role = str(row.get("role_type") or "").strip()
        if role:
            roles_by_person[person_id].add(role)
    return roles_by_person


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed a candidate workspace from one artifact source and one entity source, with optional extra title-bank rows."
    )
    parser.add_argument(
        "--artifact-source-dir",
        required=True,
        help="Directory providing top-level modeling artifacts/configuration, e.g. test34",
    )
    parser.add_argument(
        "--entity-source-dir",
        required=True,
        help="Directory providing entities/, e.g. test32",
    )
    parser.add_argument(
        "--extra-title-source-dir",
        default=None,
        help="Optional directory whose entities/title_bank.csv should be merged in, e.g. test23",
    )
    parser.add_argument("--dest-dir", required=True, help="Destination run directory, e.g. test35")
    parser.add_argument("--target-persons", type=int, default=6000, help="Size of the seeded person pool")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when timeline repair is needed")
    args = parser.parse_args()

    artifact_source_dir = Path(args.artifact_source_dir).resolve()
    entity_source_dir = Path(args.entity_source_dir).resolve()
    extra_title_source_dir = Path(args.extra_title_source_dir).resolve() if args.extra_title_source_dir else None
    dest_dir = Path(args.dest_dir).resolve()

    if not artifact_source_dir.exists():
        raise FileNotFoundError(f"Artifact source directory does not exist: {artifact_source_dir}")
    if not entity_source_dir.exists():
        raise FileNotFoundError(f"Entity source directory does not exist: {entity_source_dir}")
    if not dest_dir.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {dest_dir}")
    if entity_source_dir == dest_dir:
        raise ValueError("Entity source and destination directories must be different")
    if extra_title_source_dir is not None and extra_title_source_dir == dest_dir:
        raise ValueError("Extra title source and destination directories must be different")

    roles_by_person = _build_roles_by_person(entity_source_dir)
    persons = _read_json(entity_source_dir / "entities" / "persons.json")
    if not isinstance(persons, list) or not persons:
        raise ValueError(f"Expected non-empty persons.json in {entity_source_dir / 'entities'}")

    selected_person_ids, selection_summary = _select_people(
        persons,
        roles_by_person,
        target_persons=int(args.target_persons),
    )
    if len(selected_person_ids) < int(args.target_persons):
        raise ValueError(
            "Entity source cannot satisfy --target-persons without shrinking the run. "
            f"Requested {int(args.target_persons)} people but only selected {len(selected_person_ids)} from "
            f"{entity_source_dir}. Use a reuse profile whose N_PERSONS matches the source, or run profile-full100 "
            "to generate a fresh procedural person pool."
        )

    _cleanup_destination(dest_dir)
    _copy_root_artifacts(artifact_source_dir, dest_dir)
    award_patch = _patch_awards(dest_dir)
    entity_counts = _copy_entities(
        entity_source_dir,
        dest_dir,
        selected_person_ids=selected_person_ids,
        seed=int(args.seed),
        extra_title_source_dir=extra_title_source_dir,
    )

    manifest = {
        "artifact_source_dir": str(artifact_source_dir),
        "entity_source_dir": str(entity_source_dir),
        "extra_title_source_dir": str(extra_title_source_dir) if extra_title_source_dir is not None else None,
        "dest_dir": str(dest_dir),
        "selected_person_pool": int(len(selected_person_ids)),
        "root_artifacts_copied": list(ROOT_ARTIFACTS),
        "entity_counts": entity_counts,
        "selection_summary": selection_summary,
        "award_priors": award_patch,
    }
    _write_json(dest_dir / "seed_manifest.json", manifest)

    print(f"Seeded workspace: {dest_dir}")
    print(f"  persons:    {entity_counts['persons']}")
    print(f"  companies:  {entity_counts.get('companies.json', 0)}")
    print(f"  keywords:   {entity_counts.get('keywords.json', 0)}")
    print(f"  titles:     {entity_counts.get('title_bank.csv', 0)}")
    if extra_title_source_dir is not None:
        print(f"  extra titles added: {entity_counts.get('extra_title_bank_added', 0)}")
    print(f"  characters: {entity_counts.get('character_bank.csv', 0)}")
    print(f"  manifest:   {dest_dir / 'seed_manifest.json'}")


if __name__ == "__main__":
    main()
