from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import (
    BASE_DIR,
    ROLE_COLUMN_CANDIDATES,
    build_skill_text,
    latest_job_raw_path,
    load_records,
    normalize_alias,
    normalize_role_name,
    pick_first_key,
)


def _load_catalog(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("skills"), dict):
        payload = payload["skills"]
    out: dict[str, list[str]] = {}
    for canonical, aliases in payload.items():
        if not isinstance(aliases, list):
            continue
        normalized = sorted({normalize_alias(a) for a in aliases if str(a).strip()})
        if normalized:
            out[str(canonical)] = normalized
    return out


def _merge_catalogs(*catalogs: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for catalog in catalogs:
        for canonical, aliases in catalog.items():
            merged[canonical].update(aliases)
            merged[canonical].add(normalize_alias(canonical))
    return {k: sorted(v) for k, v in merged.items()}


def _count_skill_support(
    records: list[dict[str, Any]],
    catalog: dict[str, list[str]],
) -> tuple[dict[str, int], dict[str, set[str]]]:
    mention_counts: dict[str, int] = defaultdict(int)
    role_support: dict[str, set[str]] = defaultdict(set)

    for row in records:
        role_name = normalize_role_name(pick_first_key(row, ROLE_COLUMN_CANDIDATES))
        text = build_skill_text(row).lower()
        if not text:
            continue
        for canonical, aliases in catalog.items():
            found = False
            for alias in aliases:
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, text):
                    found = True
                    break
            if found:
                mention_counts[canonical] += 1
                role_support[canonical].add(role_name)
    return dict(mention_counts), dict(role_support)


def build_skill_catalog(
    *,
    jobs_path: Path,
    output_path: Path,
    report_path: Path,
    base_catalog_path: Path,
    seed_catalog_path: Path | None = None,
    min_mentions: int = 3,
    min_roles: int = 1,
) -> dict[str, Any]:
    records = load_records(jobs_path)
    if not records:
        raise ValueError(f"No records found in {jobs_path}")

    base_catalog = _load_catalog(base_catalog_path)
    seed_catalog = _load_catalog(seed_catalog_path) if seed_catalog_path else {}
    merged_catalog = _merge_catalogs(base_catalog, seed_catalog)
    if not merged_catalog:
        raise ValueError("Merged catalog is empty; provide at least one base catalog with aliases.")

    mention_counts, role_support = _count_skill_support(records, merged_catalog)

    retained: dict[str, list[str]] = {}
    for canonical, aliases in merged_catalog.items():
        m_count = mention_counts.get(canonical, 0)
        r_count = len(role_support.get(canonical, set()))
        if m_count >= min_mentions or r_count >= min_roles:
            retained[canonical] = aliases

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(retained.items())), f, ensure_ascii=False, indent=2)

    report = {
        "jobs_path": str(jobs_path),
        "total_jobs": len(records),
        "base_skill_count": len(merged_catalog),
        "retained_skill_count": len(retained),
        "min_mentions": min_mentions,
        "min_roles": min_roles,
        "skill_support": {
            skill: {
                "mentions": mention_counts.get(skill, 0),
                "roles": sorted(role_support.get(skill, set())),
                "role_count": len(role_support.get(skill, set())),
            }
            for skill in sorted(retained.keys())
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build staging skill_catalog from jobs raw data.")
    parser.add_argument(
        "--jobs_path",
        default="",
        help="Path to jobs raw dataset (csv/parquet/jsonl/xlsx). Defaults to latest file under data/raw/jobs.",
    )
    parser.add_argument(
        "--base_catalog",
        default="data/reference/final/skill_catalog.json",
        help="Base canonical skill catalog used for alias matching.",
    )
    parser.add_argument(
        "--seed_catalog",
        default="",
        help="Optional seed catalog from external sources.",
    )
    parser.add_argument(
        "--output_path",
        default="data/reference/staging/skill_catalog.json",
        help="Staging output for generated skill catalog.",
    )
    parser.add_argument(
        "--report_path",
        default="data/reference/staging/skill_catalog_support_report.json",
        help="Output support report path.",
    )
    parser.add_argument("--min_mentions", type=int, default=3)
    parser.add_argument("--min_roles", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_path = Path(args.jobs_path) if args.jobs_path else latest_job_raw_path(BASE_DIR)
    report = build_skill_catalog(
        jobs_path=jobs_path,
        output_path=BASE_DIR / args.output_path,
        report_path=BASE_DIR / args.report_path,
        base_catalog_path=BASE_DIR / args.base_catalog,
        seed_catalog_path=(BASE_DIR / args.seed_catalog) if args.seed_catalog else None,
        min_mentions=int(args.min_mentions),
        min_roles=int(args.min_roles),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
