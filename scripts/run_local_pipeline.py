from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def run_cmd(cmd: list[str], dry_run: bool, extra_env: dict[str, str] | None = None) -> None:
    pretty = " ".join(cmd)
    print(f"[RUN] {pretty}")
    if dry_run:
        return
    env = dict(**os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def stage_commands() -> dict[str, list[str]]:
    py = str(PYTHON)
    return {
        "preprocess_jobs": [py, "src/pipelines/run_preprocess.py"],
        "extract_cv": [
            py,
            "src/cv/extract_cv_batch.py",
            "--input_dir",
            "data/raw/cv_samples/INFORMATION-TECHNOLOGY",
            "--output_dir",
            "data/processed/cv_extracted",
            "--aggregate_jsonl",
            "data/processed/cv_extracted/cv_extracted_dataset.jsonl",
            "--aggregate_parquet",
            "data/processed/cv_extracted/cv_extracted_dataset.parquet",
        ],
        "cv_gap": [
            py,
            "src/cv/run_gap_batch.py",
            "--cv_dataset",
            "data/processed/cv_extracted/cv_extracted_dataset.parquet",
            "--role_profiles",
            "data/reference/role_profiles.json",
            "--output_dir",
            "data/processed/cv_gap_reports",
            "--aggregate_jsonl",
            "data/processed/cv_gap_reports/cv_gap_dataset.jsonl",
            "--aggregate_parquet",
            "data/processed/cv_gap_reports/cv_gap_dataset.parquet",
        ],
        "load_core_tables": [
            py,
            "src/ingestion/load_core_tables.py",
            "--postgres_config",
            "configs/db/postgres.yaml",
            "--jobs_parquet",
            "artifacts/matching/jobs_matching_ready_v3.parquet",
            "--job_skill_map",
            "artifacts/matching/job_skill_map_v3.parquet",
            "--cv_dataset",
            "data/processed/cv_extracted/cv_extracted_dataset.parquet",
            "--gap_dir",
            "data/processed/cv_gap_reports",
        ],
        "cv_scoring": [
            py,
            "src/scoring/run_cv_scoring_batch.py",
            "--postgres_config",
            "configs/db/postgres.yaml",
            "--role_profiles",
            "data/reference/role_profiles.json",
            "--model_version",
            "cv_scoring_v1",
        ],
        "rag_ingest": [
            py,
            "src/ingestion/jobs_to_pgvector.py",
            "--jobs_path",
            "artifacts/matching/jobs_chatbot_sections_v3.parquet",
            "--postgres_config",
            "configs/db/postgres.yaml",
            "--pgvector_config",
            "configs/db/pgvector.yaml",
            "--chunking_config",
            "configs/rag/chunking.yaml",
            "--embedding_config",
            "configs/model/embedding.yaml",
        ],
        "graph_etl": [
            py,
            "src/ingestion/jobs_to_neo4j.py",
            "--postgres_config",
            "configs/db/postgres.yaml",
            "--neo4j_config",
            "configs/db/neo4j.yaml",
            "--reset_graph",
        ],
        "api_tests": [py, "-m", "pytest", "apps/api/tests", "-q", "-p", "no:cacheprovider"],
    }


DEFAULT_ORDER = [
    "preprocess_jobs",
    "extract_cv",
    "cv_gap",
    "load_core_tables",
    "cv_scoring",
    "rag_ingest",
    "graph_etl",
    "api_tests",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local end-to-end pipeline.")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=DEFAULT_ORDER,
        help=f"Subset of stages to run. Available: {', '.join(DEFAULT_ORDER)}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--skip-preprocess-embedding",
        action="store_true",
        help="Run preprocess stage without PhoBERT embedding generation.",
    )
    return parser.parse_args()


def preflight_check_reference_data() -> None:
    ref_skill = ROOT / "data" / "reference" / "skill_catalog.json"
    ref_role = ROOT / "data" / "reference" / "role_profiles.json"
    legacy_skill = ROOT / "data" / "skill_catalog.json"
    legacy_role = ROOT / "data" / "role_profiles" / "role_profiles.json"

    missing = [str(p) for p in [ref_skill, ref_role] if not p.exists()]
    if missing:
        raise SystemExit(f"Missing canonical reference files: {missing}")

    def _same_json(a: Path, b: Path) -> bool:
        try:
            return json.loads(a.read_text(encoding="utf-8")) == json.loads(b.read_text(encoding="utf-8"))
        except Exception:
            return False

    if legacy_skill.exists():
        if _same_json(ref_skill, legacy_skill):
            print(f"[WARN] Legacy duplicate exists (safe to remove): {legacy_skill}")
        else:
            raise SystemExit(
                "Detected drift between canonical and legacy skill catalog. "
                f"Canonical={ref_skill}, Legacy={legacy_skill}"
            )
    if legacy_role.exists():
        if _same_json(ref_role, legacy_role):
            print(f"[WARN] Legacy duplicate exists (safe to remove): {legacy_role}")
        else:
            raise SystemExit(
                "Detected drift between canonical and legacy role profiles. "
                f"Canonical={ref_role}, Legacy={legacy_role}"
            )


def main() -> None:
    args = parse_args()
    preflight_check_reference_data()
    commands = stage_commands()

    for stage in args.stages:
        if stage not in commands:
            valid = ", ".join(DEFAULT_ORDER)
            raise SystemExit(f"Unknown stage: {stage}. Valid stages: {valid}")
        print(f"\n=== Stage: {stage} ===")
        stage_env = None
        if stage == "preprocess_jobs" and args.skip_preprocess_embedding:
            stage_env = {
                "PREPROCESS_RUN_EMBEDDING": "0",
                "PREPROCESS_RUN_SECTION_EMBEDDING": "0",
            }
            print("[WARN] PREPROCESS_RUN_EMBEDDING=0 (skip PhoBERT embedding for this run).")
        run_cmd(commands[stage], dry_run=args.dry_run, extra_env=stage_env)

    print("\n[DONE] Pipeline execution completed.")


if __name__ == "__main__":
    main()
