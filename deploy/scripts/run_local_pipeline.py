from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").strip()
DEFAULT_CHATBOT_QUESTION = os.getenv(
    "CHATBOT_QUESTION",
    "Toi la data analyst, nen chuan bi gi de len senior?",
).strip()
DEFAULT_CHATBOT_TOP_K = os.getenv("CHATBOT_TOP_K", "5").strip()
DEFAULT_CHATBOT_CV_ID = os.getenv("CHATBOT_CV_ID", "").strip()


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
        "crawl_jobs": [py, "-m", "src.data_processing.pipelines.run_crawl"],
        "preprocess_jobs": [py, "-m", "src.data_processing.pipelines.run_preprocess"],
        "extract_cv": [
            py,
            "-m",
            "src.models.cv.extract_cv_batch",
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
            "-m",
            "src.models.cv.run_gap_batch",
            "--cv_dataset",
            "data/processed/cv_extracted/cv_extracted_dataset.parquet",
            "--role_profiles",
            "data/reference/final/role_profiles.json",
            "--output_dir",
            "data/processed/cv_gap_reports",
            "--aggregate_jsonl",
            "data/processed/cv_gap_reports/cv_gap_dataset.jsonl",
            "--aggregate_parquet",
            "data/processed/cv_gap_reports/cv_gap_dataset.parquet",
        ],
        "load_core_tables": [
            py,
            "-m",
            "src.data_processing.ingestion.load_core_tables",
            "--postgres_config",
            "config/db/postgres.yaml",
            "--jobs_parquet",
            "experiments/artifacts/matching/jobs_matching_ready.parquet",
            "--job_skill_map",
            "experiments/artifacts/matching/job_skill_map.parquet",
            "--cv_dataset",
            "data/processed/cv_extracted/cv_extracted_dataset.parquet",
            "--gap_dir",
            "data/processed/cv_gap_reports",
        ],
        "cv_scoring": [
            py,
            "-m",
            "src.models.scoring.run_cv_scoring_batch",
            "--postgres_config",
            "config/db/postgres.yaml",
            "--role_profiles",
            "data/reference/final/role_profiles.json",
            "--model_version",
            "cv_scoring_v1",
        ],
        "rag_ingest": [
            py,
            "-m",
            "src.data_processing.ingestion.jobs_to_pgvector",
            "--jobs_path",
            "experiments/artifacts/matching/jobs_chatbot_sections.parquet",
            "--postgres_config",
            "config/db/postgres.yaml",
            "--pgvector_config",
            "config/db/pgvector.yaml",
            "--chunking_config",
            "config/rag/chunking.yaml",
            "--embedding_config",
            "config/model/embedding.yaml",
        ],
        "graph_etl": [
            py,
            "-m",
            "src.data_processing.ingestion.jobs_to_neo4j",
            "--postgres_config",
            "config/db/postgres.yaml",
            "--neo4j_config",
            "config/db/neo4j.yaml",
            "--reset_graph",
        ],
        "api_tests": [py, "-m", "pytest", "apps/api/tests", "-q", "-p", "no:cacheprovider"],
        "chatbot_smoke": [
            py,
            "-c",
            (
                "import json, os, sys, urllib.request; "
                "base = os.getenv('API_BASE_URL', 'http://127.0.0.1:8000').rstrip('/'); "
                "url = base + '/api/v1/chat/ask'; "
                "payload = {"
                "  'question': os.getenv('CHATBOT_QUESTION', 'Toi la data analyst, nen chuan bi gi de len senior?'),"
                "  'top_k': int(os.getenv('CHATBOT_TOP_K', '5'))"
                "}; "
                "cv_id = os.getenv('CHATBOT_CV_ID', '').strip(); "
                "payload = dict(payload, cv_id=int(cv_id)) if cv_id else payload; "
                "data = json.dumps(payload).encode('utf-8'); "
                "req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}); "
                "resp = urllib.request.urlopen(req, timeout=240); "
                "print(resp.read().decode('utf-8'))"
            ),
        ],
    }


DEFAULT_ORDER = [
    "crawl_jobs",
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
    ref_skill = ROOT / "data" / "reference" / "final" / "skill_catalog.json"
    ref_role = ROOT / "data" / "reference" / "final" / "role_profiles.json"
    legacy_skill = ROOT / "data" / "skill_catalog.json"
    legacy_role = ROOT / "data" / "role_profiles" / "role_profiles.json"
    raw_jobs_dir = ROOT / "data" / "raw" / "jobs"
    raw_cvs_dir = ROOT / "data" / "raw" / "cv_samples" / "INFORMATION-TECHNOLOGY"

    missing = [str(p) for p in [ref_skill, ref_role, raw_jobs_dir, raw_cvs_dir] if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required local inputs/refs: {missing}")

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

    print("[INFO] Local pipeline prerequisites:")
    print("[INFO] - PostgreSQL running on localhost:5432")
    print("[INFO] - Neo4j running on bolt://127.0.0.1:7687")
    print("[INFO] - Ollama running on http://localhost:11434 with qwen2.5:3b available")
    print("[INFO] - Embedding service running on http://127.0.0.1:8081 or EMBEDDING_SERVICE_URL set")
    print("[INFO] - API server running before chatbot_smoke if you include that stage")

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
        if stage == "chatbot_smoke":
            stage_env = {
                "API_BASE_URL": DEFAULT_API_BASE_URL,
                "CHATBOT_QUESTION": DEFAULT_CHATBOT_QUESTION,
                "CHATBOT_TOP_K": DEFAULT_CHATBOT_TOP_K,
            }
            if DEFAULT_CHATBOT_CV_ID:
                stage_env["CHATBOT_CV_ID"] = DEFAULT_CHATBOT_CV_ID
        run_cmd(commands[stage], dry_run=args.dry_run, extra_env=stage_env)

    print("\n[DONE] Pipeline execution completed.")


if __name__ == "__main__":
    main()



