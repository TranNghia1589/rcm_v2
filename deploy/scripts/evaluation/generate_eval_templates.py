from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = BASE_DIR / "data" / "reference" / "eval_templates"
CV_DATASET_PATH = BASE_DIR / "data" / "processed" / "cv_extracted" / "cv_extracted_dataset.parquet"
JOBS_DATASET_PATH = BASE_DIR / "experiments" / "artifacts" / "matching" / "jobs_matching_ready.parquet"


def _json_list(items: list[Any]) -> str:
    return json.dumps(items, ensure_ascii=False)


def _load_cv_file_names() -> list[str]:
    if not CV_DATASET_PATH.exists():
        return [f"cv_{i:03d}.pdf" for i in range(1, 123)]
    df = pd.read_parquet(CV_DATASET_PATH)
    if "file_name" not in df.columns:
        return [f"cv_{i:03d}.pdf" for i in range(1, max(len(df), 122) + 1)]
    names = [str(x).strip() for x in df["file_name"].dropna().tolist() if str(x).strip()]
    return names or [f"cv_{i:03d}.pdf" for i in range(1, 123)]


def _load_job_count() -> int:
    if not JOBS_DATASET_PATH.exists():
        return 367
    df = pd.read_parquet(JOBS_DATASET_PATH)
    return max(1, int(len(df)))


def _gen_cv_extraction(cv_ids: list[int], cv_file_names: list[str], n: int) -> pd.DataFrame:
    rows = []
    for i, cv_id in enumerate(cv_ids[:n]):
        rows.append(
            {
                "cv_id": cv_id,
                "file_name": cv_file_names[i % len(cv_file_names)],
                "skills": _json_list([]),
                "projects": _json_list([]),
                "professional_company_names": _json_list([]),
                "degree_names": _json_list([]),
                "target_role": "",
                "experience_years": "",
                "reviewer": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _gen_cv_scoring_calibration(cv_ids: list[int], n: int) -> pd.DataFrame:
    rows = []
    for cv_id in cv_ids[:n]:
        rows.append(
            {
                "cv_id": cv_id,
                "predicted_score": "",
                "human_score": "",
                "reviewer": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _gen_cv_scoring_stability(cv_ids: list[int], n_cv: int, runs_per_cv: int) -> pd.DataFrame:
    rows = []
    for cv_id in cv_ids[:n_cv]:
        for run in range(1, runs_per_cv + 1):
            rows.append(
                {
                    "cv_id": cv_id,
                    "run_id": f"run_{run:03d}",
                    "total_score": "",
                    "grade": "",
                    "model_version": "cv_scoring_v1",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def _gen_skill_gap_labels(cv_ids: list[int], n: int) -> pd.DataFrame:
    rows = []
    for cv_id in cv_ids[:n]:
        rows.append(
            {
                "cv_id": cv_id,
                "expected_missing_skills": _json_list([]),
                "reviewer": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _gen_graph_query_cases(cv_ids: list[int], n_cv: int) -> pd.DataFrame:
    rows = []
    for cv_id in cv_ids[:n_cv]:
        for query in ["recommend_jobs", "skill_gap", "career_path"]:
            rows.append(
                {
                    "query": query,
                    "cv_id": cv_id,
                    "limit": 10 if query != "career_path" else 5,
                    "expected_contains": _json_list([]),
                    "expected_not_contains": _json_list([]),
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def _gen_rag_retrieval_cases(n: int) -> pd.DataFrame:
    prompts = [
        "CV này thiếu kỹ năng gì để apply Data Analyst?",
        "Lộ trình 30-60-90 ngày để lên Data Scientist cho CV này là gì?",
        "Role nào phù hợp nhất với CV này dựa trên kinh nghiệm hiện có?",
        "Kỹ năng nào trong CV đang match tốt với job thị trường?",
        "Dự án nào cần bổ sung để tăng tỉ lệ đậu role Data Engineer?",
        "CV này nên ưu tiên học kỹ năng nào trong 3 tháng tới?",
    ]
    rows = []
    for i in range(n):
        rows.append(
            {
                "query": prompts[i % len(prompts)],
                "relevant_chunk_ids": _json_list([]),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _gen_rag_groundedness(n: int) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        rows.append(
            {
                "answer": "",
                "supported_facts": _json_list([]),
                "used_fallback": False,
                "retrieval_count": 0,
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _gen_rag_citation(n: int) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        rows.append(
            {
                "answer": "",
                "sources": _json_list([]),
                "source_chunk_ids": _json_list([]),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _gen_recommendation_qrels(cv_ids: list[int], n_cv: int, job_count: int, jobs_per_cv: int) -> pd.DataFrame:
    rows = []
    for cv_idx, cv_id in enumerate(cv_ids[:n_cv]):
        start = (cv_idx * jobs_per_cv) % job_count
        for j in range(jobs_per_cv):
            job_id = ((start + j) % job_count) + 1
            rows.append(
                {
                    "cv_id": cv_id,
                    "job_id": job_id,
                    "relevance": "",
                    "method_hits": "",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def _gen_recommendation_predictions(
    cv_ids: list[int],
    n_cv: int,
    job_count: int,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for cv_idx, cv_id in enumerate(cv_ids[:n_cv]):
        start = (cv_idx * top_k) % job_count
        for rank in range(1, top_k + 1):
            job_id = ((start + rank - 1) % job_count) + 1
            rows.append(
                {
                    "cv_id": cv_id,
                    "job_id": job_id,
                    "rank": rank,
                    "score": "",
                    "method": "hybrid",
                    "job_family": "",
                    "popularity": "",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def _gen_recommendation_explanations(cv_ids: list[int], n_rows: int, job_count: int) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        cv_id = cv_ids[i % len(cv_ids)]
        job_id = (i % job_count) + 1
        rows.append(
            {
                "cv_id": cv_id,
                "job_id": job_id,
                "method": "hybrid",
                "explanation": "",
                "supporting_chunk_ids": _json_list([]),
                "matched_skills": _json_list([]),
                "missing_skills": _json_list([]),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _gen_system_api_cases(cv_ids: list[int]) -> pd.DataFrame:
    sample_cv = cv_ids[:5]
    rows = [
        {
            "name": "health_check",
            "method": "GET",
            "path": "/healthz",
            "json": "{}",
            "params": "{}",
            "notes": "liveness",
        },
        {
            "name": "jobs_list",
            "method": "GET",
            "path": "/api/v1/jobs",
            "json": "{}",
            "params": "{\"limit\":10,\"offset\":0}",
            "notes": "catalog listing",
        },
    ]
    for cv_id in sample_cv:
        rows.append(
            {
                "name": f"chat_ask_cv_{cv_id}",
                "method": "POST",
                "path": "/api/v1/chat/ask",
                "json": f"{{\"question\":\"CV này thiếu gì để ứng tuyển Data Analyst?\",\"top_k\":3,\"cv_id\":{cv_id}}}",
                "params": "{}",
                "notes": "chat probe",
            }
        )
        rows.append(
            {
                "name": f"cv_score_{cv_id}",
                "method": "GET",
                "path": f"/api/v1/cv/score/{cv_id}",
                "json": "{}",
                "params": "{}",
                "notes": "scoring probe",
            }
        )
    return pd.DataFrame(rows)


def _gen_system_logs(n: int) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "request_id": f"req_{i:05d}",
                "endpoint": "",
                "used_fallback": "",
                "is_timeout": "",
                "status_code": "",
                "latency_ms": "",
                "fallback_stage": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate larger eval label templates.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=TEMPLATE_DIR)
    parser.add_argument("--num_cv", type=int, default=122)
    parser.add_argument("--num_jobs", type=int, default=367)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_file_names = _load_cv_file_names()
    job_count = _load_job_count() if args.num_jobs <= 0 else args.num_jobs
    cv_count = max(1, args.num_cv)
    cv_ids = list(range(1, cv_count + 1))

    files = {
        "cv_extraction_field_labels.csv": _gen_cv_extraction(cv_ids, cv_file_names, n=min(80, cv_count)),
        "cv_scoring_calibration_labels.csv": _gen_cv_scoring_calibration(cv_ids, n=min(80, cv_count)),
        "cv_scoring_stability_runs.csv": _gen_cv_scoring_stability(cv_ids, n_cv=min(40, cv_count), runs_per_cv=3),
        "skill_gap_labels.csv": _gen_skill_gap_labels(cv_ids, n=min(80, cv_count)),
        "graph_query_cases.csv": _gen_graph_query_cases(cv_ids, n_cv=min(30, cv_count)),
        "rag_retrieval_cases.csv": _gen_rag_retrieval_cases(n=120),
        "rag_groundedness_answers.csv": _gen_rag_groundedness(n=120),
        "rag_citation_answers.csv": _gen_rag_citation(n=120),
        "recommendation_qrels_labels.csv": _gen_recommendation_qrels(
            cv_ids,
            n_cv=min(50, cv_count),
            job_count=max(1, job_count),
            jobs_per_cv=20,
        ),
        "recommendation_predictions_template.csv": _gen_recommendation_predictions(
            cv_ids,
            n_cv=min(50, cv_count),
            job_count=max(1, job_count),
            top_k=10,
        ),
        "recommendation_explanation_labels.csv": _gen_recommendation_explanations(
            cv_ids,
            n_rows=200,
            job_count=max(1, job_count),
        ),
        "system_api_cases.csv": _gen_system_api_cases(cv_ids),
        "system_request_logs_template.csv": _gen_system_logs(n=200),
    }

    for name, df in files.items():
        target = out_dir / name
        df.to_csv(target, index=False, encoding="utf-8")
        print(f"[OK] {name}: {len(df)} rows")


if __name__ == "__main__":
    main()


