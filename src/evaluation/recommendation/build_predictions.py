from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.infrastructure.db.postgres_client import PostgresClient, PostgresConfig
from src.models.recommendation.candidate_generation import vector_candidates
from src.models.recommendation.graph_ranking import graph_candidates
from src.models.recommendation.orchestrator import run_hybrid_recommendation


BASE_DIR = Path(__file__).resolve().parents[3]


def _parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def _parse_methods(raw: str) -> list[str]:
    allowed = {"vector", "graph", "hybrid"}
    methods = [x.strip().lower() for x in raw.split(",") if x.strip()]
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported methods: {bad}. Allowed: {sorted(allowed)}")
    return methods


def _load_cv_info(postgres_cfg: Path, cv_ids: list[int]) -> dict[int, dict]:
    pg_conf = PostgresConfig.from_yaml(postgres_cfg)
    info: dict[int, dict] = {}
    with PostgresClient(pg_conf) as client:
        for cv_id in cv_ids:
            row = client.fetch_one(
                """
                SELECT cv_id, COALESCE(target_role, ''), COALESCE(raw_text, '')
                FROM cv_profiles
                WHERE cv_id = %s
                LIMIT 1
                """,
                (int(cv_id),),
            )
            if not row:
                info[int(cv_id)] = {"target_role": "", "raw_text": ""}
            else:
                info[int(cv_id)] = {"target_role": str(row[1] or ""), "raw_text": str(row[2] or "")}
    return info


def _load_job_family_map(postgres_cfg: Path) -> dict[int, str]:
    pg_conf = PostgresConfig.from_yaml(postgres_cfg)
    out: dict[int, str] = {}
    with PostgresClient(pg_conf) as client:
        rows = client.fetch_all(
            """
            SELECT job_id, COALESCE(job_family, '')
            FROM jobs
            """
        )
        for job_id, fam in rows:
            out[int(job_id)] = str(fam or "")
    return out


def _default_question(target_role: str) -> str:
    role = target_role.strip() or "Data Analyst"
    return f"Goi y viec lam {role} phu hop voi CV nay"


def _rows_for_method(
    *,
    method: str,
    cv_id: int,
    question: str,
    top_k: int,
    postgres_cfg: Path,
    retrieval_cfg: Path,
    embedding_cfg: Path,
    neo4j_cfg: Path,
    hybrid_cfg: Path,
) -> list[dict]:
    if method == "vector":
        items = vector_candidates(
            question=question,
            postgres_config_path=postgres_cfg,
            retrieval_config_path=retrieval_cfg,
            embedding_config_path=embedding_cfg,
            top_k=top_k,
        )
        out = []
        for i, x in enumerate(items[:top_k], start=1):
            out.append(
                {
                    "cv_id": cv_id,
                    "job_id": int(x["job_id"]),
                    "rank": i,
                    "score": float(x.get("vector_score", 0.0)),
                    "method": "vector",
                    "question": question,
                }
            )
        return out

    if method == "graph":
        items = graph_candidates(
            cv_id=cv_id,
            neo4j_config_path=neo4j_cfg,
            top_k=top_k,
        )
        out = []
        for i, x in enumerate(items[:top_k], start=1):
            out.append(
                {
                    "cv_id": cv_id,
                    "job_id": int(x["job_id"]),
                    "rank": i,
                    "score": float(x.get("graph_score", 0.0)),
                    "method": "graph",
                    "question": question,
                }
            )
        return out

    if method == "hybrid":
        result = run_hybrid_recommendation(
            question=question,
            cv_id=cv_id,
            postgres_config_path=postgres_cfg,
            neo4j_config_path=neo4j_cfg,
            retrieval_config_path=retrieval_cfg,
            embedding_config_path=embedding_cfg,
            hybrid_config_path=hybrid_cfg,
        )
        items = result.get("items", [])
        out = []
        for i, x in enumerate(items[:top_k], start=1):
            out.append(
                {
                    "cv_id": cv_id,
                    "job_id": int(x["job_id"]),
                    "rank": i,
                    "score": float(x.get("hybrid_score", 0.0)),
                    "method": "hybrid",
                    "question": question,
                }
            )
        return out

    raise ValueError(f"Unknown method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prediction table for offline recommendation evaluation.")
    parser.add_argument("--cv_ids", required=True, help="Comma-separated cv ids. Example: 1,2,3")
    parser.add_argument("--methods", default="vector,graph,hybrid", help="Subset of vector,graph,hybrid")
    parser.add_argument("--top_k", type=int, default=20, help="Top K per method per cv.")
    parser.add_argument(
        "--postgres_config",
        default=str(BASE_DIR / "config" / "db" / "postgres.yaml"),
        help="Postgres yaml config path.",
    )
    parser.add_argument(
        "--neo4j_config",
        default=str(BASE_DIR / "config" / "db" / "neo4j.yaml"),
        help="Neo4j yaml config path.",
    )
    parser.add_argument(
        "--retrieval_config",
        default=str(BASE_DIR / "config" / "rag" / "retrieval.yaml"),
        help="Retrieval yaml config path.",
    )
    parser.add_argument(
        "--embedding_config",
        default=str(BASE_DIR / "config" / "model" / "embedding.yaml"),
        help="Embedding yaml config path.",
    )
    parser.add_argument(
        "--hybrid_config",
        default=str(BASE_DIR / "config" / "recommendation" / "hybrid.yaml"),
        help="Hybrid yaml config path.",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "artifacts" / "evaluation" / "predictions.csv"),
        help="Output predictions csv path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv_ids = _parse_csv_ints(args.cv_ids)
    methods = _parse_methods(args.methods)
    if not cv_ids:
        raise ValueError("No cv_ids provided.")
    if args.top_k <= 0:
        raise ValueError("top_k must be > 0.")

    postgres_cfg = Path(args.postgres_config)
    neo4j_cfg = Path(args.neo4j_config)
    retrieval_cfg = Path(args.retrieval_config)
    embedding_cfg = Path(args.embedding_config)
    hybrid_cfg = Path(args.hybrid_config)

    cv_info = _load_cv_info(postgres_cfg, cv_ids)
    job_family_map = _load_job_family_map(postgres_cfg)

    rows: list[dict] = []
    for cv_id in cv_ids:
        question = _default_question(cv_info.get(cv_id, {}).get("target_role", ""))
        for method in methods:
            rows.extend(
                _rows_for_method(
                    method=method,
                    cv_id=cv_id,
                    question=question,
                    top_k=args.top_k,
                    postgres_cfg=postgres_cfg,
                    retrieval_cfg=retrieval_cfg,
                    embedding_cfg=embedding_cfg,
                    neo4j_cfg=neo4j_cfg,
                    hybrid_cfg=hybrid_cfg,
                )
            )

    if not rows:
        raise RuntimeError("No prediction rows generated.")

    df = pd.DataFrame(rows)
    df["job_family"] = df["job_id"].map(lambda x: job_family_map.get(int(x), ""))
    df = df.sort_values(["method", "cv_id", "rank"], ascending=[True, True, True]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[DONE] Saved predictions: {out_path}")
    print(f"rows={len(df)}, methods={sorted(df['method'].unique().tolist())}, cvs={sorted(df['cv_id'].unique().tolist())}")


if __name__ == "__main__":
    main()

