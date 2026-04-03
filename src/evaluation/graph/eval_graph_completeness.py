from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import mean, save_outputs
from src.utils.infrastructure.db.neo4j_client import Neo4jClient, Neo4jConfig
from src.utils.infrastructure.db.postgres_client import PostgresClient, PostgresConfig


def _pg_count(client: PostgresClient, query: str) -> int:
    row = client.fetch_one(query)
    return int(row[0]) if row else 0


def _neo_count(client: Neo4jClient, query: str) -> int:
    rows = client.run(query)
    if not rows:
        return 0
    return int(rows[0].get("cnt", 0))


def evaluate_graph_completeness(postgres_config: str | Path, neo4j_config: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pg_cfg = PostgresConfig.from_yaml(postgres_config)
    neo_cfg = Neo4jConfig.from_yaml(neo4j_config)

    checks: list[dict[str, Any]] = []
    with PostgresClient(pg_cfg) as pg, Neo4jClient(neo_cfg) as neo:
        specs = [
            ("users", "SELECT COUNT(*) FROM users", "MATCH (n:User) RETURN COUNT(n) AS cnt"),
            ("cvs", "SELECT COUNT(*) FROM cv_profiles", "MATCH (n:CV) RETURN COUNT(n) AS cnt"),
            ("skills", "SELECT COUNT(*) FROM skills", "MATCH (n:Skill) RETURN COUNT(n) AS cnt"),
            ("jobs", "SELECT COUNT(*) FROM jobs", "MATCH (n:Job) RETURN COUNT(n) AS cnt"),
            ("cv_skill_edges", "SELECT COUNT(*) FROM cv_skills", "MATCH ()-[r:HAS_SKILL]->() RETURN COUNT(r) AS cnt"),
            ("job_skill_edges", "SELECT COUNT(*) FROM job_skills", "MATCH ()-[r:REQUIRES_SKILL]->() RETURN COUNT(r) AS cnt"),
        ]
        for name, pg_q, neo_q in specs:
            pg_cnt = _pg_count(pg, pg_q)
            neo_cnt = _neo_count(neo, neo_q)
            ratio = (neo_cnt / pg_cnt) if pg_cnt > 0 else (1.0 if neo_cnt == 0 else 0.0)
            checks.append(
                {
                    "component": name,
                    "postgres_count": int(pg_cnt),
                    "neo4j_count": int(neo_cnt),
                    "completeness_ratio": float(ratio),
                }
            )

    detail_df = pd.DataFrame(checks)
    summary_df = pd.DataFrame(
        [
            {
                "components": int(len(detail_df)),
                "completeness_ratio_mean": mean(detail_df["completeness_ratio"].tolist()),
                "min_completeness_ratio": float(detail_df["completeness_ratio"].min()) if not detail_df.empty else 0.0,
            }
        ]
    )
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate graph completeness vs PostgreSQL source tables.")
    parser.add_argument("--postgres_config", default="config/db/postgres.yaml")
    parser.add_argument("--neo4j_config", default="config/db/neo4j.yaml")
    parser.add_argument(
        "--output",
        default="experiments/artifacts/evaluation/graph_completeness.csv",
        help="Output summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df, detail_df = evaluate_graph_completeness(args.postgres_config, args.neo4j_config)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] Graph completeness evaluation")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()

