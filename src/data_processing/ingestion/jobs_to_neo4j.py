from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.graph.etl import GraphETL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync PostgreSQL jobs/skills data into Neo4j graph.")
    parser.add_argument(
        "--postgres_config",
        default=str(BASE_DIR / "config" / "db" / "postgres.yaml"),
    )
    parser.add_argument(
        "--neo4j_config",
        default=str(BASE_DIR / "config" / "db" / "neo4j.yaml"),
    )
    parser.add_argument(
        "--reset_graph",
        action="store_true",
        help="Delete all nodes/rels before sync.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    etl = GraphETL(
        postgres_cfg_path=args.postgres_config,
        neo4j_cfg_path=args.neo4j_config,
    )
    stats = etl.run(reset_graph=args.reset_graph)
    print(
        "[DONE] "
        f"users={stats.users}, cvs={stats.cvs}, skills={stats.skills}, jobs={stats.jobs}, "
        f"cv_skill_links={stats.cv_skill_links}, job_skill_links={stats.job_skill_links}, "
        f"role_links={stats.role_links}, lack_skill_links={stats.lack_skill_links}, "
        f"education_links={stats.education_links}, experience_links={stats.experience_links}, "
        f"project_links={stats.project_links}, certification_links={stats.certification_links}, "
        f"language_links={stats.language_links}"
    )


if __name__ == "__main__":
    main()

