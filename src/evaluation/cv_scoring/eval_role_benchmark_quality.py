from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import mean, save_outputs
from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig


def _to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        import json

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return []


def evaluate_role_benchmark_quality(postgres_config: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = PostgresConfig.from_yaml(postgres_config)
    with PostgresClient(cfg) as client:
        benchmark_rows = client.fetch_all(
            """
            SELECT role_name, top_market_skills, top_profile_skills, generated_at
            FROM role_skill_benchmarks
            """
        )
        role_rows = client.fetch_all(
            """
            SELECT DISTINCT COALESCE(target_role, '')
            FROM cv_profiles
            WHERE COALESCE(target_role, '') <> ''
            """
        )

    target_roles = {str(r[0]).strip() for r in role_rows if str(r[0]).strip()}
    detail_rows = []
    now = datetime.now(timezone.utc)
    for role_name, market_skills, profile_skills, generated_at in benchmark_rows:
        market = _to_list(market_skills)
        profile = _to_list(profile_skills)
        overlap = len(set(x.lower() for x in market) & set(x.lower() for x in profile))
        union = len(set(x.lower() for x in market) | set(x.lower() for x in profile))
        jaccard = (overlap / union) if union > 0 else 0.0
        age_days = 0.0
        if generated_at is not None:
            try:
                dt = generated_at if isinstance(generated_at, datetime) else datetime.fromisoformat(str(generated_at))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
            except Exception:
                age_days = 0.0
        detail_rows.append(
            {
                "role_name": str(role_name),
                "market_skill_count": float(len(market)),
                "profile_skill_count": float(len(profile)),
                "market_profile_jaccard": jaccard,
                "age_days": age_days,
                "is_target_role_covered": 1.0 if str(role_name) in target_roles else 0.0,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        summary_df = pd.DataFrame(
            [
                {
                    "num_benchmark_roles": 0,
                    "target_roles_from_cv": int(len(target_roles)),
                    "role_coverage_rate": 0.0,
                    "avg_market_skill_count": 0.0,
                    "avg_profile_skill_count": 0.0,
                    "avg_market_profile_jaccard": 0.0,
                    "avg_benchmark_age_days": 0.0,
                }
            ]
        )
        return summary_df, detail_df

    summary_df = pd.DataFrame(
        [
            {
                "num_benchmark_roles": int(len(detail_df)),
                "target_roles_from_cv": int(len(target_roles)),
                "role_coverage_rate": mean(detail_df["is_target_role_covered"].tolist()),
                "avg_market_skill_count": mean(detail_df["market_skill_count"].tolist()),
                "avg_profile_skill_count": mean(detail_df["profile_skill_count"].tolist()),
                "avg_market_profile_jaccard": mean(detail_df["market_profile_jaccard"].tolist()),
                "avg_benchmark_age_days": mean(detail_df["age_days"].tolist()),
            }
        ]
    )
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate quality/completeness of role benchmark cache.")
    parser.add_argument("--postgres_config", default="configs/db/postgres.yaml")
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/role_benchmark_quality.csv",
        help="Output summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df, detail_df = evaluate_role_benchmark_quality(args.postgres_config)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] Role benchmark quality evaluation")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()

