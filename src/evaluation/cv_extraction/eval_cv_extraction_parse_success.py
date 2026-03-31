from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import mean, parse_list, percentile, read_table, save_outputs


def _list_input_files(input_dir: Path) -> list[Path]:
    exts = {".pdf", ".docx", ".txt"}
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: str(p).lower())


def _match_key_columns(df: pd.DataFrame) -> pd.Series:
    if "source_path" in df.columns:
        return df["source_path"].astype(str).map(lambda x: Path(x).name.lower())
    if "file_name" in df.columns:
        return df["file_name"].astype(str).str.lower()
    if "cv_key" in df.columns:
        return df["cv_key"].astype(str).str.lower()
    return pd.Series([""] * len(df))


def evaluate_parse_success(
    extracted_df: pd.DataFrame,
    *,
    input_files: list[Path] | None = None,
    required_fields: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if required_fields is None:
        required_fields = ["skills", "target_role", "experience_years", "projects"]

    df = extracted_df.copy()
    df["_key"] = _match_key_columns(df)
    df = df[df["_key"] != ""].copy()
    df = df.drop_duplicates(subset=["_key"], keep="last")

    parsed_keys = set(df["_key"].tolist())
    total_input = len(input_files) if input_files is not None else len(parsed_keys)
    input_keys = {p.name.lower() for p in input_files} if input_files is not None else parsed_keys
    success_keys = input_keys & parsed_keys

    details: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        key = str(row["_key"])
        if key not in input_keys:
            continue
        filled = 0
        for field in required_fields:
            value = row.get(field)
            if isinstance(value, str):
                vals = parse_list(value)
                ok = bool(vals) if value.strip().startswith("[") else bool(value.strip()) and value.strip().lower() != "unknown"
            elif isinstance(value, (list, tuple)):
                ok = len(value) > 0
            else:
                ok = value is not None and str(value).strip() != ""
            filled += 1 if ok else 0
        completeness = filled / len(required_fields) if required_fields else 0.0
        val_errors = parse_list((row.get("metadata") or ""))
        details.append(
            {
                "file_key": key,
                "parse_success": 1.0,
                "required_field_fill_rate": completeness,
                "has_validation_errors": 1.0 if "validation_errors" in str(row.get("metadata", "")) else 0.0,
                "skill_count": float(len(parse_list(row.get("skills")))),
                "project_count": float(len(parse_list(row.get("projects")))),
                "_unused": len(val_errors),
            }
        )

    for miss in sorted(input_keys - parsed_keys):
        details.append(
            {
                "file_key": miss,
                "parse_success": 0.0,
                "required_field_fill_rate": 0.0,
                "has_validation_errors": 0.0,
                "skill_count": 0.0,
                "project_count": 0.0,
                "_unused": 0,
            }
        )

    detail_df = pd.DataFrame(details)
    if detail_df.empty:
        raise ValueError("No records available to evaluate parse success.")

    summary_df = pd.DataFrame(
        [
            {
                "total_input_files": int(total_input),
                "parsed_records": int(len(success_keys)),
                "parse_success_rate": mean(detail_df["parse_success"].tolist()),
                "required_field_fill_rate_mean": mean(detail_df["required_field_fill_rate"].tolist()),
                "required_field_fill_rate_p50": percentile(detail_df["required_field_fill_rate"].tolist(), 50),
                "required_field_fill_rate_p95": percentile(detail_df["required_field_fill_rate"].tolist(), 95),
                "validation_error_rate": mean(detail_df["has_validation_errors"].tolist()),
                "avg_skill_count": mean(detail_df["skill_count"].tolist()),
                "avg_project_count": mean(detail_df["project_count"].tolist()),
            }
        ]
    )
    return summary_df, detail_df.drop(columns=["_unused"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CV extraction parse success/completeness.")
    parser.add_argument("--extracted", required=True, help="Extracted CV dataset (csv/parquet/jsonl/json).")
    parser.add_argument(
        "--input_dir",
        default="",
        help="Optional folder with raw CV files. If provided, parse success uses this denominator.",
    )
    parser.add_argument(
        "--required_fields",
        default="skills,target_role,experience_years,projects",
        help="Comma-separated required fields for completeness.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/cv_extraction_parse_success.csv",
        help="Output summary csv/parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extracted_df = read_table(args.extracted)
    input_files = None
    if args.input_dir.strip():
        input_files = _list_input_files(Path(args.input_dir))
    required_fields = [x.strip() for x in args.required_fields.split(",") if x.strip()]
    summary_df, detail_df = evaluate_parse_success(extracted_df, input_files=input_files, required_fields=required_fields)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] CV extraction parse success")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()
