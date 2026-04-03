from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "cv_id",
    "role",
    "final_label_score",
    "grade",
    "subscore_skill",
    "subscore_experience",
    "subscore_achievement",
    "subscore_education",
    "subscore_formatting",
    "subscore_keywords",
    "label_source",
    "labeler_id",
    "label_confidence",
    "labeled_at",
]

LABEL_SOURCES = {"human_single", "human_double", "committee", "rubric_bootstrap"}

SUBSCORE_BOUNDS = {
    "subscore_skill": (0.0, 30.0),
    "subscore_experience": (0.0, 25.0),
    "subscore_achievement": (0.0, 20.0),
    "subscore_education": (0.0, 10.0),
    "subscore_formatting": (0.0, 10.0),
    "subscore_keywords": (0.0, 5.0),
}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        return float(txt)
    except Exception:
        return None


def _grade_from_score(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "E"


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                txt = line.strip()
                if txt:
                    rows.append(json.loads(txt))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported file format: {path}")


def _load_existing_cv_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = _load_table(path)
    if "cv_id" not in df.columns:
        return set()
    ids: set[str] = set()
    for value in df["cv_id"].tolist():
        txt = str(value).strip()
        if txt:
            ids.add(txt)
    return ids


def validate_cv_labels(
    *,
    labels_path: str | Path,
    extracted_dataset_path: str | Path | None = None,
    score_tolerance: float = 1.0,
) -> dict[str, Any]:
    label_path = Path(labels_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Labels file not found: {label_path}")

    df = _load_table(label_path)
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return {
            "ok": False,
            "rows": int(len(df)),
            "errors": errors,
            "warnings": warnings,
        }

    # Duplicate cv_id
    dup_ids = (
        df["cv_id"].astype(str).str.strip().value_counts()
        .loc[lambda s: s > 1]
        .index.tolist()
    )
    if dup_ids:
        errors.append(f"Duplicate cv_id detected: {dup_ids[:20]}")

    # Referential integrity
    if extracted_dataset_path:
        extracted_path = Path(extracted_dataset_path)
        if extracted_path.exists():
            existing_ids = _load_existing_cv_ids(extracted_path)
            if existing_ids:
                missing_ref = []
                for cv_id in df["cv_id"].astype(str).str.strip().tolist():
                    if cv_id and cv_id not in existing_ids:
                        missing_ref.append(cv_id)
                if missing_ref:
                    errors.append(f"cv_id not found in extracted dataset: {missing_ref[:20]}")
            else:
                warnings.append("Extracted dataset does not contain `cv_id` column; skipped referential check.")
        else:
            warnings.append(f"Extracted dataset path not found; skipped referential check: {extracted_path}")

    for idx, row in df.iterrows():
        row_no = int(idx) + 2  # header is line 1
        cv_id = str(row.get("cv_id", "")).strip()

        if not cv_id:
            errors.append(f"row {row_no}: cv_id is empty.")
            continue

        role = str(row.get("role", "")).strip()
        if not role:
            errors.append(f"row {row_no}: role is empty.")

        score = _to_float(row.get("final_label_score"))
        if score is None or score < 0 or score > 100:
            errors.append(f"row {row_no}: final_label_score must be in [0, 100].")
            continue

        grade = str(row.get("grade", "")).strip().upper()
        expected_grade = _grade_from_score(score)
        if grade != expected_grade:
            errors.append(
                f"row {row_no}: grade mismatch (got={grade}, expected={expected_grade}, score={score})."
            )

        source = str(row.get("label_source", "")).strip()
        if source not in LABEL_SOURCES:
            errors.append(f"row {row_no}: invalid label_source={source}.")

        conf = _to_float(row.get("label_confidence"))
        if conf is None or conf < 0 or conf > 1:
            errors.append(f"row {row_no}: label_confidence must be in [0, 1].")

        subsum = 0.0
        sub_ok = True
        for col, (lo, hi) in SUBSCORE_BOUNDS.items():
            val = _to_float(row.get(col))
            if val is None:
                errors.append(f"row {row_no}: {col} is not numeric.")
                sub_ok = False
                continue
            if val < lo or val > hi:
                errors.append(f"row {row_no}: {col} must be in [{lo}, {hi}].")
                sub_ok = False
            subsum += float(val)

        if sub_ok and abs(subsum - score) > score_tolerance:
            errors.append(
                f"row {row_no}: subscore sum mismatch (sum={subsum:.2f}, final={score:.2f}, tol={score_tolerance})."
            )

    return {
        "ok": len(errors) == 0,
        "rows": int(len(df)),
        "errors": errors,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cv_labels dataset against data contract.")
    parser.add_argument("--labels", required=True, help="Path to cv_labels CSV/JSONL/Parquet.")
    parser.add_argument(
        "--cv_extracted",
        default="",
        help="Optional path to cv_extracted dataset for cv_id referential check.",
    )
    parser.add_argument(
        "--score_tolerance",
        type=float,
        default=1.0,
        help="Allowed absolute diff between subscore sum and final score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_cv_labels(
        labels_path=args.labels,
        extracted_dataset_path=args.cv_extracted.strip() or None,
        score_tolerance=float(args.score_tolerance),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
