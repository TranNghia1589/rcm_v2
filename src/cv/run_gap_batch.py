from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.matching.gap_analysis import ROLE_PROFILE_PATH, analyze_cv_against_roles, load_json


def _safe_cv_key(record: dict[str, Any]) -> str:
    for key in ["cv_key", "cv_id", "file_name", "source_path"]:
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return f"cv_{abs(hash(json.dumps(record, ensure_ascii=False))) % 10_000_000}"


def _safe_stem(text: str) -> str:
    raw = Path(text).stem if "." in text else text
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def _load_records(path: Path) -> list[dict[str, Any]]:
    def _decode_json_like(v: Any) -> Any:
        if not isinstance(v, str):
            return v
        t = v.strip()
        if not t:
            return v
        if (t.startswith("[") and t.endswith("]")) or (t.startswith("{") and t.endswith("}")):
            try:
                return json.loads(t)
            except Exception:
                return v
        return v

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        rows = [dict(x) for x in df.to_dict(orient="records")]
        return [{k: _decode_json_like(v) for k, v in row.items()} for row in rows]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(x) for x in data]
        if isinstance(data, dict):
            return [data]
    raise ValueError(f"Unsupported cv dataset format: {path}")


def _records_to_parquet_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    flat_rows: list[dict[str, Any]] = []
    for row in records:
        out: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (dict, list)):
                out[k] = json.dumps(v, ensure_ascii=False)
            else:
                out[k] = v
        flat_rows.append(out)
    return pd.DataFrame(flat_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gap analysis for all CV records in a dataset.")
    parser.add_argument(
        "--cv_dataset",
        default=str(BASE_DIR / "data" / "processed" / "cv_extracted" / "cv_extracted_dataset.parquet"),
        help="Input CV dataset (.parquet/.jsonl/.json).",
    )
    parser.add_argument(
        "--role_profiles",
        default=str(ROLE_PROFILE_PATH),
        help="Path to role_profiles.json.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(BASE_DIR / "data" / "processed" / "cv_gap_reports"),
        help="Output directory for per-cv gap json files.",
    )
    parser.add_argument(
        "--aggregate_jsonl",
        default=str(BASE_DIR / "data" / "processed" / "cv_gap_reports" / "cv_gap_dataset.jsonl"),
        help="Aggregate jsonl output path.",
    )
    parser.add_argument(
        "--aggregate_parquet",
        default=str(BASE_DIR / "data" / "processed" / "cv_gap_reports" / "cv_gap_dataset.parquet"),
        help="Aggregate parquet output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv_dataset = Path(args.cv_dataset)
    output_dir = Path(args.output_dir)
    aggregate_jsonl = Path(args.aggregate_jsonl)
    aggregate_parquet = Path(args.aggregate_parquet)

    if not cv_dataset.exists():
        raise FileNotFoundError(f"CV dataset not found: {cv_dataset}")

    role_profiles = load_json(Path(args.role_profiles))
    records = _load_records(cv_dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_jsonl.parent.mkdir(parents=True, exist_ok=True)
    aggregate_parquet.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for record in records:
        gap = analyze_cv_against_roles(record, role_profiles)
        cv_key = _safe_cv_key(record)
        gap["cv_key"] = cv_key
        gap["file_name"] = str(record.get("file_name", ""))
        out_path = output_dir / f"{_safe_stem(cv_key)}_gap.json"
        out_path.write_text(json.dumps(gap, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(gap)

    with open(aggregate_jsonl, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if results:
        _records_to_parquet_df(results).to_parquet(aggregate_parquet, index=False)

    summary = {
        "cv_dataset": str(cv_dataset.resolve()),
        "output_dir": str(output_dir.resolve()),
        "aggregate_jsonl": str(aggregate_jsonl.resolve()),
        "aggregate_parquet": str(aggregate_parquet.resolve()),
        "total_records": len(records),
        "gap_reports_written": len(results),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
