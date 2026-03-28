from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.cv.extract_cv_info import extract_cv_info


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def _find_cv_files(input_dir: Path) -> list[Path]:
    exts = {".pdf", ".docx", ".txt"}
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: str(p).lower())
    return files


def _records_to_parquet_df(records: list[dict]) -> pd.DataFrame:
    flat_rows: list[dict] = []
    for row in records:
        out: dict = {}
        for k, v in row.items():
            if isinstance(v, (dict, list)):
                out[k] = json.dumps(v, ensure_ascii=False)
            else:
                out[k] = v
        flat_rows.append(out)
    return pd.DataFrame(flat_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch extract CV structured data from a folder.")
    parser.add_argument(
        "--input_dir",
        default=str(BASE_DIR / "data" / "raw" / "cv_samples" / "INFORMATION-TECHNOLOGY"),
        help="Folder containing CV files (.pdf/.docx/.txt).",
    )
    parser.add_argument(
        "--output_dir",
        default=str(BASE_DIR / "data" / "processed" / "cv_extracted"),
        help="Folder to save per-CV JSON files.",
    )
    parser.add_argument(
        "--aggregate_jsonl",
        default=str(BASE_DIR / "data" / "processed" / "cv_extracted" / "cv_extracted_dataset.jsonl"),
        help="Path to aggregated jsonl file.",
    )
    parser.add_argument(
        "--aggregate_parquet",
        default=str(BASE_DIR / "data" / "processed" / "cv_extracted" / "cv_extracted_dataset.parquet"),
        help="Path to aggregated parquet file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    aggregate_jsonl = Path(args.aggregate_jsonl)
    aggregate_parquet = Path(args.aggregate_parquet)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_jsonl.parent.mkdir(parents=True, exist_ok=True)
    aggregate_parquet.parent.mkdir(parents=True, exist_ok=True)

    cv_files = _find_cv_files(input_dir)
    if not cv_files:
        print(f"No CV files found in: {input_dir}")
        return

    records: list[dict] = []
    failed: list[dict[str, str]] = []

    for cv_path in cv_files:
        try:
            record = extract_cv_info(str(cv_path))
            records.append(record)

            out_name = f"{_safe_stem(cv_path)}.json"
            out_path = output_dir / out_name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            failed.append({"file": str(cv_path), "error": str(exc)})

    with open(aggregate_jsonl, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if records:
        _records_to_parquet_df(records).to_parquet(aggregate_parquet, index=False)

    summary = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "aggregate_jsonl": str(aggregate_jsonl.resolve()),
        "aggregate_parquet": str(aggregate_parquet.resolve()),
        "total_files": len(cv_files),
        "success_count": len(records),
        "failed_count": len(failed),
    }
    if failed:
        summary["failed"] = failed[:20]

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
