from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if ext == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        raise ValueError(f"Unsupported JSON format in: {p}")
    raise ValueError(f"Unsupported file extension: {ext}. Use csv/parquet/jsonl/json")


def parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
            if isinstance(parsed, dict):
                return [str(x).strip() for x in parsed.values() if str(x).strip()]
        except Exception:
            pass
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_token_set(values: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for v in values:
        t = normalize_text(v)
        if t:
            out.add(t)
    return out


def precision_recall_f1(pred: set[str], truth: set[str]) -> tuple[float, float, float]:
    if not pred and not truth:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals)) / float(len(vals)) if vals else 0.0


def percentile(values: Iterable[float], p: float) -> float:
    vals = sorted(float(x) for x in values)
    if not vals:
        return 0.0
    if p <= 0:
        return vals[0]
    if p >= 100:
        return vals[-1]
    rank = (len(vals) - 1) * (p / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return vals[lo]
    frac = rank - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def save_outputs(summary: pd.DataFrame, output: str | Path, details: pd.DataFrame | None = None) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".parquet", ".pq"}:
        summary.to_parquet(out, index=False)
    else:
        summary.to_csv(out, index=False, encoding="utf-8")
    if details is not None:
        details_out = out.with_name(f"{out.stem}_details{out.suffix or '.csv'}")
        if details_out.suffix.lower() in {".parquet", ".pq"}:
            details.to_parquet(details_out, index=False)
        else:
            details.to_csv(details_out, index=False, encoding="utf-8")

