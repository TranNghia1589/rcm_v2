from __future__ import annotations

import math
from typing import Iterable


def precision_at_k(relevance: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = relevance[:k]
    if not top_k:
        return 0.0
    return float(sum(1 for r in top_k if r > 0)) / float(k)


def recall_at_k(relevance: list[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0 or k <= 0:
        return 0.0
    top_k = relevance[:k]
    hit = sum(1 for r in top_k if r > 0)
    return float(hit) / float(total_relevant)


def hit_rate_at_k(relevance: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return 1.0 if any(r > 0 for r in relevance[:k]) else 0.0


def mrr_at_k(relevance: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    for idx, rel in enumerate(relevance[:k], start=1):
        if rel > 0:
            return 1.0 / float(idx)
    return 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    if k <= 0:
        return 0.0
    out = 0.0
    for i, g in enumerate(gains[:k], start=1):
        out += (2.0**float(g) - 1.0) / math.log2(i + 1.0)
    return out


def ndcg_at_k(gains: list[float], k: int) -> float:
    if k <= 0:
        return 0.0
    actual = dcg_at_k(gains, k)
    ideal = dcg_at_k(sorted(gains, reverse=True), k)
    if ideal <= 0.0:
        return 0.0
    return actual / ideal


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals)) / float(len(vals))
