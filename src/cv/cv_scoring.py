from __future__ import annotations

# Backward-compatible bridge. The active scoring implementation lives in src/scoring.
from src.scoring.cv_scoring import score_cv_record

__all__ = ["score_cv_record"]
