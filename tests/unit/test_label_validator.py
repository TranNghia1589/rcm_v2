from __future__ import annotations

from pathlib import Path

from src.evaluation.cv_scoring.label_validator import validate_cv_labels


def test_validate_cv_labels_ok(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "\n".join(
            [
                "cv_id,role,final_label_score,grade,subscore_skill,subscore_experience,subscore_achievement,subscore_education,subscore_formatting,subscore_keywords,label_source,labeler_id,label_confidence,labeled_at",
                "cv_001,Data Analyst,75,B,24,18,12,8,9,4,human_single,r1,0.9,2026-04-01T00:00:00Z",
            ]
        ),
        encoding="utf-8",
    )
    extracted = tmp_path / "cv_extracted.csv"
    extracted.write_text("cv_id\ncv_001\n", encoding="utf-8")

    report = validate_cv_labels(labels_path=labels, extracted_dataset_path=extracted)
    assert report["ok"] is True
    assert report["errors"] == []


def test_validate_cv_labels_fail_grade_and_bounds(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "\n".join(
            [
                "cv_id,role,final_label_score,grade,subscore_skill,subscore_experience,subscore_achievement,subscore_education,subscore_formatting,subscore_keywords,label_source,labeler_id,label_confidence,labeled_at",
                "cv_001,Data Analyst,75,A,35,18,12,8,9,4,human_single,r1,1.2,2026-04-01T00:00:00Z",
            ]
        ),
        encoding="utf-8",
    )
    report = validate_cv_labels(labels_path=labels)
    assert report["ok"] is False
    assert any("grade mismatch" in e for e in report["errors"])
    assert any("subscore_skill" in e for e in report["errors"])
    assert any("label_confidence" in e for e in report["errors"])


def test_validate_cv_labels_fail_missing_cv_id_reference(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "\n".join(
            [
                "cv_id,role,final_label_score,grade,subscore_skill,subscore_experience,subscore_achievement,subscore_education,subscore_formatting,subscore_keywords,label_source,labeler_id,label_confidence,labeled_at",
                "cv_missing,Data Engineer,60,C,18,14,9,7,8,4,committee,r2,0.8,2026-04-01T00:00:00Z",
            ]
        ),
        encoding="utf-8",
    )
    extracted = tmp_path / "cv_extracted.csv"
    extracted.write_text("cv_id\ncv_001\n", encoding="utf-8")
    report = validate_cv_labels(labels_path=labels, extracted_dataset_path=extracted)
    assert report["ok"] is False
    assert any("not found in extracted dataset" in e for e in report["errors"])
