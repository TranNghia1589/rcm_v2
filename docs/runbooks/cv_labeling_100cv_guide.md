# CV Labeling Guide - 100 CV Pilot

## Goal
Gan nhan 100 CV dau tien theo `Rubric v1 Final`, dam bao du chat luong cho baseline hybrid scoring.

## Inputs
- Rubric: `docs/data_dictionary/cv_scoring_rubric_v1_final.md`
- Template: `data/reference/label_templates/cv_labels_template.csv`
- Extracted CV dataset:
  - `data/processed/cv_extracted/cv_extracted_dataset.parquet`
  - hoac `data/processed/cv_extracted/cv_extracted_dataset.jsonl`

## Target output
- File labels pilot:
  - `data/reference/review/cv_labels_pilot_100.csv`

## Sampling policy (100 CV)
1. Chia theo role muc tieu de tranh lech mau:
- Data Analyst: 25
- Data Engineer: 20
- Data Scientist: 20
- AI Engineer: 15
- Software/Backend/Frontend/Fullstack: 20

2. Trong moi role, co du 3 nhom chat luong:
- Nhom manh
- Nhom trung binh
- Nhom yeu

## How to fill each row
Bat buoc dien:
- `cv_id`
- `role`
- `final_label_score`
- `grade`
- 6 cot `subscore_*`
- `label_source`
- `labeler_id`
- `label_confidence`
- `labeled_at`

Nen dien:
- `missing_skills` (json list)
- `strengths` (json list)
- `rationale_text` (1-3 cau)
- `notes`

## Scoring checklist per CV
1. Doc nhanh profile + role muc tieu.
2. Cham 6 subscore theo rubric.
3. Cong tong ra `final_label_score`.
4. Map `grade` theo bang A/B/C/D/E.
5. Dien `missing_skills`, `strengths`, `rationale_text`.

## Validation command
Chay truoc khi import DB:

```powershell
python deploy/scripts/evaluation/validate_cv_labels.py `
  --labels data/reference/review/cv_labels_pilot_100.csv `
  --cv_extracted data/processed/cv_extracted/cv_extracted_dataset.parquet `
  --score_tolerance 1.0
```

## Acceptance criteria for pilot batch
1. `rows = 100`
2. Khong loi validation hard-error
3. Moi dong co `rationale_text` khong rong
4. Phan bo role khong lech qua muc sampling policy

## Suggested review workflow
1. Labeler A cham 100 CV (`label_source=human_single`)
2. Chon 20 CV overlap cho labeler B cham lai
3. Neu lech nhieu, hop committee de chot rubric interpretation
4. Dong bang file pilot sau review
