# CV Scoring Data Contract

## Scope
Data contract nay quy dinh du lieu dau vao cho bai toan cham diem CV theo huong hybrid:
- Rule/Rubric score (explicit)
- Model score (se bo sung o phase sau)

Version: `final`

Rubric chinh thuc dung cho gan nhan:
- `docs/data_dictionary/cv_scoring_rubric_v1_final.md`

## Canonical datasets
1. `cv_raw`
   - Nguon: `data/raw/cv_samples/INFORMATION-TECHNOLOGY/*.pdf`
2. `cv_extracted`
   - Nguon du kien sau pipeline: `data/processed/cv_extracted/cv_extracted_dataset.parquet`
3. `cv_labels`
   - Nguon nhan thu cong theo template:
     - `data/reference/label_templates/cv_labels_template.csv`

## Table: cv_raw
Purpose: luu dau vao goc de truy vet va tai parse.

Required fields:
- `cv_id` (string): dinh danh CV
- `file_name` (string)
- `source_path` (string)
- `file_hash` (string): md5/sha256
- `file_type` (string): `pdf|docx|txt`
- `ingested_at` (datetime iso8601)

## Table: cv_extracted
Purpose: du lieu cau truc phuc vu scoring/gap/recommend.

Required fields:
- `cv_id` (string)
- `skills` (json list[string])
- `experience_years` (string or numeric)
- `projects` (json list[string])
- `target_role` (string)
- `schema_version` (string)
- `metadata` (json object)

## Table: cv_labels
Purpose: ground-truth labels cho calibration/train/eval.

Required fields:
- `cv_id` (string)
- `role` (string)
- `final_label_score` (float, range 0..100)
- `grade` (enum `A|B|C|D|E`)
- `subscore_skill` (float, range 0..30)
- `subscore_experience` (float, range 0..25)
- `subscore_achievement` (float, range 0..20)
- `subscore_education` (float, range 0..10)
- `subscore_formatting` (float, range 0..10)
- `subscore_keywords` (float, range 0..5)
- `label_source` (enum `human_single|human_double|committee|rubric_bootstrap`)
- `labeler_id` (string)
- `label_confidence` (float, range 0..1)
- `labeled_at` (datetime iso8601)

Optional fields:
- `missing_skills` (json list[string])
- `strengths` (json list[string])
- `rationale_text` (string)
- `notes` (string)

## Validation rules
1. Khong duoc thieu cot bat buoc.
2. `cv_id` phai unique trong cung 1 file labels.
3. Gia tri diem nam trong mien hop le theo tung subscore.
4. `grade` phai khop voi `final_label_score`:
   - `A`: >= 85
   - `B`: >= 70 va < 85
   - `C`: >= 55 va < 70
   - `D`: >= 40 va < 55
   - `E`: < 40
5. Tong 6 subscore phai xap xi `final_label_score` (sai so <= 1.0).
6. Neu co `cv_extracted` canonical thi `cv_id` trong labels phai ton tai trong `cv_extracted`.

## Governance notes
- Chua su dung CV user production de train model.
- `cv_labels` trong phase nay la tap nhan noi bo da duoc review.
- `label_source=rubric_bootstrap` chi dung weak-label baseline, khong dung lam nhan human chuan.
