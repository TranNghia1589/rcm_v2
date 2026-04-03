# CV Scoring Rubric v1 Final

Status: Final for phase-1 labeling  
Effective date: 2026-04-03

## Scope
Rubric nay duoc dung de gan nhan `cv_labels` cho batch pilot va danh gia calibration/stability.

## Scoring dimensions
Tong diem: 100

1. `subscore_skill` (0..30)
- Muc do khop ky nang voi role muc tieu.
- Uu tien ky nang must-have truoc nice-to-have.

2. `subscore_experience` (0..25)
- So nam kinh nghiem + do lien quan domain.
- Relevance quan trong hon tong so nam.

3. `subscore_achievement` (0..20)
- Co bang chung impact/metric cu the.
- Uu tien ket qua do duoc (%, $, KPI).

4. `subscore_education` (0..10)
- Hoc van/chung chi lien quan role.
- Co gia tri bo tro, khong thay the skill/experience.

5. `subscore_formatting` (0..10)
- CV ro rang, de doc, cau truc hop ly.
- Muc tieu la trinh bay minh bach, ATS friendly.

6. `subscore_keywords` (0..5)
- Keyword quan trong xuat hien hop ly theo role.
- Khong cham diem cao cho viec nhoi keyword khong co context.

## Final score
`final_label_score = subscore_skill + subscore_experience + subscore_achievement + subscore_education + subscore_formatting + subscore_keywords`

Sai so cho phep khi validate: `<= 1.0`.

## Grade mapping
- `A`: 85..100
- `B`: 70..<85
- `C`: 55..<70
- `D`: 40..<55
- `E`: <40

## Role guidance (ap dung cung mot bo trong so)
Trong so giu nguyen cho tat ca role trong phase-1. Khac biet nam o cach danh gia "match" theo role:

1. Data Analyst
- Uu tien: SQL, BI (Power BI/Tableau), thong ke co ban, business context.

2. Data Engineer
- Uu tien: ETL/pipeline, data warehouse, SQL nang cao, Spark/Airflow/cloud data stack.

3. Data Scientist
- Uu tien: modeling, experiment/evaluation, feature engineering, python stack.

4. AI Engineer
- Uu tien: ML engineering, deployment/inference, llm/rag stack neu role yeu cau.

5. Software/Backend/Frontend/Fullstack
- Uu tien: stack ky thuat theo role + bang chung du an san xuat.

## Labeling metadata requirements
- `label_source`: `human_single|human_double|committee|rubric_bootstrap`
- `labeler_id`: dinh danh nguoi cham
- `label_confidence`: 0..1
- `rationale_text`: bat buoc ghi ngan gon vi sao cham diem

## Governance
- Khong dung CV user production de train model trong phase nay.
- Chi dung tap `cv_labels` noi bo da review.
- `rubric_bootstrap` chi la weak label, khong thay the nhan human chuan.
