# Architecture Review (Current -> Unified)

## Hiện trạng rời rạc
- Dữ liệu và artifacts nằm ở nhiều nơi: `data_topcv/`, `data_processed/`, `outputs_preprocessing_v3/`, `Resume_recommendation/career_chatbot/data/`.
- Pipeline preprocess và pipeline chatbot/matching tách folder, chưa có một root project thống nhất.
- Module chatbot phụ thuộc kết quả matching qua file JSON, nhưng không có workflow chuẩn hóa chung theo một thư mục.
- Tồn tại nhiều notebook bản sao (`preprocessing copy*.ipynb`) gây khó xác định bản chuẩn.
- Có thành phần nặng không trực tiếp thuộc domain lõi (ví dụ cây `llama.cpp`) làm tăng độ nhiễu kiến trúc.

## Kiến trúc unified đã dựng
- Tạo project mới độc lập: `job_matching_cv_chatbot_unified/`
- Chuẩn hóa theo luồng dữ liệu:
  1. `data/raw` -> dữ liệu nguồn (jobs + cv)
  2. `src/pipelines` + `src/data_processing` -> chuẩn hóa dữ liệu job
  3. `artifacts/matching` -> embedding/index/parquet phục vụ matching
  4. `src/cv_processing` -> trích xuất CV
  5. `src/matching` -> gap analysis
  6. `src/chatbot` -> tư vấn dựa trên gap result
  7. `src/evaluation` -> test/eval scenario

## Mapping tài nguyên đã copy
- `Resume_recommendation/career_chatbot/src/*` -> `src/*`
- `preprocess/preprocessing.py` -> `src/pipelines/preprocessing.py`
- `preprocess/*.ipynb` -> `notebooks/preprocessing/*`
- `outputs_preprocessing_v3/*` -> `artifacts/matching/*`
- `Resume_recommendation/career_chatbot/data/skill_catalog.json` -> `data/skill_catalog.json` và `data/reference/skill_catalog.json`
- `Resume_recommendation/career_chatbot/data/role_profiles/role_profiles.json` -> `data/role_profiles/role_profiles.json` và `data/reference/role_profiles.json`
- `Resume_recommendation/career_chatbot/data/processed/*.json` -> `data/processed/*.json`
- dữ liệu mẫu jobs/cv -> `data/raw/jobs`, `data/raw/cv_samples`

## Đề xuất bước tiếp theo
- Chuẩn hóa entrypoint pipeline (ví dụ `scripts/run_pipeline.ps1`) để chạy end-to-end bằng một lệnh.
- Tách cấu hình đường dẫn/model sang `configs/*.yaml` thay vì hard-code trong module.
- Chuẩn hóa encoding tiếng Việt trong file script (UTF-8) để tránh lỗi hiển thị.
- Viết test tự động cho các chức năng:
  - trích xuất skill từ CV
  - scoring role matching
  - intent classification cho chatbot
