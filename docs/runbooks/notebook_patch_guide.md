
# Hướng dẫn sửa notebook `preprocessing copy 3.ipynb` theo từng cell

Tài liệu này đi theo đúng thứ tự cell của notebook cũ để bạn thấy cell nào **giữ**, cell nào **sửa**, cell nào **bổ sung chức năng**.

## Tổng quan thay đổi lớn
- Giữ nguyên cấu trúc 5 phần lớn của notebook.
- Chuyển phần cuối từ **demo matching** sang **artifact preprocessing** cho:
  - recommender (`jobs_matching_ready_v2`)
  - chatbot summary (`jobs_chatbot_ready_v2`)
  - chatbot chunks / RAG (`jobs_chatbot_sections_v2`)
  - explainability (`job_skill_map_v2`, `role_skill_stats_v2`)
- Biến phần embedding thành **tuỳ chọn** thay vì chạy mặc định.

---

## Cell 0
**Trạng thái:** sửa  
**Làm gì:**
- thêm `html`, `warnings`
- thêm `NOTEBOOK_VERSION`, `RUN_EMBEDDING`, `RUN_SECTION_EMBEDDING`
- bỏ hardcode path một kiểu duy nhất, thay bằng `RAW_INPUT_CANDIDATES`
- tạo `OUTPUT_DIR`, `ARTIFACT_DIR`, `REPORT_DIR`
- in config đầu run

**Lý do sửa:** notebook cũ phụ thuộc đường dẫn máy chạy; bản mới portable hơn.

## Cell 1
**Trạng thái:** giữ  
**Làm gì:** giữ cell option hiển thị, chỉ đóng vai trò debug.

## Cell 2
**Trạng thái:** giữ  
**Làm gì:** giữ markdown tiêu đề phần 1.

## Cell 3
**Trạng thái:** sửa mạnh  
**Làm gì:**
- giữ các helper cũ
- bổ sung `get_series()` để merge cột an toàn khi thiếu cột
- bổ sung `save_table()` có fallback CSV nếu thiếu parquet engine
- nâng `remove_html()` để unescape entity
- thêm `deduplicate_text_lines()` để giảm lặp trong JD
- giữ 3 nhánh clean: `light`, `preserve_structure`, `strict`

**Lý do sửa:** đây là cell nền của cả notebook, cần đủ robust.

## Cell 4
**Trạng thái:** giữ, sửa nhẹ  
**Làm gì:** giữ logic load file nhưng bọc theo `RAW_INPUT_PATH` mới.

## Cell 5
**Trạng thái:** sửa  
**Làm gì:**
- vẫn giữ ý tưởng merge các cột semantic giống nhau
- thay `df.get()` trực tiếp bằng `get_series()`
- giữ output schema canonical 25 cột

**Lý do sửa:** tránh lỗi nếu file source khác thiếu cột.

## Cell 6
**Trạng thái:** giữ  
**Làm gì:** giữ markdown phần audit & clean.

## Cell 7
**Trạng thái:** giữ  
**Làm gì:** copy `df -> df_clean`.

## Cell 8
**Trạng thái:** sửa  
**Làm gì:**
- chuẩn hoá empty token ngay trong cell audit
- tính `language_type`
- thêm `dense_encoder_route` (`phobert` vs `multilingual`)
- tính audit report sau khi normalize empty

**Lý do sửa:** để language được dùng thật ở downstream và missing report đúng hơn.

## Cell 9
**Trạng thái:** sửa nhẹ  
**Làm gì:** preview lại các raw text columns sau normalize.

## Cell 10
**Trạng thái:** giữ  
**Làm gì:** giữ việc tạo `*_clean_light` và `*_clean_struct`.

## Cell 11
**Trạng thái:** giữ  
**Làm gì:** giữ việc tạo `*_clean_strict`.

## Cell 12
**Trạng thái:** giữ  
**Làm gì:** thống kê các cột clean vừa tạo.

## Cell 13
**Trạng thái:** giữ  
**Làm gì:** preview raw / cleaned.

## Cell 14
**Trạng thái:** giữ  
**Làm gì:** đo empty ratio sau clean.

## Cell 15
**Trạng thái:** sửa nhẹ  
**Làm gì:** thêm kiểm tra độ dài cho cả `*_clean_struct`.

## Cell 16
**Trạng thái:** sửa  
**Làm gì:** thay `to_parquet()` bằng `save_table()` để tự fallback CSV.

**Lý do sửa:** môi trường nào không có `pyarrow/fastparquet` vẫn chạy được.

## Cell 17
**Trạng thái:** giữ  
**Làm gì:** giữ markdown phần normalize metadata.

## Cell 18
**Trạng thái:** sửa mạnh  
**Làm gì:**
- thay `TITLE_NOISE_PATTERNS` bằng bộ rule mới cho title
- không xóa mù mọi nội dung trong ngoặc
- thêm `strip_bracket_noise()` để chỉ xóa bracket chứa noise
- mở rộng `JOB_FAMILY_RULES`
- sửa rule seniority theo thứ tự ưu tiên

**Lý do sửa:** fix lỗi `senior/cao cấp` bị nuốt và giảm `job_family=other`.

## Cell 19
**Trạng thái:** giữ, sửa nhẹ  
**Làm gì:** apply title normalization mới.

## Cell 20
**Trạng thái:** sửa  
**Làm gì:**
- thay `VIETNAM_CITIES` bằng `CITY_ALIAS_MAP`
- thêm alias HN/HCM/TP HCM...
- thêm `WORK_MODE_RULES`
- thêm `has_multi_location()`

## Cell 21
**Trạng thái:** sửa mạnh  
**Làm gì:**
- giữ `parse_working_address()`
- bổ sung `parse_deadline()`
- tách logic normalize location và deadline rõ ràng

## Cell 22
**Trạng thái:** sửa  
**Làm gì:**
- apply location fields
- apply `is_multi_location`
- apply `deadline_date`, `days_to_deadline`, `deadline_type`, `is_expired`

## Cell 23
**Trạng thái:** sửa  
**Làm gì:** preview distribution cho `location_norm`, `work_mode`, `deadline_type`.

## Cell 24
**Trạng thái:** sửa mạnh  
**Làm gì:**
- thay `_to_number()` bằng `parse_numeric_token()`
- fix bug USD có dấu phẩy `2,500`
- thêm `detect_salary_multiplier()`
- giữ normalize về `VND/month`

## Cell 25
**Trạng thái:** sửa  
**Làm gì:** thêm test case salary USD có dấu phẩy.

## Cell 26
**Trạng thái:** giữ, sửa nhẹ  
**Làm gì:** apply salary parse mới.

## Cell 27
**Trạng thái:** sửa  
**Làm gì:**
- parse thêm `6 tháng`, `1+ năm`
- tách `experience_type` tốt hơn
- giữ output min/max years

## Cell 28
**Trạng thái:** sửa nhẹ  
**Làm gì:** thêm examples cho month / plus-year.

## Cell 29
**Trạng thái:** giữ  
**Làm gì:** apply experience parse.

## Cell 30
**Trạng thái:** giữ  
**Làm gì:** giữ map education.

## Cell 31
**Trạng thái:** giữ  
**Làm gì:** apply education normalize.

## Cell 32
**Trạng thái:** sửa  
**Làm gì:** bỏ `remote/hybrid` khỏi `EMPLOYMENT_TYPE_MAP` vì đó là work mode, không phải contract type.

## Cell 33
**Trạng thái:** giữ  
**Làm gì:** apply employment type normalize.

## Cell 34
**Trạng thái:** sửa mạnh  
**Làm gì:**
- thay `JOB_LEVEL_MAP` bằng `JOB_LEVEL_RULES` có thứ tự ưu tiên
- ưu tiên `director/manager/lead/senior` trước `junior`

## Cell 35
**Trạng thái:** giữ, sửa nhẹ  
**Làm gì:** apply job level normalize mới và preview thêm `seniority_from_title`.

## Cell 36
**Trạng thái:** sửa  
**Làm gì:** metadata preview có thêm `deadline`, `is_multi_location`.

## Cell 37
**Trạng thái:** sửa nhẹ  
**Làm gì:** normalize coverage report có thêm `deadline_date`.

## Cell 38
**Trạng thái:** giữ  
**Làm gì:** giữ markdown section skill.

## Cell 39
**Trạng thái:** giữ  
**Làm gì:** giữ `normalize_tags()`.

## Cell 40
**Trạng thái:** sửa mạnh  
**Làm gì:** mở rộng `SKILL_TAXONOMY` cho:
- AI/ML: `llm`, `rag`, `gen ai`, `langchain`, `pytorch`, `tensorflow`, `sklearn`
- data engineering: `kafka`, `databricks`, `bigquery`, `snowflake`, `redshift`
- BI: `power query`, `dax`
- deployment/platform: `linux`, `git`, `ci/cd`, `fastapi`, `flask`
- BA/PM: `business analysis`, `project management`, `agile`, `scrum`

## Cell 41
**Trạng thái:** sửa mạnh  
**Làm gì:**
- compile regex cho skill
- thêm `REQUIRED_HINTS` và `PREFERRED_HINTS`
- thay `extract_skills_from_text()` bằng `extract_skill_records_from_text()`
- output record có:
  - `skill`
  - `skill_group`
  - `source_field`
  - `importance`
  - `excerpt`

## Cell 42
**Trạng thái:** sửa  
**Làm gì:** extract skill record riêng cho:
- tags
- title
- requirements
- description

## Cell 43
**Trạng thái:** sửa  
**Làm gì:** thêm helper merge record và helper tạo list từ records.

## Cell 44
**Trạng thái:** sửa mạnh  
**Làm gì:**
- tạo `skill_records`
- sinh:
  - `skills_extracted`
  - `skills_required`
  - `skills_preferred`
  - `skill_groups`
- tạo text field tương ứng

**Lý do sửa:** chatbot cần skill gap chứ không chỉ list skill chung.

## Cell 45
**Trạng thái:** bổ sung chức năng  
**Làm gì:** tạo `job_skill_map_df` bằng cách explode từng job-skill record.

**Vì sao thêm:** đây là bảng rất hữu ích cho explainability và chatbot.

## Cell 46
**Trạng thái:** sửa nhẹ  
**Làm gì:** thêm thống kê `num_required_skills`, `num_preferred_skills`.

## Cell 47
**Trạng thái:** sửa  
**Làm gì:** ngoài top skills, tạo thêm `role_skill_stats_df`.

## Cell 48
**Trạng thái:** giữ  
**Làm gì:** giữ empty skill ratio.

## Cell 49
**Trạng thái:** sửa mạnh  
**Làm gì:**
- thay text builder cũ bằng 3 nhánh rõ hơn:
  - `job_text_sparse` cho TF-IDF/BM25
  - `job_text_dense` cho PhoBERT / dense retriever
  - `job_text_chatbot` cho chatbot summary
- thêm `job_chatbot_profile`

**Lý do sửa:** tách input theo đúng use case.

## Cell 50
**Trạng thái:** sửa  
**Làm gì:** apply các text builder mới.

## Cell 51
**Trạng thái:** giữ, sửa nhẹ  
**Làm gì:** preview text artifacts mới.

## Cell 52
**Trạng thái:** bổ sung chức năng lớn  
**Làm gì:**
- thêm `split_long_text()`
- thêm `build_job_section_records()`
- tạo `job_sections_df`

**Vì sao thêm:** đây là bảng chunk section-ready cho chatbot/RAG.

## Cell 53
**Trạng thái:** sửa  
**Làm gì:**
- length stats cho `job_text_sparse`, `job_text_dense`, `job_text_chatbot`
- preview short/long cases
- preview số lượng section theo type

## Cell 54
**Trạng thái:** sửa nhẹ  
**Làm gì:** đổi markdown từ “PhoBERT section-aware matching” sang “Dense embeddings (tuỳ chọn)”.

## Cell 55
**Trạng thái:** sửa mạnh  
**Làm gì:**
- thêm `RUN_EMBEDDING`, `RUN_SECTION_EMBEDDING`
- không load model mặc định nữa
- chỉ load khi cần

**Lý do sửa:** preprocessing notebook không nên bắt user encode ngay.

## Cell 56
**Trạng thái:** sửa  
**Làm gì:**
- giữ `mean_pooling`
- thêm `maybe_segment_vi_text()` cho hướng PhoBERT

## Cell 57
**Trạng thái:** sửa mạnh  
**Làm gì:**
- bọc `encode_texts()` dưới điều kiện `RUN_EMBEDDING`
- thêm `prepare_dense_input()`
- route text theo `dense_encoder_route`

## Cell 58
**Trạng thái:** sửa  
**Làm gì:** encode `job_text_dense` ở cấp job nếu bật embedding.

## Cell 59
**Trạng thái:** sửa  
**Làm gì:** encode `job_sections_df` ở cấp chunk nếu bật section embedding.

## Cell 60
**Trạng thái:** bổ sung  
**Làm gì:** thêm `DOWNSTREAM_FIELD_GUIDE` để chỉ rõ field nào dùng cho use case nào.

## Cell 61
**Trạng thái:** sửa mạnh  
**Làm gì:** build `df_matching_ready` mới, có thêm:
- deadline
- skills_required / preferred
- dense route
- `job_text_sparse`, `job_text_dense`

## Cell 62
**Trạng thái:** sửa mạnh  
**Làm gì:** build `df_chatbot_ready` mới, có thêm `job_chatbot_profile`.

## Cell 63
**Trạng thái:** bổ sung  
**Làm gì:** tạo `job_sections_ready`.

## Cell 64
**Trạng thái:** bổ sung  
**Làm gì:** tạo `job_skill_map_ready`.

## Cell 65
**Trạng thái:** bổ sung  
**Làm gì:** tạo `role_skill_stats_ready`.

## Cell 66
**Trạng thái:** sửa mạnh  
**Làm gì:** export tất cả artifact chính:
- `jobs_matching_ready_v2`
- `jobs_chatbot_ready_v2`
- `jobs_chatbot_sections_v2`
- `job_skill_map_v2`
- `role_skill_stats_v2`
- embeddings nếu có

## Cell 67
**Trạng thái:** bổ sung  
**Làm gì:** tạo bảng summary các artifact đã lưu.

## Cell 68
**Trạng thái:** sửa  
**Làm gì:** tạo `manifest` JSON cho toàn bộ run.

## Cell 69
**Trạng thái:** sửa  
**Làm gì:** thay markdown rỗng bằng ghi chú sử dụng artifact và bước tiếp theo.

---

## Kết quả notebook mới sinh ra
Notebook sửa xong: `preprocessing_copy_3_revised_v2.ipynb`

### Artifact mà notebook mới sẽ sinh ra
- `jobs_matching_ready_v2`
- `jobs_chatbot_ready_v2`
- `jobs_chatbot_sections_v2`
- `job_skill_map_v2`
- `role_skill_stats_v2`
- `preprocessing_manifest_v2.json`

---

## Gợi ý cách dùng sau khi sửa
- recommender:
  - TF-IDF / BM25 -> `job_text_sparse`
  - dense retrieval -> `job_text_dense`
- chatbot:
  - summary level -> `jobs_chatbot_ready_v2`
  - RAG level -> `jobs_chatbot_sections_v2`
  - gap analysis -> `job_skill_map_v2` + `role_skill_stats_v2`
