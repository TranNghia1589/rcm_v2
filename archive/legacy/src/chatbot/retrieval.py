from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

try:
    import requests

    _HAS_REQUESTS = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore
    _HAS_REQUESTS = False


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

BASE_DIR = Path(__file__).resolve().parents[2]
LLAMA_CPP_BIN = os.getenv("LLAMA_CPP_BIN", "main")
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "")

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.matching.recommend_engine import find_latest_jobs_file, get_top_job_recommendations
from src.models.matching.gap_engine import build_market_gap_report


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_job_block(recommended_jobs: List[Dict[str, Any]]) -> str:
    if not recommended_jobs:
        return "Khong co du lieu goi y job."

    lines = []
    for idx, job in enumerate(recommended_jobs, 1):
        lines.append(
            f"{idx}. {job.get('job_title', 'Unknown')} | {job.get('company_name', 'N/A')} | "
            f"{job.get('location', 'N/A')} | score={job.get('score', 0)} | {job.get('job_url', '')}"
        )
        matched = job.get("matched_skills", [])
        if matched:
            lines.append(f"   matched_skills: {', '.join(matched)}")
    return "\n".join(lines)


def ask_ollama(prompt: str, timeout_sec: int = 180) -> str:
    if not _HAS_REQUESTS:
        raise RuntimeError("Requests library unavailable, cannot call Ollama.")

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    data = response.json()
    value = data.get("response", "")
    if value is None:
        return ""
    return str(value).strip()


def get_llama_cpp_model_path() -> str:
    if LLAMA_CPP_MODEL:
        return LLAMA_CPP_MODEL

    candidates = [
        BASE_DIR / "models" / "ggml-model.bin",
        BASE_DIR / "models" / "llama.bin",
        BASE_DIR / "llama" / "ggml-model.bin",
        BASE_DIR / "llama" / "llama.bin",
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "No llama.cpp model found. Set LLAMA_CPP_MODEL or place a model in models/ or llama/."
    )


def ask_llama_cpp(prompt: str) -> str:
    model_path = get_llama_cpp_model_path()
    cmd = [
        LLAMA_CPP_BIN,
        "-m",
        model_path,
        "--prompt",
        prompt,
        "--n_predict",
        "512",
        "--temp",
        "0.2",
        "--top_k",
        "40",
        "--top_p",
        "0.95",
        "--repeat_last_n",
        "64",
        "--repeat_penalty",
        "1.1",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    result.check_returncode()

    output = result.stdout.strip()
    if prompt and output.startswith(prompt):
        output = output[len(prompt) :].strip()
    return output


def classify_question(question: str) -> str:
    q = question.lower()

    cv_keywords = [
        "cv",
        "resume",
        "hồ sơ",
        "job phù hợp",
        "việc phù hợp",
        "gợi ý việc làm",
        "top job",
        "thiếu gì",
        "thiếu kỹ năng",
        "phù hợp nghề",
        "hợp nghề nào",
        "điểm mạnh",
        "điểm yếu",
        "dựa trên cv",
        "dựa trên hồ sơ",
        "gap",
        "ứng tuyển",
    ]

    career_keywords = [
        "nên học gì",
        "roadmap",
        "lộ trình",
        "nên phát triển gì",
        "nên làm project gì",
        "phát triển kỹ năng",
        "3 tháng",
        "6 tháng",
        "để trở thành",
        "định hướng nghề nghiệp",
    ]

    if any(k in q for k in cv_keywords):
        return "cv_analysis"
    if any(k in q for k in career_keywords):
        return "career_advice"
    return "general_question"


def _vi_guard() -> str:
    return (
        "BẮT BUỘC trả lời hoàn toàn bằng tiếng Việt. "
        "Không dùng tiếng Anh, trừ tên công nghệ như SQL/Python/Power BI."
    )


def build_cv_prompt(gap_result: dict, user_question: str, recommended_jobs: List[Dict[str, Any]]) -> str:
    return f"""
Bạn là chatbot tư vấn nghề nghiệp và phân tích CV cho nhóm ngành Data/AI.
{_vi_guard()}
Không bịa thông tin ngoài dữ liệu cung cấp.

Bắt buộc trả lời theo 6 mục:
1. Mức độ phù hợp
2. Điểm mạnh hiện tại
3. Điểm còn thiếu
4. Kỹ năng nên phát triển tiếp
5. Hành động đề xuất trong 1-3 tháng
6. Top 10 job phù hợp nhất từ dữ liệu tuyển dụng

Dữ liệu phân tích CV:
{json.dumps(gap_result, ensure_ascii=False, indent=2)}

Top jobs đã được chấm theo CV:
{json.dumps(recommended_jobs, ensure_ascii=False, indent=2)}

Câu hỏi người dùng:
{user_question}
""".strip()


def build_career_prompt(gap_result: dict, user_question: str, recommended_jobs: List[Dict[str, Any]]) -> str:
    return f"""
Bạn là chatbot tư vấn nghề nghiệp Data/AI.
{_vi_guard()}
Tư vấn thực tế, rõ ràng, ngắn gọn, có ưu tiên hành động.

Ưu tiên nội dung:
- kỹ năng nên học trước
- role phù hợp hơn
- project nên làm
- hành động cụ thể trong 1-3 tháng

Dữ liệu phân tích:
{json.dumps(gap_result, ensure_ascii=False, indent=2)}

Top jobs đã được chấm theo CV:
{json.dumps(recommended_jobs, ensure_ascii=False, indent=2)}

Câu hỏi:
{user_question}
""".strip()


def build_general_prompt(user_question: str) -> str:
    return f"""
Bạn là trợ lý tư vấn nghề nghiệp Data/AI.
{_vi_guard()}
Giải thích dễ hiểu, đúng trọng tâm.

Câu hỏi:
{user_question}
""".strip()


def looks_english_heavy(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    english_markers = [
        "based on",
        "i recommend",
        "in addition",
        "you should",
        "project",
        "skills",
    ]
    score = sum(1 for m in english_markers if m in lower)
    vi_markers = ["bạn", "kỹ năng", "đề xuất", "tháng", "mức độ", "phù hợp"]
    vi_score = sum(1 for m in vi_markers if m in lower)
    return score >= 2 and vi_score == 0


def enforce_vietnamese(answer: str) -> str:
    if not answer:
        return answer
    if not looks_english_heavy(answer):
        return answer
    return (
        "Mình đã nhận được phản hồi nhưng mô hình trả lời tiếng Anh. "
        "Vui lòng hỏi lại một lần nữa, hoặc mình sẽ tự động dịch ở bước tiếp theo nếu bạn muốn.\n\n"
        + answer
    )


def fallback_answer(
    intent: str,
    gap_result: dict | None = None,
    recommended_jobs: List[Dict[str, Any]] | None = None,
) -> str:
    if intent == "general_question":
        return "Hiện chưa gọi được mô hình LLM (Ollama/llama.cpp). Hãy kiểm tra Ollama rồi thử lại."

    roles = gap_result.get("best_fit_roles", []) if gap_result else []
    missing = gap_result.get("missing_skills", []) if gap_result else []
    strengths = gap_result.get("strengths", []) if gap_result else []

    lines = ["Mình đang ở chế độ dự phòng nên trả lời ngắn gọn."]
    if roles:
        lines.append(f"Role phù hợp nhất hiện tại: {roles[0]}")
    if strengths:
        lines.append("Điểm mạnh:")
        lines.extend([f"- {s}" for s in strengths[:5]])
    if missing:
        lines.append("Điểm còn thiếu:")
        lines.extend([f"- {m}" for m in missing[:5]])
    return "\n".join(lines)


def _detect_internal_intent(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["job", "việc", "vị trí", "ứng tuyển", "top"]):
        return "job_match"
    if any(k in q for k in ["thiếu", "gap", "còn thiếu"]):
        return "gap_focus"
    if any(k in q for k in ["cải thiện cv", "sửa cv", "review cv", "nên học", "roadmap", "lộ trình"]):
        return "improve_cv"
    if any(k in q for k in ["kinh nghiệm", "experience"]):
        return "experience_focus"
    return "summary"


def build_internal_answer(
    *,
    intent: str,
    gap_result: dict | None,
    recommended_jobs: List[Dict[str, Any]],
    market_gap_report: Dict[str, Any] | None,
    user_question: str,
) -> str:
    if intent == "general_question":
        return (
            "Chatbot dang o che do internal-only, chi tra loi dua tren du lieu noi bo "
            "(CV da trich xuat, gap analysis, va jobs preprocess). "
            "Vui long dat cau hoi theo huong CV/job matching."
        )

    if not gap_result:
        return "Khong co gap_result de phan tich noi bo."

    roles = gap_result.get("best_fit_roles", [])
    strengths = gap_result.get("strengths", [])
    missing = gap_result.get("missing_skills", [])
    domain_fit = gap_result.get("domain_fit", "unknown")
    report = market_gap_report or {}
    top_missing_skills = [x.get("skill", "") for x in report.get("top_missing_skills", []) if x.get("skill")]
    missing_project_signals = report.get("missing_project_signals", [])
    exp_info = report.get("experience", {})
    internal_intent = _detect_internal_intent(user_question)

    lines = [f"Tom tat theo du lieu noi bo cho cau hoi: {user_question}"]
    if roles:
        lines.append(f"Muc do phu hop hien tai: {domain_fit}; role gan nhat: {roles[0]}.")

    if internal_intent == "job_match":
        lines.append("\nTop job matching de ung tuyen:")
        if recommended_jobs:
            for idx, job in enumerate(recommended_jobs[:10], 1):
                lines.append(
                    f"{idx}. {job.get('job_title', 'Unknown')} | {job.get('company_name', 'N/A')} | "
                    f"{job.get('location', 'N/A')} | score={job.get('score', 0)}"
                )
        else:
            lines.append("- Chua co du lieu job preprocess.")

    elif internal_intent == "gap_focus":
        lines.append("\nCac thieu hut uu tien:")
        if top_missing_skills:
            lines.extend([f"- {x}" for x in top_missing_skills[:8]])
        elif missing:
            lines.extend([f"- {x}" for x in missing[:8]])
        else:
            lines.append("- Chua phat hien thieu hut ro rang.")

    elif internal_intent == "experience_focus":
        lines.append("\nKhoang cach kinh nghiem so voi top job:")
        cv_exp = exp_info.get("cv_experience_years", None)
        expected = exp_info.get("expected_years_from_top_jobs", None)
        gap = exp_info.get("experience_gap_years", None)
        lines.append(f"- CV: {cv_exp if cv_exp is not None else 'Unknown'} nam")
        lines.append(f"- Thi truong top job: {expected if expected is not None else 'Unknown'} nam")
        lines.append(f"- Khoang cach: {gap if gap is not None else 0} nam")

    else:
        lines.append("\nDiem manh hien tai:")
        if strengths:
            lines.extend([f"- {x}" for x in strengths[:6]])
        else:
            lines.append("- Chua co nhieu tin hieu ky nang noi bat.")

        lines.append("\nDiem can cai thien:")
        if top_missing_skills:
            lines.extend([f"- {x}" for x in top_missing_skills[:6]])
        elif missing:
            lines.extend([f"- {x}" for x in missing[:6]])
        else:
            lines.append("- Chua phat hien thieu hut ro rang.")

        if missing_project_signals:
            lines.append("\nLoai project nen bo sung:")
            lines.extend([f"- {x}" for x in missing_project_signals[:4]])

        roadmap = report.get("roadmap_30_60_90", {})
        if roadmap:
            lines.append("\nRoadmap de xuat:")
            for phase in ["day_1_30", "day_31_60", "day_61_90"]:
                tasks = roadmap.get(phase, [])
                if tasks:
                    lines.append(f"- {phase}: {tasks[0]}")

    lines.append("\nGhi chu: cau tra loi duoc tao tu du lieu noi bo, khong dung kien thuc ngoai.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--gap_result", default="", help="Optional path to gap analysis result JSON")
    parser.add_argument("--cv_json", default="", help="Optional path to extracted CV JSON")
    parser.add_argument("--jobs_path", default="", help="Optional path to jobs_nlp_ready_*.csv")
    parser.add_argument("--top_k_jobs", type=int, default=10, help="Number of jobs to recommend")
    parser.add_argument(
        "--allow_external_llm",
        action="store_true",
        help="Allow calling Ollama/llama.cpp. Default is internal-only mode.",
    )
    args = parser.parse_args()

    intent = classify_question(args.question)
    gap_result = None
    cv_info = None
    recommended_jobs: List[Dict[str, Any]] = []
    market_gap_report: Dict[str, Any] | None = None

    if args.gap_result:
        gap_path = Path(args.gap_result)
        if gap_path.exists():
            gap_result = load_json(str(gap_path))
    if args.cv_json:
        cv_path = Path(args.cv_json)
        if cv_path.exists():
            cv_info = load_json(str(cv_path))

    if intent in {"cv_analysis", "career_advice"} and not gap_result:
        raise ValueError("Câu hỏi dạng CV/career cần --gap_result")

    if intent in {"cv_analysis", "career_advice"} and gap_result:
        jobs_path = Path(args.jobs_path) if args.jobs_path else find_latest_jobs_file(BASE_DIR)
        try:
            recommended_jobs = get_top_job_recommendations(
                cv_info=cv_info,
                gap_result=gap_result,
                jobs_path=jobs_path,
                top_k=args.top_k_jobs,
            )
            market_gap_report = build_market_gap_report(
                cv_info=cv_info,
                gap_result=gap_result,
                recommended_jobs=recommended_jobs,
            )
        except Exception as e:
            print(f"[Warning] Job recommendation failed: {e}")
            recommended_jobs = []
            market_gap_report = None

    # Default mode: internal-only, no external LLM call.
    if not args.allow_external_llm:
        answer = build_internal_answer(
            intent=intent,
            gap_result=gap_result,
            recommended_jobs=recommended_jobs,
            market_gap_report=market_gap_report,
            user_question=args.question,
        )
    else:
        if intent == "cv_analysis":
            prompt = build_cv_prompt(gap_result, args.question, recommended_jobs)  # type: ignore[arg-type]
        elif intent == "career_advice":
            prompt = build_career_prompt(gap_result, args.question, recommended_jobs)  # type: ignore[arg-type]
        else:
            prompt = build_general_prompt(args.question)

        answer = ""
        try:
            answer = ask_ollama(prompt)
        except Exception as e:
            print(f"[Warning] Ollama call failed: {e}")
            try:
                answer = ask_llama_cpp(prompt)
            except Exception as e2:
                print(f"[Warning] llama.cpp call failed: {e2}")

        if not isinstance(answer, str) or not answer.strip():
            answer = fallback_answer(intent, gap_result, recommended_jobs)

        answer = enforce_vietnamese(answer.strip())
        if recommended_jobs and intent in {"cv_analysis", "career_advice"}:
            answer = (
                f"{answer}\n\n===== TOP {len(recommended_jobs)} JOB MATCHING =====\n"
                f"{format_job_block(recommended_jobs)}"
            )

    print("\n===== INTENT =====")
    print(intent)
    print("\n===== ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()


