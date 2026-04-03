#!/usr/bin/env python
"""
Interactive tester for project_v3 pipeline:
Extract CV -> Gap Analysis -> Chatbot

Run:
python deploy/scripts/dev/test_chatbot_interactive.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


BASE_DIR = Path(__file__).resolve().parents[2]
CV_SAMPLES_DIR = BASE_DIR / "data" / "raw" / "cv_samples"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def header(text: str) -> None:
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def section(text: str) -> None:
    print(f"\n--- {text} ---")


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def check_requirements() -> bool:
    header("1) CHECK REQUIREMENTS")
    required = ["requests", "pandas", "fitz"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
            print(f"[OK] {pkg}")
        except ImportError:
            print(f"[MISSING] {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\nInstall missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements\\ml.txt")
        return False

    print("\nAll required packages are available.")
    return True


def list_cv_samples() -> list[tuple[str, Path]]:
    header("2) AVAILABLE CV SAMPLES")

    samples: list[tuple[str, Path]] = []
    if CV_SAMPLES_DIR.exists():
        txt_files = sorted(CV_SAMPLES_DIR.glob("*.txt"))
        pdf_files = sorted(CV_SAMPLES_DIR.glob("*.pdf"))

        idx = 1
        for f in txt_files:
            print(f"{idx}. {f.name} (TXT)")
            samples.append(("txt", f))
            idx += 1

        for f in pdf_files:
            print(f"{idx}. {f.name} (PDF)")
            samples.append(("pdf", f))
            idx += 1

    if not samples:
        print("No CV sample found in data/raw/cv_samples")

    return samples


def extract_cv_step(samples: list[tuple[str, Path]]) -> str | None:
    header("3) STEP 1 - EXTRACT CV")

    if not samples:
        return None

    print("Choose CV file:")
    for i, (fmt, path) in enumerate(samples, 1):
        print(f"  {i}. {path.name} ({fmt.upper()})")
    print("  0. Skip")

    choice = input(f"Select (0-{len(samples)}): ").strip()

    try:
        idx = int(choice) - 1
        if idx == -1:
            return None
        if idx < 0 or idx >= len(samples):
            print("Invalid selection.")
            return None

        _, cv_path = samples[idx]
        output_file = PROCESSED_DIR / f"test_extracted_{cv_path.stem}.json"

        cmd = [
            "python",
            "src/models/cv/extract_cv_info.py",
            "--cv_path",
            str(cv_path),
            "--output_path",
            str(output_file),
        ]

        section("Running extract")
        print("$ " + " ".join(cmd))
        result = run_cmd(cmd)

        if result.returncode != 0:
            print("Extract failed:")
            print(result.stderr)
            return None

        print(f"Extract success: {output_file.name}")
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"- target_role: {data.get('target_role', 'N/A')}")
        print(f"- skills: {len(data.get('skills', []))}")
        print(f"- experience_years: {data.get('experience_years', 'N/A')}")
        return str(output_file)

    except ValueError:
        print("Invalid input.")
        return None


def gap_analysis_step(extracted_file: str | None) -> str | None:
    header("4) STEP 2 - GAP ANALYSIS")

    if not extracted_file:
        gap_files = sorted(PROCESSED_DIR.glob("*gap*.json"))
        if not gap_files:
            print("No extracted file and no existing gap file found.")
            return None

        print("Existing gap files:")
        for i, f in enumerate(gap_files, 1):
            print(f"  {i}. {f.name}")
        print("  0. Cancel")

        choice = input("Select: ").strip()
        try:
            idx = int(choice) - 1
            if idx == -1:
                return None
            if 0 <= idx < len(gap_files):
                return str(gap_files[idx])
        except ValueError:
            pass
        print("Invalid selection.")
        return None

    output_file = PROCESSED_DIR / f"test_gap_{Path(extracted_file).stem}.json"
    cmd = [
        "python",
        "src/matching/gap_analysis.py",
        "--cv_json",
        str(extracted_file),
        "--output_path",
        str(output_file),
    ]

    section("Running gap analysis")
    print("$ " + " ".join(cmd))
    result = run_cmd(cmd)

    if result.returncode != 0:
        print("Gap analysis failed:")
        print(result.stderr)
        return None

    print(f"Gap analysis success: {output_file.name}")
    with open(output_file, "r", encoding="utf-8") as f:
        gap = json.load(f)
    print(f"- domain_fit: {gap.get('domain_fit')}")
    print(f"- best_fit_roles: {gap.get('best_fit_roles', [])}")
    top = gap.get("top_role_result") or {}
    if top:
        print(f"- top_role: {top.get('role')} (score: {top.get('score')})")

    return str(output_file)


def chatbot_step(gap_analysis_file: str | None) -> None:
    header("5) STEP 3 - CHATBOT TEST")

    if not gap_analysis_file:
        print("Need gap analysis file first.")
        return

    sample_questions = {
        "1": "CV của tôi phù hợp với vị trí nào nhất?",
        "2": "Nên học gì để trở thành Data Engineer?",
        "3": "Trong 3 tháng tới tôi nên làm gì?",
        "4": "Machine Learning là gì?",
        "5": "Tôi thiếu kỹ năng gì?",
        "6": "SQL khác gì NoSQL?",
        "7": "(custom)",
    }

    print("Choose a question:")
    for k, q in sample_questions.items():
        print(f"  {k}. {q}")
    print("  0. Skip")

    choice = input("Select: ").strip()
    if choice == "0":
        return
    if choice not in sample_questions:
        print("Invalid selection.")
        return

    if choice == "7":
        question = input("Enter your question: ").strip()
        if not question:
            print("Question cannot be empty.")
            return
    else:
        question = sample_questions[choice]

    cmd = [
        "python",
        "src/chatbot/retrieval.py",
        "--question",
        question,
        "--gap_result",
        str(gap_analysis_file),
    ]

    section("Running chatbot")
    print("$ " + " ".join(cmd))
    result = run_cmd(cmd)

    if result.returncode != 0:
        print("Chatbot failed:")
        print(result.stderr)
        return

    print(result.stdout)


def batch_test() -> None:
    header("BATCH TEST - FULL PIPELINE")

    samples = sorted(CV_SAMPLES_DIR.glob("*.txt"))
    if not samples:
        print("No TXT CV sample found.")
        return

    cv_file = samples[0]
    print(f"Using CV: {cv_file.name}")

    extracted_file = PROCESSED_DIR / "batch_extracted.json"
    gap_file = PROCESSED_DIR / "batch_gap.json"

    extract_cmd = [
        "python", "src/models/cv/extract_cv_info.py", "--cv_path", str(cv_file), "--output_path", str(extracted_file)
    ]
    gap_cmd = [
        "python", "src/matching/gap_analysis.py", "--cv_json", str(extracted_file), "--output_path", str(gap_file)
    ]

    print("1) Extract CV ...", end=" ", flush=True)
    r1 = run_cmd(extract_cmd)
    print("OK" if r1.returncode == 0 else "FAILED")
    if r1.returncode != 0:
        print(r1.stderr)
        return

    print("2) Gap Analysis ...", end=" ", flush=True)
    r2 = run_cmd(gap_cmd)
    print("OK" if r2.returncode == 0 else "FAILED")
    if r2.returncode != 0:
        print(r2.stderr)
        return

    queries = [
        "CV của tôi phù hợp với vị trí nào nhất?",
        "Tôi nên học gì trong 3 tháng tới?",
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n3.{i}) Chatbot query: {q}")
        cmd = ["python", "src/chatbot/retrieval.py", "--question", q, "--gap_result", str(gap_file)]
        rr = run_cmd(cmd)
        if rr.returncode == 0:
            print(rr.stdout)
        else:
            print(rr.stderr)


def main() -> None:
    print("\n" + "=" * 64)
    print("INTERACTIVE CHATBOT TESTING TOOL (project_v3)")
    print("=" * 64)
    print("Flow: Extract CV -> Gap Analysis -> Chatbot")

    if not check_requirements():
        return

    while True:
        print("\n" + "-" * 64)
        print("Choose mode:")
        print("  1. Step by step")
        print("  2. Batch test")
        print("  3. Show CV samples")
        print("  0. Exit")

        choice = input("Select (0-3): ").strip()

        if choice == "1":
            samples = list_cv_samples()
            extracted = extract_cv_step(samples)
            gap_file = gap_analysis_step(extracted)
            chatbot_step(gap_file)
        elif choice == "2":
            batch_test()
        elif choice == "3":
            list_cv_samples()
        elif choice == "0":
            print("Bye.")
            break
        else:
            print("Invalid selection.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnhandled error: {e}")
        import traceback

        traceback.print_exc()



