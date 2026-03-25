import json
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_CASES_PATH = BASE_DIR / "data" / "evaluation_cases.json"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def run_command(command: list[str]) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        print("Lỗi khi chạy lệnh:")
        print(" ".join(command))
        print(result.stderr)
        return ""

    return result.stdout


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cases = load_json(EVAL_CASES_PATH)

    for case in cases:
        print("\n" + "=" * 60)
        print(f"CASE: {case['case_id']}")
        print(case["description"])

        cv_path = BASE_DIR / case["cv_file"]

        extracted_path = PROCESSED_DIR / f"{case['case_id']}_extracted.json"
        gap_path = PROCESSED_DIR / f"{case['case_id']}_gap.json"
        if extracted_path.exists():
            extracted_path.unlink()

        if gap_path.exists():
            gap_path.unlink()
        # Step 1: extract CV info

        extract_out = run_command([
            "python",
            str(BASE_DIR / "src" / "cv_processing" / "extract_cv_info.py"),
            "--cv_path", str(cv_path),
            "--output_path", str(extracted_path)
        ])

        if not extracted_path.exists():
            print("FAIL: Không tạo được file extracted.")
            continue

        gap_out = run_command([
            "python",
            str(BASE_DIR / "src" / "matching" / "gap_analysis.py"),
            "--cv_json", str(extracted_path),
            "--output_path", str(gap_path)
        ])

        if not gap_path.exists():
            print("FAIL: Không tạo được file gap analysis.")
            continue

        gap_result = load_json(gap_path)

        actual_target_role = gap_result.get("target_role_from_cv", "")
        actual_domain_fit = gap_result.get("domain_fit", "")
        actual_best_fit_roles = gap_result.get("best_fit_roles", [])
        actual_missing_skills = gap_result.get("missing_skills", [])

        print(f"Expected target_role: {case['expected_target_role']}")
        print(f"Actual target_role:   {actual_target_role}")

        print(f"Expected domain_fit:  {case['expected_domain_fit']}")
        print(f"Actual domain_fit:    {actual_domain_fit}")

        print(f"Actual best_fit_roles: {actual_best_fit_roles}")
        print(f"Actual missing_skills: {actual_missing_skills}")

        target_ok = actual_target_role == case["expected_target_role"]
        domain_ok = actual_domain_fit == case["expected_domain_fit"]

        best_fit_ok = all(
            role in actual_best_fit_roles
            for role in case["expected_best_fit_roles_contains"]
        )

        missing_ok = all(
            skill in actual_missing_skills
            for skill in case["expected_missing_skills_contains"]
        )

        print(f"Check target role: {'OK' if target_ok else 'FAIL'}")
        print(f"Check domain fit:  {'OK' if domain_ok else 'FAIL'}")
        print(f"Check best roles:  {'OK' if best_fit_ok else 'FAIL'}")
        print(f"Check missing:     {'OK' if missing_ok else 'FAIL'}")


if __name__ == "__main__":
    main()