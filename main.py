"""
main.py
-------
Main entry point.  Run from the project root:
    python main.py

What it does:
  1. Builds (or loads) the FAISS catalog index
  2. Runs 7 sample interactions and prints them
  3. Runs the 25-query evaluation suite and saves results to outputs/
"""

import sys
import json
import os
from pathlib import Path

# ── Make sure both `src` and `tests` are importable ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from agents.agents import CourseAdvisorOrchestrator   # noqa: E402


# ── Sample interactions ───────────────────────────────────────────────────────

SAMPLE_INTERACTIONS = [
    {
        "name": "1. Eligible — CS 302",
        "query": "Am I eligible to enroll in CS 302 (Operating Systems)?",
        "profile": {
            "major": "BS Computer Science",
            "catalog_year": "2024-2025",
            "completed_courses": [
                {"code": "CS 101", "grade": "A"},
                {"code": "CS 102", "grade": "B+"},
                {"code": "CS 201", "grade": "B"},
                {"code": "CS 210", "grade": "C+"},
            ],
            "target_term": "Fall 2025",
            "max_credits": 15,
        },
    },
    {
        "name": "2. Not Eligible — D grade in CS 201",
        "query": "Can I take CS 302 if I got a D in CS 201?",
        "profile": {
            "major": "BS Computer Science",
            "catalog_year": "2024-2025",
            "completed_courses": [
                {"code": "CS 201", "grade": "D"},
                {"code": "CS 210", "grade": "B"},
            ],
            "target_term": "Fall 2025",
        },
    },
    {
        "name": "3. Prereq chain — CS 401",
        "query": "What is the complete prerequisite chain for CS 401 Machine Learning?",
        "profile": {
            "major": "BS Computer Science",
            "catalog_year": "2024-2025",
            "completed_courses": [],
        },
    },
    {
        "name": "4. Course plan — AI/ML track, Spring 2025",
        "query": "Please suggest a course plan for Spring 2025 for the AI/ML concentration.",
        "profile": {
            "major": "BS Computer Science",
            "catalog_year": "2024-2025",
            "track": "AI/ML",
            "completed_courses": [
                {"code": "CS 101", "grade": "A"},
                {"code": "CS 102", "grade": "B+"},
                {"code": "CS 201", "grade": "B"},
                {"code": "CS 210", "grade": "B"},
                {"code": "CS 220", "grade": "A"},
                {"code": "MATH 115", "grade": "B+"},
                {"code": "STAT 201", "grade": "B"},
            ],
            "target_term": "Spring 2025",
            "max_credits": 15,
        },
    },
    {
        "name": "5. Policy — Pass/Fail",
        "query": "Can I take CS core courses on Pass/Fail to protect my GPA?",
        "profile": {"major": "BS Computer Science"},
    },
    {
        "name": "6. Abstention — professor assignment",
        "query": "Which professor teaches CS 340 in Fall 2025?",
        "profile": {"major": "BS Computer Science"},
    },
    {
        "name": "7. Abstention — summer availability",
        "query": "Is CS 401 (Machine Learning) available in Summer 2025?",
        "profile": {"major": "BS Computer Science"},
    },
]


def run_sample_interactions(orchestrator: CourseAdvisorOrchestrator) -> list:
    print("\n" + "=" * 70)
    print("SAMPLE INTERACTIONS")
    print("=" * 70)

    results = []
    for sample in SAMPLE_INTERACTIONS:
        print(f"\n{'─' * 70}")
        print(f"[{sample['name']}]")
        print(f"Query: {sample['query']}")
        print("─" * 70)

        result = orchestrator.answer(
            student_query=sample["query"],
            student_profile=sample.get("profile", {}),
            skip_intake=False,
        )
        print(result["answer"])
        results.append({
            "name":         sample["name"],
            "query":        sample["query"],
            "answer":       result["answer"],
            "audit_result": result.get("audit_result"),
        })

    return results


def main():
    print("=" * 70)
    print("AGENTIC RAG COURSE PLANNING ASSISTANT")
    print("State University — CS Department")
    print("=" * 70)

    use_mock = (os.getenv("GROQ_API_KEY") is None and os.getenv("OPENAI_API_KEY") is None)
    orchestrator = CourseAdvisorOrchestrator(use_mock_llm=use_mock)

    # ── Sample interactions ────────────────────────────────────────────────────
    sample_results = run_sample_interactions(orchestrator)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "sample_interactions.json", "w", encoding="utf-8") as f:
        json.dump(sample_results, f, indent=2, ensure_ascii=False)
    print(f"\nSample interactions saved → {out_dir / 'sample_interactions.json'}")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Running 25-query evaluation suite …")
    print("=" * 70)

    from evaluation import run_evaluation, save_evaluation_report  # noqa: E402
    eval_output = run_evaluation(orchestrator)
    save_evaluation_report(eval_output, out_dir / "evaluation_report.json")

    print("\n✅  All done!  Outputs saved to ./outputs/")
    print("     sample_interactions.json")
    print("     evaluation_report.json")
    print("     faiss_index/")
    print("\nTo run the Streamlit demo:")
    print("     streamlit run demo/demo_app.py")


if __name__ == "__main__":
    main()