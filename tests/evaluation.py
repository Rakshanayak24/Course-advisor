"""
evaluation.py
-------------
25-query evaluation suite for the Agentic RAG Course Planning Assistant.

Run standalone:
    python tests/evaluation.py

Or imported by main.py via `from evaluation import run_evaluation, save_evaluation_report`
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# ── Ensure src/ is on the path however this file is invoked ──────────────────
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent          # tests/../  =  project root
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── 1-10: Prerequisite checks ─────────────────────────────────────────────
    {
        "id": "PC-01", "category": "prereq_check", "expected_decision": "NOT_ELIGIBLE",
        "query": "Can I take CS 301 if I've only completed CS 101 and MATH 120?",
        "profile": {"major": "BS Computer Science", "catalog_year": "2024-2025",
                    "completed_courses": [{"code": "CS 101", "grade": "B"}],
                    "target_term": "Spring 2025", "max_credits": 15},
        "notes": "Missing CS 102 and CS 201. NOT ELIGIBLE.",
    },
    {
        "id": "PC-02", "category": "prereq_check", "expected_decision": "ELIGIBLE",
        "query": "Am I eligible to enroll in CS 302 (Operating Systems)?",
        "profile": {"major": "BS Computer Science", "catalog_year": "2024-2025",
                    "completed_courses": [{"code": "CS 101", "grade": "A"},
                                          {"code": "CS 102", "grade": "B+"},
                                          {"code": "CS 201", "grade": "B"},
                                          {"code": "CS 210", "grade": "C+"}],
                    "target_term": "Fall 2025", "max_credits": 15},
        "notes": "CS 302 needs CS 201 AND CS 210 (C+). ELIGIBLE.",
    },
    {
        "id": "PC-03", "category": "prereq_check", "expected_decision": "NOT_ELIGIBLE",
        "query": "Can I take CS 302 if I got a D in CS 201?",
        "profile": {"major": "BS Computer Science", "catalog_year": "2024-2025",
                    "completed_courses": [{"code": "CS 201", "grade": "D"},
                                          {"code": "CS 210", "grade": "B"}],
                    "target_term": "Fall 2025"},
        "notes": "D in CS 201 fails the C-or-better threshold. NOT ELIGIBLE.",
    },
    {
        "id": "PC-04", "category": "prereq_check", "expected_decision": "ELIGIBLE",
        "query": "Can I enroll in CS 340 (Artificial Intelligence)?",
        "profile": {"major": "BS Computer Science", "catalog_year": "2024-2025",
                    "completed_courses": [{"code": "CS 201", "grade": "B+"},
                                          {"code": "MATH 210", "grade": "B"}],
                    "target_term": "Fall 2025"},
        "notes": "CS 340 needs CS 201 AND MATH 210. Both C+. ELIGIBLE.",
    },
    {
        "id": "PC-05", "category": "prereq_check", "expected_decision": "NOT_ELIGIBLE",
        "query": "Can I take CS 401 if I've completed CS 340 and STAT 201 but not MATH 301?",
        "profile": {"major": "BS Computer Science", "catalog_year": "2024-2025",
                    "completed_courses": [{"code": "CS 340", "grade": "B"},
                                          {"code": "STAT 201", "grade": "B+"}],
                    "target_term": "Spring 2025"},
        "notes": "Missing MATH 301. NOT ELIGIBLE.",
    },
    {
        "id": "PC-06", "category": "prereq_check", "expected_decision": "NOT_ELIGIBLE",
        "query": "I completed CS 310 with a C. Can I take CS 450 (Capstone I)?",
        "profile": {"major": "BS Computer Science", "catalog_year": "2024-2025",
                    "credits_completed": 92, "standing": "Senior", "cs_gpa": 2.8,
                    "completed_courses": [{"code": "CS 310", "grade": "C"}],
                    "target_term": "Fall 2025"},
        "notes": "CS 450 needs CS 310 B or better. C fails. NOT ELIGIBLE.",
    },
    {
        "id": "PC-07", "category": "prereq_check", "expected_decision": "ELIGIBLE",
        "query": "Am I eligible for CS 450 (Capstone I)?",
        "profile": {"major": "BS Computer Science", "catalog_year": "2024-2025",
                    "credits_completed": 95, "standing": "Senior", "cs_gpa": 2.7,
                    "completed_courses": [{"code": "CS 310", "grade": "B+"}],
                    "target_term": "Fall 2025"},
        "notes": "CS 310 B+ satisfies B-or-better for CS 450. ELIGIBLE.",
    },
    {
        "id": "PC-08", "category": "prereq_check", "expected_decision": "NEED_MORE_INFO",
        "query": "Can I take CS 480 Special Topics?",
        "profile": {"major": "BS Computer Science", "catalog_year": "2024-2025",
                    "completed_courses": [{"code": "CS 301", "grade": "B"}],
                    "target_term": "Fall 2025"},
        "notes": "CS 480 needs CS 301 (satisfied) AND instructor consent — cannot confirm.",
    },
    {
        "id": "PC-09", "category": "prereq_check", "expected_decision": "NOT_ELIGIBLE",
        "query": "Can I take CS 410 (Compilers) next semester?",
        "profile": {"major": "BS Computer Science", "catalog_year": "2024-2025",
                    "completed_courses": [{"code": "CS 302", "grade": "B"}],
                    "target_term": "Spring 2025"},
        "notes": "CS 410 needs CS 302 AND CS 350. Missing CS 350. NOT ELIGIBLE.",
    },
    {
        "id": "PC-10", "category": "prereq_check", "expected_decision": "ELIGIBLE",
        "query": "Can I take CS 461 (NLP) if I've completed CS 401?",
        "profile": {"major": "BS Computer Science", "catalog_year": "2024-2025",
                    "completed_courses": [{"code": "CS 401", "grade": "B+"}],
                    "target_term": "Spring 2025"},
        "notes": "CS 461 needs CS 401 C+. B+ satisfies. ELIGIBLE.",
    },

    # ── 11-15: Prerequisite chain questions ───────────────────────────────────
    {
        "id": "PQ-01", "category": "prereq_chain", "expected_decision": "CHAIN",
        "query": "What is the complete prerequisite chain for CS 401 Machine Learning?",
        "profile": {"major": "BS Computer Science", "completed_courses": []},
        "notes": "Full chain includes CS 101→CS 102→CS 201→CS 340→CS 401 + MATH + STAT.",
    },
    {
        "id": "PQ-02", "category": "prereq_chain", "expected_decision": "CHAIN",
        "query": "What is the full prerequisite chain to take CS 420 (Distributed Systems)?",
        "profile": {"major": "BS Computer Science", "completed_courses": []},
        "notes": "CS 420 ← CS 302+CS 330; CS 330 ← CS 302; CS 302 ← CS 201+CS 210.",
    },
    {
        "id": "PQ-03", "category": "prereq_chain", "expected_decision": "CHAIN",
        "query": "Is the co-requisite for CS 102 (MATH 115) required concurrently or can it be completed beforehand?",
        "profile": {"major": "BS Computer Science",
                    "completed_courses": [{"code": "CS 101", "grade": "A"},
                                          {"code": "MATH 115", "grade": "B"}]},
        "notes": "Policy: co-reqs can be satisfied by prior completion.",
    },
    {
        "id": "PQ-04", "category": "prereq_chain", "expected_decision": "CHAIN",
        "query": "What do I need to complete before I can enroll in CS 410 (Compilers)?",
        "profile": {"major": "BS Computer Science",
                    "completed_courses": [{"code": "CS 201", "grade": "B"},
                                          {"code": "CS 210", "grade": "B"}]},
        "notes": "CS 410 ← CS 302+CS 350; CS 350 ← CS 201+CS 320; CS 320 ← CS 201+CS 220.",
    },
    {
        "id": "PQ-05", "category": "prereq_chain", "expected_decision": "CHAIN",
        "query": "I've only taken CS 101. What is the minimum path to reach CS 302?",
        "profile": {"major": "BS Computer Science",
                    "completed_courses": [{"code": "CS 101", "grade": "B+"}]},
        "notes": "Path: CS 102 → CS 201 → CS 302; also need CS 210 (requires only CS 101).",
    },

    # ── 16-20: Program requirement questions ──────────────────────────────────
    {
        "id": "PR-01", "category": "program_requirement", "expected_decision": "INFO",
        "query": "How many total credit hours do I need to graduate with a BS in Computer Science?",
        "profile": {"major": "BS Computer Science"},
        "notes": "Answer: 120 credit hours.",
    },
    {
        "id": "PR-02", "category": "program_requirement", "expected_decision": "INFO",
        "query": "What courses satisfy the elective requirement for the General CS track?",
        "profile": {"major": "BS Computer Science", "track": "General CS"},
        "notes": "General track: CS 340 required + 3 electives from 300/400-level.",
    },
    {
        "id": "PR-03", "category": "program_requirement", "expected_decision": "INFO",
        "query": "What is the residency requirement for the BSCS degree?",
        "profile": {"major": "BS Computer Science"},
        "notes": "30 of final 45 credits at State U; 18 upper-div CS at State U.",
    },
    {
        "id": "PR-04", "category": "program_requirement", "expected_decision": "INFO",
        "query": "Can I take CS core courses Pass/Fail to protect my GPA?",
        "profile": {"major": "BS Computer Science"},
        "notes": "No — policy §1.4 prohibits Pass/Fail for CS core courses.",
    },
    {
        "id": "PR-05", "category": "program_requirement", "expected_decision": "PLAN",
        "query": "I've completed CS 101, CS 102, CS 201, CS 210, CS 220, MATH 115, STAT 201. Suggest a course plan for Fall 2025. Max 15 credits.",
        "profile": {"major": "BS Computer Science", "track": "AI/ML",
                    "completed_courses": [{"code": "CS 101", "grade": "A"},
                                          {"code": "CS 102", "grade": "B+"},
                                          {"code": "CS 201", "grade": "B"},
                                          {"code": "CS 210", "grade": "B"},
                                          {"code": "CS 220", "grade": "A"},
                                          {"code": "MATH 115", "grade": "B+"},
                                          {"code": "STAT 201", "grade": "B"}],
                    "target_term": "Fall 2025", "max_credits": 15},
        "notes": "Should suggest CS 301, CS 302, CS 310, CS 340 (with MATH 210 equivalent via CS 220).",
    },

    # ── 21-25: Not-in-docs / abstention ──────────────────────────────────────
    {
        "id": "ND-01", "category": "abstention", "expected_decision": "ABSTAIN",
        "query": "Is CS 401 available in the Summer 2025 session?",
        "profile": {"major": "BS Computer Science"},
        "notes": "Catalog says Fall+Spring only. Summer not confirmed — must abstain.",
    },
    {
        "id": "ND-02", "category": "abstention", "expected_decision": "ABSTAIN",
        "query": "Which professor teaches CS 340 in Fall 2025?",
        "profile": {"major": "BS Computer Science"},
        "notes": "Faculty assignments not in catalog. Must abstain.",
    },
    {
        "id": "ND-03", "category": "abstention", "expected_decision": "ABSTAIN",
        "query": "How do I get instructor consent for CS 480? What criteria does the professor use?",
        "profile": {"major": "BS Computer Science",
                    "completed_courses": [{"code": "CS 301", "grade": "B"}]},
        "notes": "Catalog says 'consent required' but gives no criteria. Must abstain on criteria.",
    },
    {
        "id": "ND-04", "category": "abstention", "expected_decision": "ABSTAIN",
        "query": "Will CS 480 on Blockchain be offered this fall?",
        "profile": {"major": "BS Computer Science"},
        "notes": "Specific topic not guaranteed — must abstain.",
    },
    {
        "id": "ND-05", "category": "abstention", "expected_decision": "ABSTAIN",
        "query": "What is the current waitlist length for CS 302?",
        "profile": {"major": "BS Computer Science"},
        "notes": "Waitlist info not in catalog. Must abstain.",
    },
]


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_evaluation(orchestrator, test_cases: list = None) -> Dict[str, Any]:
    if test_cases is None:
        test_cases = TEST_CASES

    results = []
    citation_count       = 0
    eligibility_correct  = 0
    eligibility_total    = 0
    abstention_correct   = 0
    abstention_total     = 0

    print("=" * 70)
    print(f"EVALUATION — {len(test_cases)} queries")
    print("=" * 70)

    for tc in test_cases:
        print(f"\n[{tc['id']}] {tc['category'].upper()} — {tc['query'][:60]}…")

        t0 = time.time()
        result = orchestrator.answer(
            student_query=tc["query"],
            student_profile=tc.get("profile", {}),
            skip_intake=True,
        )
        elapsed = round(time.time() - t0, 2)

        answer = result.get("answer", "")
        ans_lo  = answer.lower()

        # Citation present?
        has_cit = "chunk_" in answer or "N/A" in answer
        if has_cit:
            citation_count += 1

        # Decide correctness
        if tc["category"] == "abstention":
            abstention_total += 1
            correct = any(p in ans_lo for p in [
                "don't have that information",
                "not in the provided",
                "not covered in catalog",
                "outside the scope",
                "n/a",
            ])
            if correct:
                abstention_correct += 1

        elif tc["category"] == "prereq_check":
            eligibility_total += 1
            exp = tc["expected_decision"]
            if exp == "ELIGIBLE":
                correct = "eligible" in ans_lo and "not eligible" not in ans_lo
            elif exp == "NOT_ELIGIBLE":
                correct = "not eligible" in ans_lo or "not satisfied" in ans_lo
            elif exp == "NEED_MORE_INFO":
                correct = ("consent" in ans_lo or "need more info" in ans_lo
                           or "cannot confirm" in ans_lo)
            else:
                correct = True
            if correct:
                eligibility_correct += 1

        else:
            correct = True   # chain / plan / info — not graded binary

        mark  = "✓" if correct  else "✗"
        cite  = "📎" if has_cit else "⚠"
        print(f"  {mark} correct | {cite} cited | {elapsed}s")

        results.append({
            "id":             tc["id"],
            "category":       tc["category"],
            "query":          tc["query"],
            "expected":       tc["expected_decision"],
            "has_citation":   has_cit,
            "correct":        correct,
            "elapsed_s":      elapsed,
            "answer_snippet": answer[:300],
        })

    total = len(test_cases)
    metrics = {
        "total_queries":          total,
        "citation_coverage_pct":  round(citation_count / total * 100, 1),
        "eligibility_accuracy_pct": (
            round(eligibility_correct / eligibility_total * 100, 1)
            if eligibility_total else 0
        ),
        "eligibility_correct":    eligibility_correct,
        "eligibility_total":      eligibility_total,
        "abstention_accuracy_pct": (
            round(abstention_correct / abstention_total * 100, 1)
            if abstention_total else 0
        ),
        "abstention_correct":     abstention_correct,
        "abstention_total":       abstention_total,
    }

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Citation coverage   : {metrics['citation_coverage_pct']}%  ({citation_count}/{total})")
    print(f"Eligibility accuracy: {metrics['eligibility_accuracy_pct']}%  ({eligibility_correct}/{eligibility_total})")
    print(f"Abstention accuracy : {metrics['abstention_accuracy_pct']}%  ({abstention_correct}/{abstention_total})")
    print("=" * 70)

    return {"metrics": metrics, "results": results}


def save_evaluation_report(eval_output: Dict[str, Any], output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation report saved → {output_path}")


# ── Standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.agents import CourseAdvisorOrchestrator
    orch = CourseAdvisorOrchestrator(use_mock_llm=True)
    out  = run_evaluation(orch)
    save_evaluation_report(out, PROJECT_ROOT / "outputs" / "evaluation_report.json")