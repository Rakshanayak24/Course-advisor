"""
agents.py
---------
Agentic Course Planning Assistant — LangChain-based multi-agent orchestration.
Profile-aware SmartMockLLM that reads actual student data for accurate answers.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.prompts import SYSTEM_PROMPT, FORMAT_INSTRUCTIONS, build_student_profile_string
from rag.rag_pipeline import build_index, retrieve_context, format_retrieved_chunks_for_prompt
from langchain.schema import Document


# ─────────────────────────────────────────────────────────────────────────────
# CATALOG KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

COURSE_CATALOG = {
    "CS 101":  {"name": "Introduction to Programming",          "credits": 3, "prereqs": [],                       "prereqs_grades": {},                              "coreqs": [],          "offered": ["Fall", "Spring", "Summer"], "special": None,                                                              "chunk_id": "chunk_0001", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 102":  {"name": "Data Structures and Algorithms I",     "credits": 3, "prereqs": ["CS 101"],               "prereqs_grades": {"CS 101": "C"},                 "coreqs": ["MATH 115"],"offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0002", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 201":  {"name": "Data Structures and Algorithms II",    "credits": 3, "prereqs": ["CS 102"],               "prereqs_grades": {"CS 102": "C"},                 "coreqs": [],          "offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0003", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 210":  {"name": "Computer Organization and Architecture","credits": 3, "prereqs": ["CS 101"],              "prereqs_grades": {"CS 101": "C"},                 "coreqs": [],          "offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0004", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 220":  {"name": "Discrete Mathematics for CS",          "credits": 3, "prereqs": ["MATH 115"],             "prereqs_grades": {"MATH 115": "C"},               "coreqs": [],          "offered": ["Fall", "Spring"],           "special": "Cross-listed as MATH 210.",                                       "chunk_id": "chunk_0005", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 301":  {"name": "Database Systems",                     "credits": 3, "prereqs": ["CS 201"],               "prereqs_grades": {"CS 201": "C"},                 "coreqs": [],          "offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0006", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 302":  {"name": "Operating Systems",                    "credits": 3, "prereqs": ["CS 201", "CS 210"],     "prereqs_grades": {"CS 201": "C", "CS 210": "C"},  "coreqs": [],          "offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0007", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 310":  {"name": "Software Engineering",                 "credits": 3, "prereqs": ["CS 201"],               "prereqs_grades": {"CS 201": "C"},                 "coreqs": [],          "offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0008", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 320":  {"name": "Theory of Computation",                "credits": 3, "prereqs": ["CS 201", "CS 220"],     "prereqs_grades": {"CS 201": "C", "CS 220": "C"},  "coreqs": [],          "offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0009", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 330":  {"name": "Computer Networks",                    "credits": 3, "prereqs": ["CS 302"],               "prereqs_grades": {"CS 302": "C"},                 "coreqs": [],          "offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0010", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 340":  {"name": "Artificial Intelligence",              "credits": 3, "prereqs": ["CS 201", "MATH 210"],   "prereqs_grades": {"CS 201": "C", "MATH 210": "C"},"coreqs": [],          "offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0011", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 350":  {"name": "Programming Languages",                "credits": 3, "prereqs": ["CS 201", "CS 320"],     "prereqs_grades": {"CS 201": "C", "CS 320": "C"},  "coreqs": [],          "offered": ["Fall"],                     "special": "Fall only.",                                                      "chunk_id": "chunk_0012", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 401":  {"name": "Machine Learning",                     "credits": 3, "prereqs": ["CS 340", "MATH 301", "STAT 201"], "prereqs_grades": {"CS 340": "C", "MATH 301": "C", "STAT 201": "C"}, "coreqs": [], "offered": ["Fall", "Spring"], "special": "CS 340 is a strict prerequisite; no exceptions per catalog.",   "chunk_id": "chunk_0013", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 410":  {"name": "Compilers",                            "credits": 3, "prereqs": ["CS 302", "CS 350"],     "prereqs_grades": {"CS 302": "C", "CS 350": "C"},  "coreqs": [],          "offered": ["Spring"],                   "special": "Spring only.",                                                    "chunk_id": "chunk_0014", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 420":  {"name": "Distributed Systems",                  "credits": 3, "prereqs": ["CS 302", "CS 330"],     "prereqs_grades": {"CS 302": "C", "CS 330": "C"},  "coreqs": [],          "offered": ["Fall"],                     "special": "Fall only.",                                                      "chunk_id": "chunk_0015", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 430":  {"name": "Computer Security",                    "credits": 3, "prereqs": ["CS 302", "CS 330"],     "prereqs_grades": {"CS 302": "C", "CS 330": "C"},  "coreqs": [],          "offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0016", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 450":  {"name": "Senior Capstone Project I",            "credits": 3, "prereqs": ["CS 310"],               "prereqs_grades": {"CS 310": "B"},                 "coreqs": [],          "offered": ["Fall"],                     "special": "Fall only. Requires 90+ credit hours, Senior standing, and CS major GPA of 2.5+.", "chunk_id": "chunk_0017", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 451":  {"name": "Senior Capstone Project II",           "credits": 3, "prereqs": ["CS 450"],               "prereqs_grades": {"CS 450": "B"},                 "coreqs": [],          "offered": ["Spring"],                   "special": "Spring only.",                                                    "chunk_id": "chunk_0018", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 461":  {"name": "Natural Language Processing",          "credits": 3, "prereqs": ["CS 401"],               "prereqs_grades": {"CS 401": "C"},                 "coreqs": [],          "offered": ["Spring"],                   "special": "Spring only.",                                                    "chunk_id": "chunk_0019", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 470":  {"name": "Computer Graphics",                    "credits": 3, "prereqs": ["CS 201", "MATH 301"],   "prereqs_grades": {"CS 201": "C", "MATH 301": "C"},"coreqs": [],          "offered": ["Spring"],                   "special": "Spring only.",                                                    "chunk_id": "chunk_0020", "source": "https://catalog.university.edu/courses/computer-science"},
    "CS 480":  {"name": "Special Topics in Computer Science",   "credits": 3, "prereqs": ["CS 301"],               "prereqs_grades": {"CS 301": "C"},                 "coreqs": [],          "offered": ["Fall", "Spring"],           "special": "Instructor consent REQUIRED. Topic varies by semester.",          "chunk_id": "chunk_0021", "source": "https://catalog.university.edu/courses/computer-science"},
    "MATH 115":{"name": "Calculus I",                           "credits": 4, "prereqs": [],                       "prereqs_grades": {},                              "coreqs": [],          "offered": ["Fall", "Spring", "Summer"], "special": None,                                                              "chunk_id": "chunk_0030", "source": "https://catalog.university.edu/courses/mathematics"},
    "MATH 116":{"name": "Calculus II",                          "credits": 4, "prereqs": ["MATH 115"],             "prereqs_grades": {"MATH 115": "C"},               "coreqs": [],          "offered": ["Fall", "Spring"],           "special": None,                                                              "chunk_id": "chunk_0031", "source": "https://catalog.university.edu/courses/mathematics"},
    "MATH 210":{"name": "Discrete Mathematics",                 "credits": 3, "prereqs": ["MATH 115"],             "prereqs_grades": {"MATH 115": "C"},               "coreqs": [],          "offered": ["Fall", "Spring"],           "special": "Cross-listed as CS 220.",                                         "chunk_id": "chunk_0032", "source": "https://catalog.university.edu/courses/mathematics"},
    "MATH 250":{"name": "Linear Algebra",                       "credits": 3, "prereqs": ["MATH 115"],             "prereqs_grades": {"MATH 115": "C"},               "coreqs": [],          "offered": ["Fall", "Spring"],           "special": "NOT interchangeable with MATH 301 for CS 401.",                   "chunk_id": "chunk_0033", "source": "https://catalog.university.edu/courses/mathematics"},
    "MATH 301":{"name": "Linear Algebra (Advanced)",            "credits": 3, "prereqs": ["MATH 116"],             "prereqs_grades": {"MATH 116": "C"},               "coreqs": [],          "offered": ["Spring"],                   "special": "Spring only. Required specifically for CS 401.",                  "chunk_id": "chunk_0034", "source": "https://catalog.university.edu/courses/mathematics"},
    "STAT 201":{"name": "Probability and Statistics",           "credits": 3, "prereqs": ["MATH 115"],             "prereqs_grades": {"MATH 115": "C"},               "coreqs": [],          "offered": ["Fall", "Spring"],           "special": "Cross-listed as MATH 310.",                                       "chunk_id": "chunk_0035", "source": "https://catalog.university.edu/courses/mathematics"},
}

GRADE_ORDER = {
    "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "F": 0.0,
}

EQUIVALENCIES = {
    "CS 220":   "MATH 210", "MATH 210": "CS 220",
    "STAT 201": "MATH 310", "MATH 310": "STAT 201",
}


def _grade_val(grade: str) -> float:
    g = grade.strip().upper() if grade else "F"
    return GRADE_ORDER.get(g, -1.0)


def _grade_ok(earned: str, required: str) -> bool:
    return _grade_val(earned) >= _grade_val(required)


def _norm_code(code: str) -> str:
    code = code.strip().upper()
    return re.sub(r'([A-Z]+)(\d)', r'\1 \2', code)


def _get_completed(profile: dict) -> dict:
    """Return {COURSE_CODE: GRADE} including cross-list equivalents."""
    completed = {}
    for c in profile.get("completed_courses", []):
        if isinstance(c, dict):
            code  = _norm_code(c.get("code", ""))
            grade = c.get("grade", "C").strip().upper()
        else:
            parts = str(c).rsplit(" ", 1)
            code  = _norm_code(parts[0])
            grade = parts[1].upper() if len(parts) == 2 else "C"
        if code:
            completed[code] = grade
            eq = EQUIVALENCIES.get(code)
            if eq and eq not in completed:
                completed[eq] = grade
    return completed


def _check_prereqs(course_code: str, completed: dict) -> dict:
    code = _norm_code(course_code)
    if code not in COURSE_CATALOG:
        return {"found": False, "course": code, "decision": "UNKNOWN", "prereq_checks": []}

    info   = COURSE_CATALOG[code]
    checks = []
    all_ok = True

    for prereq in info["prereqs"]:
        min_g = info["prereqs_grades"].get(prereq, "C")
        alts  = [prereq, EQUIVALENCIES.get(prereq, prereq)]
        earned = next((completed[a] for a in alts if a in completed), None)

        if earned is None:
            ok     = False
            status = "✗ NOT SATISFIED — not completed"
            all_ok = False
        elif _grade_ok(earned, min_g):
            ok     = True
            status = f"✓ SATISFIED — earned {earned} (need {min_g} or better)"
        else:
            ok     = False
            status = f"✗ NOT SATISFIED — earned {earned}, need {min_g} or better"
            all_ok = False

        checks.append({
            "prereq":    prereq,
            "min_grade": min_g,
            "earned":    earned,
            "ok":        ok,
            "status":    status,
        })

    return {
        "found":         True,
        "course":        code,
        "name":          info["name"],
        "credits":       info["credits"],
        "prereq_checks": checks,
        "all_ok":        all_ok,
        "decision":      "ELIGIBLE" if all_ok else "NOT ELIGIBLE",
        "special":       info.get("special"),
        "coreqs":        info.get("coreqs", []),
        "offered":       info.get("offered", []),
        "chunk_id":      info["chunk_id"],
        "source":        info["source"],
    }


def _chain(course_code: str, visited: set = None, depth: int = 0) -> list:
    if visited is None:
        visited = set()
    code = _norm_code(course_code)
    if code in visited or depth > 6:
        return []
    visited.add(code)
    if code not in COURSE_CATALOG:
        return []
    info   = COURSE_CATALOG[code]
    result = [{
        "course":   code,
        "name":     info["name"],
        "prereqs":  info["prereqs"],
        "depth":    depth,
        "chunk_id": info["chunk_id"],
        "source":   info["source"],
    }]
    for p in info["prereqs"]:
        result.extend(_chain(p, visited, depth + 1))
    return result


def _extract_course(text: str) -> Optional[str]:
    m = re.findall(r'\b([A-Z]{2,5}\s*\d{3})\b', text.upper())
    return _norm_code(m[0]) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# SMART MOCK LLM
# ─────────────────────────────────────────────────────────────────────────────

class SmartMockLLM:
    name = "SmartMockLLM (Catalog-Grounded Demo)"

    def invoke(self, query: str, profile: dict, context_docs=None) -> str:
        q         = query.lower()
        completed = _get_completed(profile)

        # Abstention triggers
        if any(t in q for t in [
            "which professor", "what professor", "who teaches", "waitlist",
            "approval criteria", "what criteria", "blockchain", "quantum",
            "robotics", "summer 2025",
        ]):
            return self._abstain(query)

        # Chain questions
        if any(t in q for t in [
            "full chain", "complete chain", "prerequisite chain",
            "what do i need before", "minimum path", "full prerequisite",
        ]):
            target = _extract_course(query)
            if target:
                return self._chain_resp(target, completed)

        # Course plan
        if any(t in q for t in [
            "suggest", "plan", "recommend", "next semester",
            "next term", "what should i take", "course plan",
        ]):
            return self._plan_resp(profile, completed)

        # Policy questions
        if any(t in q for t in [
            "pass/fail", "pass fail", "residency", "how many credits",
            "total credits", "graduate", "elective requirement",
        ]):
            return self._policy_resp(q)

        # Prereq check for a specific course
        target = _extract_course(query)
        if target:
            return self._prereq_resp(target, completed, profile)

        return self._default_resp()

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _prereq_resp(self, course_code: str, completed: dict, profile: dict) -> str:
        r = _check_prereqs(course_code, completed)

        if not r["found"]:
            return self._abstain(course_code, f"Course {course_code} was not found in the catalog documents.")

        # ── Already completed? Show that clearly instead of eligibility ────────
        if course_code in completed:
            grade    = completed[course_code]
            chunk_id = r["chunk_id"]
            source   = r["source"]
            return f"""---
**Answer / Plan:**
📚 **Already Completed** — You have already completed **{course_code} — {r["name"]}** (Grade: {grade}).

You do not need to re-enroll in this course. It counts as a completed prerequisite for any courses that depend on it.

**Why (Requirements/Prereqs Satisfied):**
{course_code} appears in your completed courses list with a grade of {grade}. [{chunk_id} | {source}]

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
| {chunk_id} | {source} | {course_code} course details |

**Clarifying Questions (if needed):**
None.

**Assumptions / Not in Catalog:**
If you want to check eligibility for a course that *requires* {course_code}, ask: "Am I eligible for [next course]?"
---"""

        lines = []
        for chk in r["prereq_checks"]:
            lines.append(
                f"- **{chk['prereq']}** (min: {chk['min_grade']} or better): "
                f"{chk['status']} [{r['chunk_id']} | {r['source']}]"
            )
        if r.get("special"):
            lines.append(f"- ⚠️ Special condition: {r['special']} [{r['chunk_id']} | {r['source']}]")
        if r.get("coreqs"):
            lines.append(
                f"- Co-requisite(s): {', '.join(r['coreqs'])} — must be taken concurrently "
                f"or already completed [chunk_0050 | https://catalog.university.edu/policies/academic]"
            )

        prereq_text = "\n".join(lines) if lines else "No prerequisites — open enrollment."

        if r["decision"] == "ELIGIBLE":
            summary      = f"✅ **ELIGIBLE** — You meet all prerequisites for **{course_code} — {r['name']}**."
            offered_note = f" Offered in: {', '.join(r['offered'])}." if r.get("offered") else ""
            summary     += offered_note
        else:
            missing     = [c["prereq"] for c in r["prereq_checks"] if not c["ok"]]
            missing_str = ", ".join(missing)
            summary = (
                f"❌ **NOT ELIGIBLE** — You are missing: **{missing_str}**. "
                f"Complete those first, then you can enroll in {course_code}."
            )

        cit_rows = f"| {r['chunk_id']} | {r['source']} | {course_code} prerequisites |"
        if r.get("coreqs"):
            cit_rows += "\n| chunk_0050 | https://catalog.university.edu/policies/academic | Co-requisite policy §3.2 |"

        return f"""---
**Answer / Plan:**
{summary}

**Why (Requirements/Prereqs Satisfied):**
The catalog lists these prerequisites for {course_code} — {r['name']} [{r['chunk_id']} | {r['source']}]:

{prereq_text}

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
{cit_rows}

**Clarifying Questions (if needed):**
None — sufficient information provided.

**Assumptions / Not in Catalog:**
{"None." if not r.get("special") else "Verify special standing/GPA requirements with your academic advisor."}
---"""

    def _chain_resp(self, course_code: str, completed: dict) -> str:
        items = _chain(course_code)
        if not items:
            return self._abstain(course_code, f"{course_code} not found in catalog.")

        # Sort by depth (deepest first = foundations first)
        items_sorted = sorted(items, key=lambda x: -x["depth"])
        lines = []
        for item in items_sorted:
            indent      = "  " * (3 - item["depth"]) if item["depth"] <= 3 else ""
            prereqs_str = " → ".join(item["prereqs"]) if item["prereqs"] else "No prerequisites"
            tick        = "✅" if item["course"] in completed else ("🎯" if item["course"] == course_code else "⬜")
            lines.append(
                f"{indent}{tick} **{item['course']}** ({item['name']}) — needs: {prereqs_str} "
                f"[{item['chunk_id']} | {item['source']}]"
            )

        chain_display = "\n".join(lines)
        all_codes     = {i["course"] for i in items}
        done          = [c for c in all_codes if c in completed and c != course_code]
        needed        = [c for c in all_codes if c not in completed and c != course_code]

        cit_set   = {}
        for i in items:
            cit_set[i["chunk_id"]] = i["source"]
        cit_table = "\n".join(
            f"| {cid} | {src} | Prerequisites chain |"
            for cid, src in cit_set.items()
        )

        return f"""---
**Answer / Plan:**
Complete prerequisite chain for **{course_code}** ({items[0]['name']}):

{chain_display}

**Student Status:**
- ✅ Already completed: {', '.join(done) if done else 'None'}
- ⬜ Still needed: {', '.join(needed) if needed else 'All prerequisites already satisfied!'}

**Why (Requirements/Prereqs Satisfied):**
Each entry above was traced from the catalog's prerequisite listings.

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
{cit_table}

**Clarifying Questions (if needed):**
None.

**Assumptions / Not in Catalog:**
- Course availability by specific term not guaranteed. Check Schedule of Classes.
- Cross-listed equivalencies (e.g., CS 220 = MATH 210) honored per catalog.
---"""

    def _plan_resp(self, profile: dict, completed: dict) -> str:
        target_term = profile.get("target_term", "upcoming semester")
        max_cred    = int(profile.get("max_credits") or 15)
        track       = profile.get("track", "General CS")

        is_fall   = "fall"   in target_term.lower()
        is_spring = "spring" in target_term.lower()
        season    = "Fall" if is_fall else ("Spring" if is_spring else "Fall")

        sequence = [
            "CS 101", "CS 102", "CS 201", "CS 210", "CS 220",
            "CS 301", "CS 302", "CS 310", "CS 320", "CS 330",
            "CS 340", "CS 350", "CS 401", "CS 461", "CS 420",
            "CS 430", "CS 410", "CS 450", "CS 451",
            "MATH 115", "MATH 116", "MATH 210", "STAT 201", "MATH 301",
        ]

        recommended  = []
        total        = 0
        reason_lines = []
        cit_set      = {}

        for code in sequence:
            if total >= max_cred:
                break
            if code in completed or code not in COURSE_CATALOG:
                continue

            info = COURSE_CATALOG[code]

            # Skip if not offered this season
            if season not in info.get("offered", []):
                continue

            # Skip consent-required courses
            if info.get("special") and "consent" in info["special"].lower():
                continue

            r = _check_prereqs(code, completed)
            if not r["all_ok"]:
                continue

            if total + info["credits"] > max_cred:
                continue

            recommended.append(code)
            total += info["credits"]

            prereq_str = ", ".join(info["prereqs"]) if info["prereqs"] else "None"
            reason_lines.append(
                f"- **{code} — {info['name']}** ({info['credits']} cr): "
                f"Prerequisites ({prereq_str}) ✓ | Offered {season} ✓ "
                f"[{info['chunk_id']} | {info['source']}]"
            )
            cit_set[info["chunk_id"]] = info["source"]

        if not recommended:
            return """---
**Answer / Plan:**
No eligible, catalog-confirmed courses could be identified for the requested term.

**Why (Requirements/Prereqs Satisfied):**
Either all prerequisite-eligible courses are already completed, or the courses available in this term require prerequisites you haven't yet satisfied.

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
| chunk_0042 | https://catalog.university.edu/programs/cs-bscs | BSCS degree sequence |

**Clarifying Questions (if needed):**
1. Which courses have you already completed (with grades)?
2. Which semester are you targeting?

**Assumptions / Not in Catalog:**
None.
---"""

        course_list = "\n".join(
            f"{i+1}. **{c}** — {COURSE_CATALOG[c]['name']} ({COURSE_CATALOG[c]['credits']} cr)"
            for i, c in enumerate(recommended)
        )
        cit_table  = "\n".join(
            f"| {cid} | {src} | Course prereqs and {season} offering |"
            for cid, src in cit_set.items()
        )
        cit_table += "\n| chunk_0042 | https://catalog.university.edu/programs/cs-bscs | BSCS core requirements |"

        return f"""---
**Answer / Plan:**
Recommended plan for **{target_term}** ({track} track, max {max_cred} credits):

{course_list}

**Total: {total} credits**

**Why (Requirements/Prereqs Satisfied):**
{chr(10).join(reason_lines)}

All recommendations verified against completed courses and catalog offering data.

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
{cit_table}

**Clarifying Questions (if needed):**
None — sufficient profile information provided.

**Assumptions / Not in Catalog:**
- Section availability, times, and seat counts are NOT in the catalog — verify in the live Schedule of Classes.
- Instructor-consent courses (CS 480, CS 490) excluded as availability cannot be catalog-confirmed.
---"""

    def _policy_resp(self, q: str) -> str:
        if "pass" in q and "fail" in q:
            return """---
**Answer / Plan:**
**No** — CS core courses and required major courses **cannot** be taken Pass/Fail.

**Why (Requirements/Prereqs Satisfied):**
Academic Policy §1.4 states: "CS core courses and required major courses CANNOT be taken Pass/Fail." Only non-required elective courses are eligible. The election must be declared before the end of Week 2 and cannot be reversed. [chunk_0050 | https://catalog.university.edu/policies/academic]

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
| chunk_0050 | https://catalog.university.edu/policies/academic | Pass/Fail restrictions — Section 1.4 |

**Clarifying Questions (if needed):**
None.

**Assumptions / Not in Catalog:**
None.
---"""

        if "residency" in q:
            return """---
**Answer / Plan:**
The BSCS residency requirement has **two parts**:
1. At least **30 of the final 45 credit hours** must be earned at State University.
2. At least **18 credits in CS courses 300+** must be earned at State University.

**Why (Requirements/Prereqs Satisfied):**
Both stated in the BSCS Requirements (Residency Requirement section) [chunk_0042 | https://catalog.university.edu/programs/cs-bscs]. Transfer credits do NOT count toward residency [chunk_0042 | ...].

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
| chunk_0042 | https://catalog.university.edu/programs/cs-bscs | Residency requirement — 30/45 and 18 upper-div CS |

**Clarifying Questions (if needed):**
None.

**Assumptions / Not in Catalog:**
None.
---"""

        if "credit" in q or "graduate" in q:
            return """---
**Answer / Plan:**
The BS in Computer Science requires **120 total credit hours** to graduate.

**Why (Requirements/Prereqs Satisfied):**
Per the BSCS Program Requirements: "completion of 120 credit hours, including general education (30 cr), mathematics (16 cr), CS core (42 cr), concentration track (12 cr), and electives." [chunk_0042 | https://catalog.university.edu/programs/cs-bscs]

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
| chunk_0042 | https://catalog.university.edu/programs/cs-bscs | BSCS total 120 credit hours and breakdown |

**Clarifying Questions (if needed):**
None.

**Assumptions / Not in Catalog:**
None.
---"""

        return self._default_resp()

    def _abstain(self, query: str = "", reason: str = None) -> str:
        detail = reason or (
            "Instructor assignments, waitlist lengths, specific Special Topics topic offerings, "
            "and Summer availability (when not explicitly listed) are not contained in the "
            "academic catalog documents provided to this system."
        )
        return f"""---
**Answer / Plan:**
I don't have that information in the provided catalog/policies.

**Why (Requirements/Prereqs Satisfied):**
{detail}

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
| N/A | N/A | Topic not covered in catalog documents |

**Clarifying Questions (if needed):**
None.

**Assumptions / Not in Catalog:**
- This information is outside the scope of the academic catalog.
- **Recommended next steps:**
  1. Check the **Schedule of Classes** on the Registrar's website for real-time data
  2. Contact the **CS Department office** directly
  3. Email the specific **course instructor** for consent-related questions
  4. Visit your **academic advisor** for personalized guidance
---"""

    def _default_resp(self) -> str:
        return """---
**Answer / Plan:**
I can answer prerequisite eligibility questions, build prerequisite chains, and generate course plans — but I need a bit more context about your specific question.

**Why (Requirements/Prereqs Satisfied):**
No specific course code or planning keyword was detected in your question.

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
| N/A | N/A | Awaiting specific question |

**Clarifying Questions (if needed):**
1. Which specific course are you asking about? (e.g., "Can I take CS 302?")
2. Are you looking for a full semester plan, a prerequisite check, or a policy question?

**Assumptions / Not in Catalog:**
None.
---"""


# ─────────────────────────────────────────────────────────────────────────────
# REAL OPENAI LLM (optional)
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(use_mock: bool = True):
    if use_mock:
        return SmartMockLLM()

    # ── Try Groq first (faster, free tier available) ──────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                api_key=groq_key,
            )
            llm.name = "Groq / llama-3.3-70b-versatile"
            print("Using Groq LLM (llama-3.3-70b-versatile)")
            return llm
        except ImportError:
            print("langchain-groq not installed. Run: pip install langchain-groq")

    # ── Fallback: OpenAI ──────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)
            llm.name = "OpenAI / gpt-4o"
            print("Using OpenAI LLM (gpt-4o)")
            return llm
        except ImportError:
            print("langchain-openai not installed.")

    print("No API key found (GROQ_API_KEY or OPENAI_API_KEY) — using SmartMockLLM.")
    return SmartMockLLM()


# ─────────────────────────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────────────────────────

class IntakeAgent:
    def process(self, query: str, profile: dict) -> Tuple[bool, List[str]]:
        q         = query.lower()
        has_courses = bool(profile.get("completed_courses"))
        has_term    = bool(profile.get("target_term"))
        has_major   = bool(profile.get("major"))

        is_prereq  = any(t in q for t in ["eligible", "can i take", "prereq", "what do i need"])
        is_plan    = any(t in q for t in ["plan", "suggest", "recommend", "next semester", "next term"])
        is_policy  = any(t in q for t in ["pass/fail", "residency", "credit", "graduate", "policy"])
        is_abstain = any(t in q for t in ["professor", "waitlist", "who teaches", "available in summer"])

        clarifying = []

        # For prereq checks we need completed courses
        if is_prereq and not has_courses:
            clarifying.append("Which courses have you already completed? Please include grades (e.g., CS 101 - A).")

        # For plans we need courses + term
        if is_plan and not has_courses:
            clarifying.append("Which courses have you already completed (with grades)?")
        if is_plan and not has_term:
            clarifying.append("Which semester are you planning for (e.g., Fall 2025)?")

        # Sufficient = can proceed without clarification
        if is_abstain or is_policy:
            return True, []
        if is_prereq and has_courses:
            return True, []
        if is_plan and has_courses and has_term:
            return True, []
        if not is_prereq and not is_plan:
            return True, []

        return len(clarifying) == 0, clarifying


class CatalogRetrieverAgent:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, profile: dict) -> List[Document]:
        queries = [query]
        target  = _extract_course(query)
        if target:
            queries.append(f"{target} prerequisites requirements credits")
        if profile.get("major"):
            queries.append("CS degree requirements core courses")

        all_docs, seen = [], set()
        for q in queries[:3]:
            for d in retrieve_context(self.vectorstore, q, k=4):
                cid = d.metadata.get("chunk_id", "")
                if cid not in seen:
                    all_docs.append(d)
                    seen.add(cid)
        return all_docs[:8]


class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm

    def answer(self, query: str, profile: dict, context_docs: List[Document]) -> str:
        if isinstance(self.llm, SmartMockLLM):
            return self.llm.invoke(query, profile, context_docs)

        # Real LLM path
        profile_str = build_student_profile_string(profile)
        context_str = format_retrieved_chunks_for_prompt(context_docs)

        # Detect abstention queries — remind LLM to keep answer short
        q_lo = query.lower()
        abstention_topics = [
            "professor", "instructor", "who teaches", "taught by",
            "waitlist", "waitlist length", "seats available",
            "room number", "section time", "class time",
            "summer 2025", "summer 2026",
        ]
        is_abstention = any(t in q_lo for t in abstention_topics)
        abstention_reminder = (
            "\n\nIMPORTANT: This question asks about information that is NOT in the academic catalog "
            "(instructor assignments, waitlists, room/section details, or unconfirmed semester availability). "
            "You MUST answer with ONLY: state the info is not in the catalog, list 2-3 places to check. "
            "Do NOT ask clarifying questions. Do NOT add a Why section. Keep the answer brief."
        ) if is_abstention else ""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"STUDENT PROFILE:\n{profile_str}\n\n"
                f"QUESTION:\n{query}\n\n"
                f"CATALOG CONTEXT:\n{context_str}\n\n"
                f"{FORMAT_INSTRUCTIONS}"
                f"{abstention_reminder}"
            )},
        ]
        try:
            result = self.llm.invoke(messages)
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            err = str(e)

            # ── Auth errors ───────────────────────────────────────────────────
            if "401" in err or "invalid_api_key" in err or "Authentication" in err:
                print(f"\n⚠️  API key error: {err}")
                print("Falling back to SmartMockLLM (demo mode)...")
                self.llm = SmartMockLLM()
                return self.llm.invoke(query, profile, context_docs)

            # ── Rate-limit / quota errors (Groq TPD, OpenAI RPM/TPM, etc.) ───
            is_rate_limit = (
                "429"              in err
                or "rate_limit"    in err.lower()
                or "rate limit"    in err.lower()
                or "quota"         in err.lower()
                or "RateLimitError" in type(e).__name__
            )
            if is_rate_limit:
                # Try to surface the retry-after hint from the error message
                import re as _re
                wait_match = _re.search(r'try again in ([^\.\n]+)', err, _re.IGNORECASE)
                wait_hint  = f" (retry after {wait_match.group(1)})" if wait_match else ""
                print(f"\n⚠️  Rate limit hit{wait_hint}: {err}")
                print("Falling back to SmartMockLLM (demo mode)...")
                self.llm = SmartMockLLM()
                return self.llm.invoke(query, profile, context_docs)

            raise


class VerifierAgent:
    def verify(self, query: str, answer: str, context_docs: List[Document]) -> Tuple[str, str]:
        issues = []
        if "chunk_" not in answer and "N/A" not in answer:
            issues.append("No chunk citations found in the response.")
        for phrase in ["i believe", "i think", "probably", "likely offered"]:
            if phrase in answer.lower():
                issues.append(f"Possible unsupported claim detected: '{phrase}'")

        if issues:
            warn = "\n".join(f"  ⚠️ {i}" for i in issues)
            return "PASS_WITH_WARNINGS", answer + f"\n\n**Verifier Notes:**\n{warn}"
        return "PASS", answer


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class CourseAdvisorOrchestrator:
    def __init__(self, use_mock_llm: bool = True, force_rebuild_index: bool = False):
        print("Initializing Course Advisor Assistant...")
        self.vectorstore = build_index(force_rebuild=force_rebuild_index)
        llm       = get_llm(use_mock=use_mock_llm)
        llm_label = getattr(llm, "name", None) or getattr(llm, "model_name", None) or type(llm).__name__
        print(f"LLM: {llm_label}")
        self.intake    = IntakeAgent()
        self.retriever = CatalogRetrieverAgent(self.vectorstore)
        self.planner   = PlannerAgent(llm)
        self.verifier  = VerifierAgent()
        print("Ready!\n")

    def answer(
        self,
        student_query: str,
        student_profile: Optional[dict] = None,
        skip_intake: bool = False,
    ) -> Dict[str, Any]:
        if student_profile is None:
            student_profile = {}

        result = {
            "query":                student_query,
            "student_profile":      student_profile,
            "clarifying_questions": [],
            "answer":               "",
            "audit_result":         "N/A",
            "context_chunks_used":  [],
        }

        # Step 1: Intake
        if not skip_intake:
            sufficient, clarifying_qs = self.intake.process(student_query, student_profile)
            result["clarifying_questions"] = clarifying_qs
            if not sufficient and clarifying_qs:
                qs = "\n".join(f"{i+1}. {q}" for i, q in enumerate(clarifying_qs))
                result["answer"] = f"""---
**Answer / Plan:**
To give you a fully catalog-grounded answer, I need a bit more information.

**Why (Requirements/Prereqs Satisfied):**
Without complete profile details, I cannot verify prerequisites or generate a reliable plan.

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
| N/A | N/A | Awaiting student profile |

**Clarifying Questions (if needed):**
{qs}

**Assumptions / Not in Catalog:**
Please answer the questions above so I can provide a grounded, cited response.
---"""
                return result

        # Step 2: Retrieve
        context_docs = self.retriever.retrieve(student_query, student_profile)
        result["context_chunks_used"] = [d.metadata.get("chunk_id", "?") for d in context_docs]

        # Step 3: Plan
        raw_answer = self.planner.answer(student_query, student_profile, context_docs)

        # Step 4: Verify
        audit, final       = self.verifier.verify(student_query, raw_answer, context_docs)
        result["answer"]       = final
        result["audit_result"] = audit
        return result