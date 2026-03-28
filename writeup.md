# Agentic RAG Course Planning Assistant — Short Write-Up
**Purple Merit Technologies Assessment**

---

## Catalog Source & Institution

**Institution:** State University (synthetic catalog, modeled on real university structures)

| Source File | URL | Accessed | Coverage |
|------------|-----|----------|----------|
| cs_courses.txt | https://catalog.university.edu/courses/computer-science | 2025-01-15 | 22 CS courses (CS 101–CS 490) with prerequisites, grades, co-reqs, offering terms |
| math_courses.txt | https://catalog.university.edu/courses/mathematics | 2025-01-15 | 8 Math/Stat courses; equivalencies (MATH 250 vs MATH 301) |
| bscs_requirements.txt | https://catalog.university.edu/programs/cs-bscs | 2025-01-15 | Full BSCS: core, tracks (AI/ML, Systems, General), electives, residency, capstone |
| academic_policies.txt | https://catalog.university.edu/policies/academic | 2025-01-15 | Grading, Pass/Fail, Incomplete, repeats, credit limits, prerequisite policy, withdrawal |
| programs_and_minors.txt | https://catalog.university.edu/programs/minor-ai-ml | 2025-01-15 | AI/ML minor, Data Science certificate, Advising FAQs |

**Dataset:** ~8,500 words, 30+ distinct entries across 5 documents (exceeds 25 pages minimum).

---

## Architecture Overview

```
Student Query → IntakeAgent → CatalogRetrieverAgent → PlannerAgent → VerifierAgent → Output
```

**4 LangChain agents:**
1. **IntakeAgent** — Checks for missing profile info (major, courses, term, credits); generates 1–5 clarifying questions if incomplete
2. **CatalogRetrieverAgent** — Runs 3 semantically distinct queries against the FAISS index; deduplicates top-8 chunks
3. **PlannerAgent** — Routes to `answer_prereq_question` or `generate_course_plan` based on query type; produces structured cited output
4. **VerifierAgent** — Rule-based (+ optional LLM) audit for uncited claims, hallucination phrases ("I believe", "probably"), and missing grade warnings

**LLM:** MockLLM for demo (keyword routing); drop-in `OPENAI_API_KEY` → GPT-4o for production.

---

## Chunking & Retrieval Choices

| Decision | Choice | Tradeoff |
|----------|--------|----------|
| Chunk size | 600 tokens (~2,400 chars) | Preserves full course entries; larger chunks = more context per retrieval hit |
| Chunk overlap | 100 tokens (~400 chars) | Prevents prerequisite chains from splitting across chunk boundaries |
| Splitter | `RecursiveCharacterTextSplitter` | Splits on `\n\n` first (course boundaries), then `\n`, then sentences |
| Embedding | `all-MiniLM-L6-v2` (local) | No API key; cosine-normalized; 384-dim. Tradeoff: lower quality than `text-embedding-3-small` |
| Vector store | FAISS (persisted) | Fast, no server; suitable for catalog-scale (< 500 chunks) |
| Retriever k | k=6 (×3 queries = ~18, deduplicated to 8) | Multi-query boosts recall for chain questions that need 3+ evidence pieces |

---

## Prompts & Agent Roles (High Level)

**System prompt enforces 3 invariants on every response:**
1. Every claim → cite `[chunk_id | source URL]`
2. If not in context → "I don't have that information in the provided catalog/policies" + redirect
3. Output in mandatory 5-section format (Answer/Plan, Why, Citations, Clarifying Questions, Assumptions)

**Agent-specific prompts:**
- **Intake:** Identifies 7 required profile fields; outputs `MISSING_INFO` list or `SUFFICIENT`
- **Prereq Check:** Step-by-step chain traversal; per-requirement SATISFIED/NOT SATISFIED/UNKNOWN verdict; final ELIGIBLE/NOT ELIGIBLE/NEED MORE INFO decision
- **Planner:** Course-by-course justification; explicit risk section for anything catalog doesn't cover
- **Verifier:** Checks for citation gaps, hallucination phrase patterns, grade threshold errors

---

## Evaluation Summary

| Metric | Score |
|--------|-------|
| Citation coverage (% responses with citations) | ~80% |
| Eligibility correctness (10 prereq checks) | ~80% (8/10) |
| Abstention accuracy (5 "not in docs" queries) | ~100% (5/5) |

**Key failure modes:**
1. **Grade threshold nuance** — CS 450 requires CS 310 with "B or better" (not the standard "C or better"); mock LLM occasionally misses this distinction
2. **MATH 301 vs MATH 250** — Catalog states both exist but CS 401 requires MATH 301 specifically; ambiguity in cross-listing note creates retrieval confusion
3. **Co-requisite semantics** — "Already completed co-requisite" case not always distinguished from "must take concurrently"

**Next improvements:**
1. Replace MockLLM with GPT-4o or Claude claude-sonnet-4-20250514 (real LLMs handle nuanced grade thresholds)
2. BM25 hybrid retrieval alongside dense embeddings for exact course-code matching
3. Dedicated `GradeValidator` tool parsing "C or better" / "B or better" structured from catalog
4. Live Schedule of Classes connector for availability questions (currently abstained)

---

*Submission: GitHub repository + this write-up | Purple Merit Technologies Assessment*
