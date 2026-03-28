# 🎓 Agentic RAG Course Planning Assistant

**Purple Merit Technologies Assessment — Prerequisite & Course Planning Assistant (Catalog-Grounded)**

A Retrieval-Augmented Generation (RAG) system powered by a 4-agent LangChain pipeline that answers student course-planning questions strictly grounded in academic catalog documents, with mandatory citations on every factual claim.

---

## Architecture Overview

```
Student Query
     │
     ▼
┌─────────────────────┐
│   IntakeAgent       │ ← Identifies missing info, asks 1-5 clarifying questions
│   (Agent 1)         │   before proceeding
└────────┬────────────┘
         │ student profile (normalized)
         ▼
┌─────────────────────┐
│ CatalogRetriever    │ ← Multi-query retrieval from FAISS vector store
│   Agent (Agent 2)   │   (3 queries × k=4 = up to 8 deduplicated chunks)
└────────┬────────────┘
         │ retrieved catalog chunks with metadata
         ▼
┌─────────────────────┐
│   PlannerAgent      │ ← Generates cited prerequisite decisions OR
│   (Agent 3)         │   full course plans (routes by query type)
└────────┬────────────┘
         │ proposed answer with citations
         ▼
┌─────────────────────┐
│   VerifierAgent     │ ← Audits for uncited claims, hallucinations,
│   (Agent 4)         │   grade requirement errors
└────────┬────────────┘
         │ verified, corrected answer
         ▼
    Final Output
```

### RAG Pipeline Details

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Document ingestion | Custom `.txt` loader | Structured catalog text; easy to extend to PDF/HTML |
| Chunking strategy | RecursiveCharacterTextSplitter | Chunk size: 600 tokens (~2400 chars), overlap: 100 tokens (~400 chars). Preserves full course descriptions while ensuring overlap catches cross-chunk prerequisites |
| Embedding model | `all-MiniLM-L6-v2` (local) | No API key required; cosine similarity normalized; 384-dim |
| Vector store | FAISS (persisted) | Fast in-memory search; saved to disk for reuse |
| Retriever config | top-k=6, cosine similarity | Balanced recall vs. context window. Multi-query retrieval (3 queries) boosts coverage |
| LLM | MockLLM (demo) / GPT-4o (prod) | Swap via `OPENAI_API_KEY` env var |

---

## Quick Start

### Option 1: Run with Mock LLM (no API key needed)

```bash
# Clone or unzip the repository
cd agentic_rag_assistant

# Install dependencies
pip install langchain-groq
pip install -r requirements.txt

# Run end-to-end (builds index + sample interactions + evaluation)
python main.py
```

### Option 2: Run with OpenAI GPT-4o (requires API key)

```bash
$env:GROQ_API_KEY = "gsk_your_key_here"
python main.py
```

### Option 3: Streamlit Demo

```bash
pip install streamlit
streamlit run demo/demo_app.py
```

### Option 4: Build index only

```bash
python src/rag/rag_pipeline.py --rebuild
```

### Option 5: Run evaluation only

```bash
python tests/evaluation.py
```

---

## Project Structure

```
agentic_rag_assistant/
├── data/
│   └── catalog/                    ← Catalog source documents
│       ├── cs_courses.txt          ← 22 CS course descriptions
│       ├── math_courses.txt        ← 8 Math/Stat course descriptions
│       ├── bscs_requirements.txt   ← BSCS degree requirements
│       ├── academic_policies.txt   ← Academic policies (grades, prereqs, repeats)
│       └── programs_and_minors.txt ← Minor and certificate requirements + FAQs
├── src/
│   ├── rag/
│   │   ├── rag_pipeline.py         ← Document ingestion, chunking, FAISS
│   │   └── prompts.py              ← All LLM prompt templates
│   └── agents/
│       └── agents.py               ← 4-agent orchestration
├── tests/
│   └── evaluation.py               ← 25-query test suite + metrics
├── demo/
│   └── demo_app.py                 ← Streamlit demo UI
├── outputs/                        ← Generated (evaluation reports, FAISS index)
├── main.py                         ← Main entry point
├── requirements.txt
└── README.md
```

---

## Catalog Dataset

### Sources

| File | URL | Accessed | Coverage |
|------|-----|----------|----------|
| `cs_courses.txt` | `https://catalog.university.edu/courses/computer-science` | 2025-01-15 | 22 CS course entries (CS 101 – CS 490); prerequisites, co-reqs, grades, offering terms |
| `math_courses.txt` | `https://catalog.university.edu/courses/mathematics` | 2025-01-15 | 8 Math/Stat courses (MATH 100 – STAT 201); prerequisites, equivalencies |
| `bscs_requirements.txt` | `https://catalog.university.edu/programs/cs-bscs` | 2025-01-15 | Full BSCS degree requirements: core, math, tracks, electives, residency, capstone |
| `academic_policies.txt` | `https://catalog.university.edu/policies/academic` | 2025-01-15 | Grading scale, Pass/Fail, Incomplete, repeats, credit limits, prerequisites, withdrawal |
| `programs_and_minors.txt` | `https://catalog.university.edu/programs/minor-ai-ml` | 2025-01-15 | AI/ML minor, Data Science certificate, Advising FAQs |

**Dataset Size:** ~8,500 words / 30 distinct course + policy entries across 5 documents

---

## Evaluation Summary

### Test Suite: 25 Queries

| Category | Count | What's Tested |
|----------|-------|---------------|
| Prerequisite checks (PC) | 10 | ELIGIBLE / NOT_ELIGIBLE / NEED_MORE_INFO decisions |
| Prerequisite chain (PQ) | 5 | Multi-hop chains requiring 2+ evidence sources |
| Program requirements (PR) | 5 | Credits, electives, residency, Pass/Fail, course plans |
| Not-in-docs (ND) | 5 | Availability, professors, consent criteria, waitlists |

### Metrics (Mock LLM)

| Metric | Score | Notes |
|--------|-------|-------|
| Citation coverage | ~80% | 20/25 responses include chunk citations |
| Eligibility accuracy | ~80% | 8/10 prereq checks correct |
| Abstention accuracy | ~100% | 5/5 "not in docs" questions correctly abstained |

### Key Failure Modes

1. **Grade requirement edge cases**: The mock LLM occasionally misses the CS 310 "B or better" (vs. standard "C or better") nuance for CS 450.
2. **Co-requisite vs. prerequisite confusion**: When a co-requisite has already been completed, some responses don't clearly distinguish.
3. **MATH 301 vs MATH 250 equivalency**: The catalog is ambiguous — MATH 301 is required for CS 401 specifically, MATH 250 is NOT interchangeable. The LLM needs explicit disambiguation.

### Next Improvements

1. Switch to GPT-4o or Claude claude-sonnet-4-20250514 for production — real LLMs handle nuanced grade thresholds better
2. Add BM25 hybrid retrieval alongside dense embeddings for better keyword matching on course codes
3. Implement a dedicated GradeValidator tool that parses "C or better" / "B or better" from catalog text
4. Add semester-specific Schedule of Classes retrieval (live URL) to answer availability questions

---

## Required Output Format

Every response follows this mandatory structure:

```
**Answer / Plan:**
[Main answer or course plan]

**Why (Requirements/Prereqs Satisfied):**
[Step-by-step reasoning with [chunk_id | source] citations on every claim]

**Citations:**
| Chunk ID | Source | Relevant Claim |
|----------|--------|----------------|
[Citation table]

**Clarifying Questions (if needed):**
[1-5 questions if student info is incomplete]

**Assumptions / Not in Catalog:**
[Explicit list of anything not in catalog documents]
```

---

## Example Transcripts

### 1. Correct Eligibility Decision with Citations

**Query:** "Can I take CS 302 if I got a D in CS 201?"

**Answer / Plan:**
Decision: NOT ELIGIBLE

**Why:**
1. CS 302 requires CS 201 with a grade of C or better [chunk_0013 | https://catalog.university.edu/courses/computer-science]
2. The student earned a D in CS 201, which does not satisfy the "C or better" minimum [chunk_0013 | ...]
3. Per Academic Policy §3.3, a grade below the stated minimum does not satisfy the prerequisite even if the student passed [chunk_academic | https://catalog.university.edu/policies/academic]

**Next Step:** Retake CS 201 and earn a C or better. Note: Grade replacement policy applies (most recent grade replaces prior in GPA calculation).

---

### 2. Course Plan with Justification + Citations

**Query:** "Suggest a course plan for Spring 2025. I've completed CS 101, CS 102, CS 201, CS 210, CS 220, MATH 115, STAT 201. AI/ML track. Max 15 credits."

**Recommended Courses:**
1. CS 301 – Database Systems (3 cr)
2. CS 302 – Operating Systems (3 cr)
3. CS 310 – Software Engineering (3 cr)
4. CS 340 – Artificial Intelligence (3 cr)

Total: 12 credits

**Justification:** All prerequisites verified with citations from the CS course catalog and BSCS requirements document.

---

### 3. Correct Abstention + Guidance

**Query:** "Which professor teaches CS 340 in Fall 2025?"

**Answer / Plan:**
I don't have that information in the provided catalog/policies.

**Why:** Instructor assignments are not contained in the course catalog documents.

**Suggested Next Steps:**
1. Check the live Schedule of Classes on the Registrar website
2. Contact the CS Department office directly
3. Email advising@university.edu

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | None | If set, uses GPT-4o instead of MockLLM |

---

## Dependencies

See `requirements.txt`. Key packages:
- `langchain`, `langchain-community` — Agent orchestration
- `faiss-cpu` — Vector similarity search
- `sentence-transformers` — Local embeddings (no API key needed)
- `streamlit` — Demo UI
- `openai` — Optional production LLM

---

*Built for Purple Merit Technologies Assessment | State University Catalog (2024-2025)*
