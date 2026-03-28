"""
demo_app.py — Streamlit UI for the Agentic RAG Course Planning Assistant
Run with:  streamlit run demo/demo_app.py
"""

import sys, re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "tests"))

import streamlit as st

st.set_page_config(page_title="Course Advisor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.app-title {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-subtitle { color: #94a3b8; font-size: 0.95rem; margin-top: 2px; }
.chip-eligible    { background:#166534; color:#bbf7d0; padding:3px 12px; border-radius:999px; font-size:0.82rem; font-weight:700; display:inline-block; margin-bottom:8px; }
.chip-not-eligible{ background:#7f1d1d; color:#fecaca; padding:3px 12px; border-radius:999px; font-size:0.82rem; font-weight:700; display:inline-block; margin-bottom:8px; }
.chip-abstain     { background:#1e3a5f; color:#bae6fd; padding:3px 12px; border-radius:999px; font-size:0.82rem; font-weight:700; display:inline-block; margin-bottom:8px; }
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.82rem; }
div[data-testid="stExpander"] { border: 1px solid #1e293b; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = ""
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "query_box" not in st.session_state:
    st.session_state["query_box"] = ""


# ── Orchestrator (cached — builds FAISS index only once) ─────────────────────
@st.cache_resource(show_spinner="Building catalog index… (first run only)")
def load_orchestrator():
    import os
    from agents.agents import CourseAdvisorOrchestrator
    has_key = bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))
    return CourseAdvisorOrchestrator(use_mock_llm=not has_key)


def parse_courses(raw: str) -> list:
    courses = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2 and re.match(r'^[A-Fa-f][+-]?$', parts[1]):
            courses.append({"code": parts[0].strip(), "grade": parts[1].strip().upper()})
        else:
            courses.append({"code": line, "grade": "?"})
    return courses


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Profile")
    st.caption("Fill in your details for accurate, cited answers.")

    # LLM mode indicator
    import os
    groq_key   = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if groq_key:
        st.success("🚀 LLM: Groq (llama-3.3-70b)")
    elif openai_key:
        st.info("🤖 LLM: OpenAI (gpt-4o)")
    else:
        st.warning("⚙️ LLM: Demo mode (rule-based)")
        st.caption("Set GROQ_API_KEY for AI-powered answers.")

    major         = st.selectbox("Major", ["BS Computer Science", "BA Computer Science", "Minor in AI/ML", "Undeclared"])
    track         = st.selectbox("Concentration Track", ["General CS", "AI/ML", "Systems & Security", "Not decided"])
    catalog_year  = st.selectbox("Catalog Year", ["2024-2025", "2023-2024"])
    target_term   = st.selectbox("Target Term", ["Fall 2025", "Spring 2025", "Fall 2026", "Spring 2026"])
    max_credits   = st.slider("Max Credits This Term", 9, 18, 15)
    credits_done  = st.number_input("Credits Completed So Far", 0, 135, 60)
    cs_gpa        = st.number_input("CS Major GPA", 0.0, 4.0, 3.0, step=0.1)

    st.markdown("---")
    st.markdown("**Completed Courses**")
    st.caption("One per line: `COURSE_CODE GRADE`  \ne.g. `CS 101 A`")
    completed_raw = st.text_area(
        "courses_input",
        value="CS 101 A\nCS 102 B+\nCS 201 B\nCS 210 B\nCS 220 A\nMATH 115 B+\nSTAT 201 B",
        height=200,
        label_visibility="collapsed",
    )

profile = {
    "major": major, "track": track, "catalog_year": catalog_year,
    "target_term": target_term, "max_credits": max_credits,
    "credits_completed": credits_done, "cs_gpa": cs_gpa,
    "completed_courses": parse_courses(completed_raw),
}


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">Course Planning Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Agentic RAG · Every answer is grounded in the catalog with citations · No guessing</div>', unsafe_allow_html=True)
st.markdown("---")


# ── EXAMPLE BUTTONS — clicking sets pending_query AND triggers a rerun ────────
EXAMPLES = [
    "Am I eligible for CS 102?",
    "Can I take CS 302 (Operating Systems)?",
    "Am I eligible for CS 340 (Artificial Intelligence)?",
    "Can I take CS 450 (Capstone) if I got a C in CS 310?",
    "What is the full prerequisite chain for CS 401 Machine Learning?",
    "What do I need before I can take CS 410 (Compilers)?",
    "Suggest a course plan for my next semester.",
    "Can I take CS core courses on Pass/Fail?",
    "What is the residency requirement for the BSCS?",
    "How many total credits do I need to graduate?",
    "Which professor teaches CS 340 next fall?",
    "Is CS 401 available in Summer 2025?",
]

with st.expander("💡 Example questions — click to try", expanded=True):
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            # Store as pending and immediately process
            st.session_state["pending_query"] = ex
            st.session_state["query_box"]     = ex


# ── QUERY INPUT ───────────────────────────────────────────────────────────────
query = st.text_area(
    "✏️ Or type your question:",
    value=st.session_state.get("query_box", ""),
    height=80,
    placeholder="e.g.  Can I take CS 302 this semester?",
    key="typed_query",
)

col1, col2 = st.columns([1, 5])
ask_clicked = col1.button("🔍 Ask", type="primary", use_container_width=True)

# Resolve which query to run:
#   - typed_query takes priority if user typed something
#   - else pending_query (set by example button)
active_query = query.strip() or st.session_state.get("pending_query", "").strip()

run_query = ask_clicked or bool(st.session_state.get("pending_query", ""))


# ── RUN ────────────────────────────────────────────────────────────────────────
if run_query and active_query:
    # Clear pending so it doesn't re-run on next interaction
    st.session_state["pending_query"] = ""

    orchestrator = load_orchestrator()

    with st.spinner(f"Checking catalog for: *{active_query}*"):
        result = orchestrator.answer(
            student_query=active_query,
            student_profile=profile,
            skip_intake=False,
        )
    st.session_state["last_result"] = result
    st.session_state["last_query"]  = active_query

elif ask_clicked and not active_query:
    st.warning("Please enter a question or click an example above.")


# ── DISPLAY RESULT ─────────────────────────────────────────────────────────────
result = st.session_state.get("last_result")
if result:
    answer     = result.get("answer", "")
    clarifying = result.get("clarifying_questions", [])
    audit      = result.get("audit_result", "N/A")
    chunks     = result.get("context_chunks_used", [])
    last_q     = st.session_state.get("last_query", "")

    st.markdown(f"---\n**Query:** *{last_q}*")

    # Status chip — only show for direct eligibility decisions, not chain/plan/policy
    ans_lo = answer.lower()
    q_lo   = last_q.lower()
    is_chain_or_plan = any(t in q_lo for t in [
        "chain", "full prerequisite", "complete prerequisite",
        "suggest", "plan", "recommend", "next semester", "next term",
        "how many", "residency", "pass/fail", "total credits",
    ])
    if not is_chain_or_plan:
        if "already completed" in ans_lo:
            st.markdown('<span style="background:#E6F1FB;color:#0C447C;padding:2px 10px;border-radius:999px;font-size:0.8rem;font-weight:600">📚 ALREADY COMPLETED</span>', unsafe_allow_html=True)
        elif "decision: eligible" in ans_lo and "not eligible" not in ans_lo:
            st.markdown('<span class="chip-eligible">✅ ELIGIBLE</span>', unsafe_allow_html=True)
        elif "not eligible" in ans_lo or "not satisfied" in ans_lo:
            st.markdown('<span class="chip-not-eligible">❌ NOT ELIGIBLE</span>', unsafe_allow_html=True)
    if "don't have that information" in ans_lo or "not covered in catalog" in ans_lo or "outside the scope" in ans_lo:
        st.markdown('<span class="chip-abstain">🔍 NOT IN CATALOG</span>', unsafe_allow_html=True)

    # Clarifying questions
    if clarifying:
        st.info("**I need a bit more information:**\n\n" + "\n".join(f"- {q}" for q in clarifying))

    # Main answer
    if answer:
        st.markdown("### 📋 Answer")
        st.markdown(answer)

    # Debug expander
    with st.expander("🔧 Debug Info"):
        c1, c2 = st.columns(2)
        c1.markdown(f"**Audit:** `{audit}`")
        c2.markdown(f"**Chunks used:** `{', '.join(chunks) if chunks else 'none'}`")
        st.json({"major": profile["major"], "track": profile["track"],
                 "target_term": profile["target_term"], "max_credits": profile["max_credits"],
                 "completed_courses": profile["completed_courses"]})


# ── EXAMPLE TRANSCRIPTS ────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Example Transcripts"):
    t1, t2, t3 = st.tabs(["✅ Eligible Decision", "📅 Course Plan", "🚫 Abstention"])

    with t1:
        st.markdown("""
**Query:** Am I eligible for CS 302 (Operating Systems)?
**Profile:** CS 101 A, CS 102 B+, CS 201 B, CS 210 C+

---
**Decision: ELIGIBLE** ✅

- **CS 201** (min: C): ✓ SATISFIED — earned B [chunk_0007 | catalog/cs]
- **CS 210** (min: C): ✓ SATISFIED — earned C+ [chunk_0007 | catalog/cs]

Offered in: Fall, Spring [chunk_0007]
        """)

    with t2:
        st.markdown("""
**Query:** Suggest a course plan for Fall 2025
**Profile:** CS 101–CS 220, MATH 115, STAT 201 all completed

---
**Recommended:**
1. CS 301 — Database Systems (3 cr) [chunk_0006]
2. CS 302 — Operating Systems (3 cr) [chunk_0007]
3. CS 310 — Software Engineering (3 cr) [chunk_0008]
4. CS 340 — Artificial Intelligence (3 cr) [chunk_0011]

**Total: 12 credits** — all prerequisites verified with citations.
        """)

    with t3:
        st.markdown("""
**Query:** Which professor teaches CS 340 next fall?

---
I don't have that information in the provided catalog/policies.

Instructor assignments are not in the catalog documents.

**Next steps:** Check the live Schedule of Classes · Email CS Department · Visit your advisor.
        """)