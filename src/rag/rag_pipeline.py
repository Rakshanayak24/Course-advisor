"""
rag_pipeline.py
---------------
Core RAG pipeline for the Agentic Course Planning Assistant.

KEY FIX: Path resolution
  This file lives at:  <project_root>/src/rag/rag_pipeline.py
  Path(__file__).parent        = src/rag/
  Path(__file__).parent.parent = src/
  Path(__file__).parent.parent.parent = <project_root>   ← correct root
  DATA_DIR  = <project_root>/data/catalog/
  INDEX_DIR = <project_root>/outputs/faiss_index/
"""

import os
from pathlib import Path
from typing import List, Optional

# ── LangChain imports ────────────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    # prefer the newer dedicated package (no deprecation warning)
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


# ── Path constants ────────────────────────────────────────────────────────────
# Resolve from THIS file's absolute location so it works regardless of cwd
_THIS_FILE   = Path(__file__).resolve()          # .../src/rag/rag_pipeline.py
_SRC_RAG     = _THIS_FILE.parent                 # .../src/rag/
_SRC         = _SRC_RAG.parent                   # .../src/
PROJECT_ROOT = _SRC.parent                       # <project_root>

DATA_DIR  = PROJECT_ROOT / "data" / "catalog"
INDEX_DIR = PROJECT_ROOT / "outputs" / "faiss_index"

CHUNK_SIZE    = 600   # tokens (approx, 1 token ≈ 4 chars)
CHUNK_OVERLAP = 100   # token overlap between adjacent chunks
RETRIEVER_K   = 6
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Document loading ──────────────────────────────────────────────────────────

def load_catalog_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """Load all .txt catalog files and attach source metadata."""

    print(f"  Looking for catalog files in: {data_dir}")

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Catalog directory not found: {data_dir}\n"
            f"Make sure the folder  data/catalog/  exists inside your project root:\n"
            f"  {PROJECT_ROOT}"
        )

    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}\n"
            f"Expected files: cs_courses.txt, math_courses.txt, "
            f"bscs_requirements.txt, academic_policies.txt, programs_and_minors.txt"
        )

    documents = []
    for filepath in txt_files:
        text = filepath.read_text(encoding="utf-8")

        # Extract SOURCE: line from the top of the file
        source_url = "unknown"
        for line in text.split("\n")[:5]:
            if line.startswith("SOURCE:"):
                source_url = line.replace("SOURCE:", "").strip()
                break

        doc = Document(
            page_content=text,
            metadata={
                "source":   source_url,
                "filename": filepath.name,
                "filepath": str(filepath),
            },
        )
        documents.append(doc)
        print(f"    ✓ {filepath.name}  ({len(text):,} chars)")

    print(f"\n  Total documents loaded: {len(documents)}")
    return documents


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks.

    Strategy:
    - Primary separator = double newline (preserves whole course entries)
    - chunk_size tokens × 4 chars/token ≈ character budget
    - 100-token overlap prevents citations from being cut at chunk edges
    """
    char_size    = chunk_size    * 4   # 600 tokens → 2 400 chars
    char_overlap = chunk_overlap * 4   # 100 tokens →   400 chars

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i:04d}"
        first_line = next(
            (l.strip() for l in chunk.page_content.split("\n") if l.strip()), "N/A"
        )
        chunk.metadata["section_hint"] = first_line[:80]

    print(f"  Total chunks created: {len(chunks)}")
    return chunks


# ── Vector store ──────────────────────────────────────────────────────────────

def _make_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    chunks: List[Document],
    index_dir: Path = INDEX_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> FAISS:
    """Build and persist FAISS index from document chunks."""

    # ── Guard: crash early with a clear message ───────────────────────────────
    if not chunks:
        raise RuntimeError(
            "No document chunks to index — the catalog directory is empty or unreachable.\n"
            f"Expected catalog .txt files in:  {DATA_DIR}\n"
            "Ensure the following files exist:\n"
            "  data/catalog/cs_courses.txt\n"
            "  data/catalog/math_courses.txt\n"
            "  data/catalog/bscs_requirements.txt\n"
            "  data/catalog/academic_policies.txt\n"
            "  data/catalog/programs_and_minors.txt"
        )

    print(f"\n  Loading embedding model: {embedding_model}")
    embeddings = _make_embeddings(embedding_model)

    print(f"  Building FAISS index over {len(chunks)} chunks …")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"  FAISS index saved → {index_dir}")

    return vectorstore


def load_vector_store(
    index_dir: Path = INDEX_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> Optional[FAISS]:
    """Load a previously built FAISS index from disk, or return None."""
    if not index_dir.exists():
        return None
    try:
        embeddings = _make_embeddings(embedding_model)
        vs = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
        print(f"  Loaded existing FAISS index from: {index_dir}")
        return vs
    except Exception as e:
        print(f"  Could not load existing index ({e}), will rebuild.")
        return None


# ── High-level build / load ───────────────────────────────────────────────────

def build_index(force_rebuild: bool = False) -> FAISS:
    """
    Build the FAISS index (or load from disk if already built).

    Args:
        force_rebuild: Re-build even if a saved index exists.
    """
    if not force_rebuild:
        existing = load_vector_store()
        if existing is not None:
            return existing

    print("=" * 60)
    print("Building RAG Index")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Catalog dir  : {DATA_DIR}")
    print(f"  Index dir    : {INDEX_DIR}")
    print("=" * 60)

    print("\n[1/3] Loading catalog documents …")
    docs = load_catalog_documents()

    print("\n[2/3] Chunking documents …")
    chunks = chunk_documents(docs)

    print("\n[3/3] Building vector store …")
    vs = build_vector_store(chunks)

    print("\n✓ Index built successfully!")
    return vs


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def retrieve_context(
    vectorstore: FAISS, query: str, k: int = RETRIEVER_K
) -> List[Document]:
    """Return the top-k most relevant chunks for a query."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    # support both old and new LangChain API
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


def format_retrieved_chunks_for_prompt(docs: List[Document]) -> str:
    """Format retrieved chunks as a labelled context block for the LLM prompt."""
    lines = []
    for i, doc in enumerate(docs):
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
        source   = doc.metadata.get("source",   "unknown")
        filename = doc.metadata.get("filename", "unknown")
        lines.append(
            f"--- CONTEXT CHUNK [{chunk_id}] | Source: {source} | File: {filename} ---\n"
            f"{doc.page_content.strip()}\n"
        )
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build / test RAG index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild")
    args = parser.parse_args()

    vs = build_index(force_rebuild=args.rebuild)
    print("\nTesting retrieval …")
    test_query = "What are the prerequisites for CS 401 Machine Learning?"
    results = retrieve_context(vs, test_query)
    print(f"Query: {test_query}")
    print(f"Retrieved {len(results)} chunks:")
    for r in results:
        snippet = r.page_content[:100].replace("\n", " ")
        print(f"  [{r.metadata.get('chunk_id')}] {r.metadata.get('filename')} | {snippet}…")