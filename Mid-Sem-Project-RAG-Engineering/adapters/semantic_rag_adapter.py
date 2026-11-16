"""
Stage C11.7.2 — Semantic RAG Adapter

Creates a separate semantic RAG pipeline while reusing
the original academic Flan-T5 generator.

Original academic files remain unchanged.
"""

from functools import lru_cache

from adapters.rag_adapter import get_rag_pipeline
from retrieval.semantic_retriever import SemanticRetriever


@lru_cache(maxsize=1)
def get_semantic_rag_pipeline():
    """
    Initialize and cache the Engineering semantic pipeline.
    """

    print("\n=== INITIALIZING SEMANTIC RAG PIPELINE ===")

    # --------------------------------------------------------
    # 1. Load original academic pipeline
    # --------------------------------------------------------

    original_pipeline = get_rag_pipeline()

    original_retriever = original_pipeline.retriever

    # --------------------------------------------------------
    # 2. Create new semantic retriever
    # --------------------------------------------------------

    semantic_retriever = SemanticRetriever()

    # --------------------------------------------------------
    # 3. Create separate pipeline instance
    # --------------------------------------------------------

    semantic_pipeline = type(original_pipeline)(
        retriever=semantic_retriever,
        generator=original_pipeline.generator,
        max_evidence_tokens=original_pipeline.max_evidence_tokens,
        fusion_method=original_pipeline.fusion_method,
    )

    # --------------------------------------------------------
    # 4. Verify original pipeline was not modified
    # --------------------------------------------------------

    if original_pipeline.retriever is not original_retriever:
        raise RuntimeError(
            "Original pipeline retriever was modified."
        )

    if semantic_pipeline is original_pipeline:
        raise RuntimeError(
            "Semantic pipeline must be a separate instance."
        )

    if semantic_pipeline.retriever is original_pipeline.retriever:
        raise RuntimeError(
            "Retrievers must be separate instances."
        )

    print("Original retriever:",
          type(original_pipeline.retriever).__name__)

    print("Semantic retriever:",
          type(semantic_pipeline.retriever).__name__)

    print("Generator shared:",
          semantic_pipeline.generator is original_pipeline.generator)

    print("Semantic RAG pipeline initialized successfully.")

    return semantic_pipeline


def answer_semantic_question(question: str, top_k: int = 1):
    """
    Execute end-to-end question answering using
    semantic retrieval and the original Flan-T5 generator.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Question must be a non-empty string."
        )

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k not in (1, 3, 5)
    ):
        raise ValueError(
            "top_k must be 1, 3, or 5."
        )

    pipeline = get_semantic_rag_pipeline()

    return pipeline.answer(
        question=question.strip(),
        top_k=top_k,
    )