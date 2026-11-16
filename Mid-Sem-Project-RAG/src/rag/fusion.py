# ============================================================
# Stage 6.4.4 — Reusable Evidence Fusion
# File: src/rag/fusion.py
# ============================================================

from rag.evidence import prepare_evidence


def concatenate_evidence(passages, tokenizer, max_tokens=80):
    """Combine shortened passages into one evidence passage."""

    prepared = prepare_evidence(
        passages=passages,
        tokenizer=tokenizer,
        max_tokens=max_tokens
    )

    combined_text = "\n\n".join(
        passage["text"] for passage in prepared
    )

    return prepared, [{"text": combined_text}]


def generate_from_evidence(
    question,
    passages,
    generator,
    tokenizer,
    max_tokens=80
):
    """
    Generate individual answers, then synthesize a final answer.

    Uses an existing generator instance (does not load another model).
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not passages:
        raise ValueError("At least one passage is required.")

    prepared = prepare_evidence(
        passages=passages,
        tokenizer=tokenizer,
        max_tokens=max_tokens
    )

    candidates = []
    total_input_tokens = 0
    total_generated_tokens = 0
    total_generation_time = 0.0

    for rank, passage in enumerate(prepared, start=1):

        result = generator.generate(
            question=question,
            passages=[passage]
        )

        candidate = {
            "rank": rank,
            "title": passage.get("title"),
            "score": passage.get("score"),
            "answer": result["answer"]
        }

        candidates.append(candidate)

        total_input_tokens += result["input_tokens"]
        total_generated_tokens += result["generated_tokens"]
        total_generation_time += result["generation_time_seconds"]

    candidate_text = "\n".join(
        f"Rank {item['rank']} "
        f"(Source: {item['title']}): {item['answer']}"
        for item in candidates
    )

    synthesis_result = generator.generate(
        question=question,
        passages=[{"text": candidate_text}]
    )

    total_input_tokens += synthesis_result["input_tokens"]
    total_generated_tokens += synthesis_result["generated_tokens"]
    total_generation_time += synthesis_result["generation_time_seconds"]

    return {
        "answer": synthesis_result["answer"],
        "candidates": candidates,
        "model_calls": len(prepared) + 1,
        "total_input_tokens": total_input_tokens,
        "total_generated_tokens": total_generated_tokens,
        "generation_time_seconds": total_generation_time
    }