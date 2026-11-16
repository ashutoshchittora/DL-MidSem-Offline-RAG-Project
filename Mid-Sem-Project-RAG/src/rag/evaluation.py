# ============================================================
# Stage 6.5.4 — Reusable RAG Evaluation Metrics
# File: src/rag/evaluation.py
# ============================================================

import re
import string

from collections import Counter


# ------------------------------------------------------------
# 1. Answer normalization
# ------------------------------------------------------------

def normalize_answer(text):
    """
    Normalize text before comparing answers.

    - Lowercase
    - Replace punctuation with spaces
    - Remove English articles
    - Normalize whitespace
    """

    text = str(text).lower()

    text = "".join(
        " " if char in string.punctuation else char
        for char in text
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text
    )

    return " ".join(text.split())


# ------------------------------------------------------------
# 2. Exact Match
# ------------------------------------------------------------

def exact_match_score(prediction, reference):
    """
    Return 1 when normalized answers match, else 0.
    """

    return int(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


# ------------------------------------------------------------
# 3. Token-level F1
# ------------------------------------------------------------

def token_f1_score(prediction, reference):
    """
    Calculate word-token overlap F1.
    """

    predicted_tokens = normalize_answer(
        prediction
    ).split()

    reference_tokens = normalize_answer(
        reference
    ).split()

    if not predicted_tokens or not reference_tokens:
        return float(
            predicted_tokens == reference_tokens
        )

    predicted_counts = Counter(predicted_tokens)
    reference_counts = Counter(reference_tokens)

    common = sum(
        (predicted_counts & reference_counts).values()
    )

    if common == 0:
        return 0.0

    precision = common / len(predicted_tokens)
    recall = common / len(reference_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ------------------------------------------------------------
# 4. Retrieval Hit@K
# ------------------------------------------------------------

def retrieval_hit_at_k(retrieved_passages, expected_article):
    """
    Check whether the expected article is retrieved.
    """

    return int(
        any(
            passage["title"] == expected_article
            for passage in retrieved_passages
        )
    )


# ------------------------------------------------------------
# 5. Evidence Support@K
# ------------------------------------------------------------

def evidence_support_at_k(retrieved_passages, reference_answer):
    """
    Check whether normalized reference text appears
    in at least one retrieved passage.

    This is a lexical evidence-support proxy,
    not a semantic entailment metric.
    """

    reference = normalize_answer(reference_answer)

    if not reference:
        return 0

    for passage in retrieved_passages:

        passage_text = normalize_answer(
            passage["text"]
        )

        if f" {reference} " in f" {passage_text} ":
            return 1

    return 0