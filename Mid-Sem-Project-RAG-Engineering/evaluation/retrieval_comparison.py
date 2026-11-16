"""
C11.8.5 — TF-IDF vs Semantic Retrieval Evaluation

Uses the original IIT academic evaluation questions.

Measures:
- ArticleHit@K
- GoldChunkHit@K
- Article reciprocal rank
- Gold chunk reciprocal rank
- Retrieval latency

No original academic files are modified.
"""

import argparse
import statistics
import time
from pathlib import Path

import pandas as pd

from adapters.rag_adapter import get_rag_pipeline

from retrieval.semantic_retriever import SemanticRetriever


ENGINEERING_ROOT = Path(__file__).resolve().parents[1]

RAG_ROOT = (
    ENGINEERING_ROOT.parent
    / "Mid-Sem-Project-RAG"
)

QUESTION_FILE = (
    RAG_ROOT
    / "data"
    / "evaluation"
    / "rag_evaluation_20.csv"
)

K_VALUES = (1, 3, 5)


# ------------------------------------------------------------
# 1. Metric definitions
# ------------------------------------------------------------

def hit_at_k(values, expected, k):
    """Return 1 if the expected value occurs in top-K."""

    return int(expected in values[:k])


def reciprocal_rank(values, expected, k):
    """Reciprocal rank within the first K results."""

    for rank, value in enumerate(values[:k], start=1):

        if value == expected:
            return 1.0 / rank

    return 0.0


# ------------------------------------------------------------
# 2. Load evaluation questions
# ------------------------------------------------------------

def load_questions(limit=None):

    if not QUESTION_FILE.is_file():
        raise FileNotFoundError(QUESTION_FILE)

    questions = pd.read_csv(QUESTION_FILE)

    required = {
        "question_id",
        "question",
        "source_article",
        "supporting_chunk_id",
    }

    if not required.issubset(questions.columns):
        raise ValueError(
            "Evaluation question columns are missing."
        )

    if len(questions) != 20:
        raise ValueError(
            "Expected exactly 20 evaluation questions."
        )

    if limit is not None:

        if limit < 1 or limit > len(questions):
            raise ValueError("Invalid evaluation limit.")

        questions = questions.head(limit)

    return questions


# ------------------------------------------------------------
# 3. Evaluate both retrieval methods
# ------------------------------------------------------------

def evaluate_retrieval(limit=None):

    print("\n=== RETRIEVAL COMPARISON ===")

    questions = load_questions(limit)

    print("Evaluation questions:", len(questions))

    # Load the original RAG pipeline.
    original_pipeline = get_rag_pipeline()

    tfidf_retriever = original_pipeline.retriever

    # Load the Engineering semantic retriever.
    semantic_retriever = SemanticRetriever()

    retrievers = {
        "tfidf": tfidf_retriever,
        "semantic": semantic_retriever,
    }

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("\n=== RETRIEVER WARM-UP ===")

    warmup_question = str(questions.iloc[0]["question"])

    for method, retriever in retrievers.items():

        retriever.retrieve(
            question=warmup_question,
            top_k=5,
        )

        print(method, "warm-up completed")

    # --------------------------------------------------------
    # Collect evaluation records
    # --------------------------------------------------------

    print("\n=== EVALUATING QUESTIONS ===")

    records = []
    latency_records = []

    for _, question_row in questions.iterrows():

        question_id = str(question_row["question_id"])
        question = str(question_row["question"])

        expected_article = str(
            question_row["source_article"]
        )

        expected_chunk = str(
            question_row["supporting_chunk_id"]
        )

        print(f"\nQuestion: {question_id}")

        for method, retriever in retrievers.items():

            start = time.perf_counter()

            results = retriever.retrieve(
                question=question,
                top_k=5,
            )

            elapsed = time.perf_counter() - start

            if len(results) != 5:
                raise ValueError(
                    "Expected five retrieved passages."
                )

            article_titles = [
                result["title"]
                for result in results
            ]

            chunk_ids = [
                result["chunk_id"]
                for result in results
            ]

            latency_records.append({
                "question_id": question_id,
                "retriever": method,
                "retrieval_k": 5,
                "retrieval_time_seconds": elapsed,
            })

            for k in K_VALUES:

                records.append({
                    "question_id": question_id,
                    "retriever": method,
                    "k": k,

                    "article_hit": hit_at_k(
                        article_titles,
                        expected_article,
                        k,
                    ),

                    "gold_chunk_hit": hit_at_k(
                        chunk_ids,
                        expected_chunk,
                        k,
                    ),

                    "article_rr": reciprocal_rank(
                        article_titles,
                        expected_article,
                        k,
                    ),

                    "gold_chunk_rr": reciprocal_rank(
                        chunk_ids,
                        expected_chunk,
                        k,
                    ),
                })

            print(
                method,
                "top-1:",
                article_titles[0],
                "| top-5 retrieval:",
                round(elapsed, 4),
                "seconds",
            )

    return (
        pd.DataFrame(records),
        pd.DataFrame(latency_records),
    )


# ------------------------------------------------------------
# 4. Summarize evaluation results
# ------------------------------------------------------------

def summarize(records, latencies):

    summary = (
        records
        .groupby(["retriever", "k"], as_index=False)
        .agg(
            questions=("question_id", "nunique"),
            article_hit_rate=("article_hit", "mean"),
            gold_chunk_hit_rate=("gold_chunk_hit", "mean"),
            article_mrr=("article_rr", "mean"),
            gold_chunk_mrr=("gold_chunk_rr", "mean"),
        )
    )

    latency_summary = (
        latencies
        .groupby("retriever", as_index=False)
        .agg(
            mean_top5_latency_seconds=(
                "retrieval_time_seconds",
                "mean",
            ),
            median_top5_latency_seconds=(
                "retrieval_time_seconds",
                "median",
            ),
        )
    )

    return summary, latency_summary


# ------------------------------------------------------------
# 5. Command-line execution
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions.",
    )

    args = parser.parse_args()

    records, latencies = evaluate_retrieval(
        limit=args.limit
    )

    summary, latency_summary = summarize(
        records,
        latencies,
    )

    print("\n=== RETRIEVAL QUALITY SUMMARY ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n=== TOP-5 RETRIEVAL LATENCY SUMMARY ===")

    print(
        latency_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    expected_records = len(
        records["question_id"].unique()
    ) * 2 * len(K_VALUES)

    checks = {
        "Expected evaluation records":
            len(records) == expected_records,

        "Both retrieval methods evaluated":
            set(records["retriever"]) == {
                "tfidf",
                "semantic",
            },

        "Expected retrieval depths":
            set(records["k"]) == set(K_VALUES),

        "Hit values valid":
            records[
                ["article_hit", "gold_chunk_hit"]
            ].isin([0, 1]).all().all(),
    }

    print("\n=== EVALUATION VERIFICATION ===")

    for name, passed in checks.items():

        print(
            name,
            "PASSED" if passed else "FAILED"
        )

    print(
        "\nStage C11.8.5:",
        "PASSED" if all(checks.values()) else "FAILED"
    )


if __name__ == "__main__":
    main()