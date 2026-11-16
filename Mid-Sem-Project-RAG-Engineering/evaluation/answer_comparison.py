"""
C11.8.6 — TF-IDF vs Semantic RAG Answer Evaluation

Compares complete question-answering pipelines using:
- Exact Match
- Token F1
- Article-level retrieval hit
- Designated supporting-chunk hit
- Original evidence support
- Prepared evidence support
- Retrieval and generation latency

Reads the original IIT evaluation dataset.

Does not modify original academic source or evaluation files.
"""

import argparse
from pathlib import Path

import pandas as pd

from adapters.rag_adapter import get_rag_pipeline

from adapters.semantic_rag_adapter import (
    get_semantic_rag_pipeline,
)

# The original adapter makes the academic src package accessible.
get_rag_pipeline()

from rag.evaluation import (
    exact_match_score,
    token_f1_score,
    retrieval_hit_at_k,
    evidence_support_at_k,
)


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

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
# 2. Load original evaluation questions
# ------------------------------------------------------------

def load_questions(limit=None):

    if not QUESTION_FILE.is_file():
        raise FileNotFoundError(QUESTION_FILE)

    questions = pd.read_csv(QUESTION_FILE)

    required_columns = {
        "question_id",
        "question",
        "reference_answer",
        "source_article",
        "supporting_chunk_id",
    }

    if not required_columns.issubset(questions.columns):
        raise ValueError(
            "Required evaluation columns are missing."
        )

    if len(questions) != 20:
        raise ValueError(
            "Expected exactly 20 academic questions."
        )

    if limit is not None:

        if limit < 1 or limit > 20:
            raise ValueError(
                "limit must be between 1 and 20."
            )

        questions = questions.head(limit)

    return questions


# ------------------------------------------------------------
# 3. Evaluate complete RAG pipelines
# ------------------------------------------------------------

def evaluate_answers(limit=None):

    print("\n=== END-TO-END RAG ANSWER EVALUATION ===")

    questions = load_questions(limit)

    print("Evaluation questions:", len(questions))

    original_pipeline = get_rag_pipeline()

    semantic_pipeline = get_semantic_rag_pipeline()

    pipelines = {
        "tfidf": original_pipeline,
        "semantic": semantic_pipeline,
    }

    records = []

    # --------------------------------------------------------
    # Warm-up both pipelines
    # --------------------------------------------------------

    print("\n=== PIPELINE WARM-UP ===")

    warmup_question = str(questions.iloc[0]["question"])

    for method, pipeline in pipelines.items():

        pipeline.answer(
            question=warmup_question,
            top_k=1,
        )

        print(method, "warm-up completed")

    # --------------------------------------------------------
    # Evaluate every question
    # --------------------------------------------------------

    print("\n=== EVALUATION EXECUTION ===")

    for _, row in questions.iterrows():

        question_id = str(row["question_id"])
        question = str(row["question"])
        reference = str(row["reference_answer"])

        expected_article = str(row["source_article"])
        expected_chunk = str(row["supporting_chunk_id"])

        print("\n" + "=" * 60)
        print("Question:", question_id, question)

        for k in K_VALUES:

            print(f"\n--- K = {k} ---")

            for method, pipeline in pipelines.items():

                result = pipeline.answer(
                    question=question,
                    top_k=k,
                )

                answer = str(result["answer"])

                retrieved = result["sources"]
                prepared = result["prepared_sources"]

                retrieved_chunk_ids = [
                    passage["chunk_id"]
                    for passage in retrieved
                ]

                gold_chunk_hit = int(
                    expected_chunk in retrieved_chunk_ids
                )

                article_hit = retrieval_hit_at_k(
                    retrieved,
                    expected_article,
                )

                original_support = evidence_support_at_k(
                    retrieved,
                    reference,
                )

                prepared_support = evidence_support_at_k(
                    prepared,
                    reference,
                )

                exact_match = exact_match_score(
                    answer,
                    reference,
                )

                token_f1 = token_f1_score(
                    answer,
                    reference,
                )

                records.append({
                    "question_id": question_id,
                    "question": question,
                    "reference_answer": reference,
                    "retriever": method,
                    "k": k,
                    "answer": answer,
                    "exact_match": exact_match,
                    "token_f1": token_f1,
                    "article_hit": article_hit,
                    "gold_chunk_hit": gold_chunk_hit,
                    "original_evidence_support": original_support,
                    "prepared_evidence_support": prepared_support,
                    "retrieval_time_seconds":
                        result["retrieval_time_seconds"],
                    "generation_time_seconds":
                        result["generation_time_seconds"],
                    "total_time_seconds":
                        result["total_time_seconds"],
                    "input_tokens":
                        result["input_tokens"],
                    "generated_tokens":
                        result["generated_tokens"],
                })

                print(
                    f"{method}: "
                    f"answer={answer!r}, "
                    f"EM={exact_match}, "
                    f"F1={token_f1:.4f}"
                )

    return pd.DataFrame(records)


# ------------------------------------------------------------
# 4. Summarize results
# ------------------------------------------------------------

def summarize_results(records):

    summary = (
        records
        .groupby(["retriever", "k"], as_index=False)
        .agg(
            questions=("question_id", "nunique"),
            exact_match=("exact_match", "mean"),
            token_f1=("token_f1", "mean"),
            article_hit=("article_hit", "mean"),
            gold_chunk_hit=("gold_chunk_hit", "mean"),
            original_evidence_support=(
                "original_evidence_support",
                "mean",
            ),
            prepared_evidence_support=(
                "prepared_evidence_support",
                "mean",
            ),
            mean_retrieval_time=(
                "retrieval_time_seconds",
                "mean",
            ),
            mean_generation_time=(
                "generation_time_seconds",
                "mean",
            ),
        )
    )

    return summary


# ------------------------------------------------------------
# 5. Command-line entry point
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate the first N academic questions.",
    )

    args = parser.parse_args()

    records = evaluate_answers(
        limit=args.limit
    )

    summary = summarize_results(records)

    print("\n=== ANSWER QUALITY SUMMARY ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n=== FINAL VERIFICATION ===")

    question_count = records["question_id"].nunique()

    expected_records = (
        question_count
        * 2
        * len(K_VALUES)
    )

    checks = {
        "Expected record count":
            len(records) == expected_records,

        "Both retrieval methods":
            set(records["retriever"])
            == {"tfidf", "semantic"},

        "All retrieval depths":
            set(records["k"])
            == set(K_VALUES),

        "Exact Match values valid":
            records["exact_match"].isin([0, 1]).all(),

        "Token F1 values valid":
            records["token_f1"].between(0, 1).all(),

        "Answers generated":
            records["answer"].notna().all(),
    }

    for name, passed in checks.items():

        print(
            name,
            "PASSED" if passed else "FAILED"
        )

    print(
        "\nStage C11.8.6:",
        "PASSED" if all(checks.values()) else "FAILED"
    )


if __name__ == "__main__":
    main()