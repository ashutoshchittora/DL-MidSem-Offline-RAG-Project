# ============================================================
# Stage A5.5.3 — RAG Evaluation Data Service
#
# Project: Mid-Sem-Project-RAG-UI
# File: services/evaluation_service.py
#
# Purpose:
# Load saved evaluation results from the main RAG project.
#
# Important:
# - Read-only access
# - No model initialization
# - No modifications to the original RAG project
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT LOCATIONS
# ============================================================

UI_PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAG_PROJECT_ROOT = (
    UI_PROJECT_ROOT.parent
    / "Mid-Sem-Project-RAG"
)

EVALUATION_DIRECTORY = (
    RAG_PROJECT_ROOT
    / "results"
    / "evaluation"
)


# ============================================================
# 2. REQUIRED EVALUATION FILES
# ============================================================

EVALUATION_FILES = {
    "full": "rag_full_evaluation.csv",
    "summary": "rag_evaluation_summary.csv",
    "fusion": "rag_fusion_comparison.csv",
    "manual": "rag_manual_assessment.csv",
}


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "full": {
        "question_id",
        "question",
        "reference_answer",
        "method",
        "k",
        "answer",
        "exact_match",
        "token_f1",
        "retrieval_time_seconds",
        "generation_time_seconds",
        "input_tokens",
        "generated_tokens",
        "model_calls",
    },

    "summary": {
        "method",
        "k",
        "questions",
        "exact_match_pct",
        "token_f1_pct",
        "mean_retrieval_time",
        "mean_generation_time",
        "mean_input_tokens",
        "mean_generated_tokens",
        "total_model_calls",
    },

    "fusion": {
        "method",
        "evaluations",
        "mean_exact_match",
        "mean_token_f1",
        "mean_generation_time",
        "total_input_tokens",
        "total_generated_tokens",
        "total_model_calls",
    },

    "manual": {
        "question_id",
        "assessment",
        "reason",
    },
}


# ============================================================
# 4. LOAD A SINGLE EVALUATION FILE
# ============================================================

def load_evaluation_file(dataset_name):
    """
    Load one saved evaluation dataset.

    Parameters
    ----------
    dataset_name : str
        One of: full, summary, fusion, manual.

    Returns
    -------
    pandas.DataFrame
        Loaded evaluation data.
    """

    if dataset_name not in EVALUATION_FILES:

        raise ValueError(
            f"Unknown evaluation dataset: {dataset_name}"
        )

    filename = EVALUATION_FILES[dataset_name]

    filepath = EVALUATION_DIRECTORY / filename

    if not filepath.is_file():

        raise FileNotFoundError(
            f"Evaluation file not found: {filepath}"
        )

    dataframe = pd.read_csv(filepath)

    # --------------------------------------------------------
    # Verify expected columns
    # --------------------------------------------------------

    expected_columns = REQUIRED_COLUMNS[dataset_name]

    missing_columns = (
        expected_columns - set(dataframe.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns in {filename}: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


# ============================================================
# 5. LOAD ALL EVALUATION DATASETS
# ============================================================

def load_all_evaluation_data():
    """
    Load all four evaluation datasets.

    Returns
    -------
    dict
        Dictionary containing four Pandas DataFrames.
    """

    return {
        dataset_name: load_evaluation_file(dataset_name)
        for dataset_name in EVALUATION_FILES
    }


# ============================================================
# 6. VALIDATE DATASET INTEGRITY
# ============================================================

def validate_evaluation_data(datasets):
    """
    Verify the expected experiment structure.

    Returns
    -------
    dict
        Summary of verification results.
    """

    full = datasets["full"]
    summary = datasets["summary"]
    fusion = datasets["fusion"]
    manual = datasets["manual"]

    # --------------------------------------------------------
    # Expected evaluation structure
    # --------------------------------------------------------

    unique_questions = full["question_id"].nunique()

    unique_methods = full["method"].nunique()

    unique_k_values = full["k"].nunique()

    expected_records = (
        unique_questions
        * unique_methods
        * unique_k_values
    )

    # --------------------------------------------------------
    # Verify every question/method/K combination is unique
    # --------------------------------------------------------

    duplicate_combinations = full.duplicated(
        subset=["question_id", "method", "k"]
    ).sum()

    # --------------------------------------------------------
    # Verify expected values
    # --------------------------------------------------------

    assert len(full) == 120, (
        "Expected 120 evaluation records."
    )

    assert unique_questions == 20, (
        "Expected 20 evaluation questions."
    )

    assert unique_methods == 2, (
        "Expected two fusion methods."
    )

    assert set(full["k"]) == {1, 3, 5}, (
        "Expected K values 1, 3 and 5."
    )

    assert len(full) == expected_records, (
        "Evaluation combinations are incomplete."
    )

    assert duplicate_combinations == 0, (
        "Duplicate evaluation combinations detected."
    )

    assert len(summary) == 6, (
        "Expected six summary combinations."
    )

    assert len(fusion) == 2, (
        "Expected two fusion comparison records."
    )

    assert len(manual) == 5, (
        "Expected five manual assessments."
    )

    return {
        "evaluation_records": len(full),
        "unique_questions": unique_questions,
        "fusion_methods": unique_methods,
        "k_values": sorted(full["k"].unique().tolist()),
        "summary_combinations": len(summary),
        "manual_assessments": len(manual),
        "duplicate_combinations": int(
            duplicate_combinations
        ),
        "status": "PASSED",
    }


# ============================================================
# 7. OPTIONAL DIRECT VERIFICATION
# ============================================================

if __name__ == "__main__":

    print("=== RAG EVALUATION SERVICE VERIFICATION ===")

    datasets = load_all_evaluation_data()

    for dataset_name, dataframe in datasets.items():

        print(
            f"{dataset_name}: "
            f"{len(dataframe)} records loaded"
        )

    verification = validate_evaluation_data(datasets)

    print("\n=== DATASET INTEGRITY ===")

    for key, value in verification.items():

        print(f"{key}: {value}")

    print("\nEvaluation service verification: PASSED")