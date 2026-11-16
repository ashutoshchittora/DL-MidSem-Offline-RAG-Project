# ============================================================
# Stage A5.5.4 — RAG Evaluation Accuracy Dashboard
#
# Project: Mid-Sem-Project-RAG-UI
# File: pages/1_Evaluation_Dashboard.py
#
# Purpose:
# Compare evaluation accuracy across retrieval K values
# and evidence fusion methods.
#
# No changes to the original RAG project.
# ============================================================

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# 2. ENABLE IMPORTS FROM UI PROJECT
# ============================================================

UI_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(UI_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_PROJECT_ROOT))


from services.evaluation_service import (
    load_all_evaluation_data,
    validate_evaluation_data,
)


# ============================================================
# 3. LOAD SAVED EVALUATION DATA
# ============================================================

@st.cache_data
def get_evaluation_data():
    """
    Load and validate saved evaluation datasets.

    No model initialization or evaluation rerun.
    """

    datasets = load_all_evaluation_data()

    verification = validate_evaluation_data(datasets)

    return datasets, verification


try:

    datasets, verification = get_evaluation_data()

except (FileNotFoundError, ValueError, AssertionError) as error:

    st.error(
        f"Unable to load evaluation datasets: {error}"
    )

    st.stop()


summary = datasets["summary"].copy()


# ============================================================
# 4. DASHBOARD TITLE
# ============================================================

st.title("📊 RAG Evaluation Dashboard")

st.write(
    "Interactive comparison of retrieval configurations "
    "and evidence fusion methods."
)

st.success(
    "Saved evaluation datasets loaded successfully."
)


# ============================================================
# 5. EXPERIMENT OVERVIEW
# ============================================================

st.subheader("Experiment Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Evaluation Records",
        verification["evaluation_records"],
    )

with col2:

    st.metric(
        "Questions",
        verification["unique_questions"],
    )

with col3:

    st.metric(
        "Fusion Methods",
        verification["fusion_methods"],
    )

with col4:

    st.metric(
        "Configurations",
        verification["summary_combinations"],
    )


# ============================================================
# 6. METHOD FILTER
# ============================================================

st.divider()

st.subheader("Accuracy Comparison")

method_options = [
    "All Methods",
    "concatenation",
    "generate_from_evidence",
]

selected_method = st.selectbox(
    "Select fusion method",
    options=method_options,
)


if selected_method == "All Methods":

    filtered_summary = summary.copy()

else:

    filtered_summary = summary[
        summary["method"] == selected_method
    ].copy()


# ============================================================
# 7. EXACT MATCH ACCURACY
# ============================================================

st.markdown("### Exact Match Accuracy")

st.caption(
    "Exact Match measures whether the generated answer "
    "matches the reference answer after normalization."
)

exact_match_data = filtered_summary.pivot(
    index="k",
    columns="method",
    values="exact_match_pct",
)

exact_match_data = exact_match_data.sort_index()

st.bar_chart(
    exact_match_data,
    x_label="Number of retrieved passages (K)",
    y_label="Exact Match (%)",
)


# ============================================================
# 8. TOKEN F1 ACCURACY
# ============================================================

st.markdown("### Token F1 Score")

st.caption(
    "Token F1 measures overlap between generated and "
    "reference answer tokens."
)

token_f1_data = filtered_summary.pivot(
    index="k",
    columns="method",
    values="token_f1_pct",
)

token_f1_data = token_f1_data.sort_index()

st.line_chart(
    token_f1_data,
    x_label="Number of retrieved passages (K)",
    y_label="Token F1 (%)",
)

# ============================================================
# 8.1 COMPUTATIONAL COST COMPARISON
# ============================================================

st.divider()

st.subheader("Computational Cost Comparison")

st.caption(
    "Compare model processing time, token consumption, "
    "and model calls across retrieval configurations."
)

# ------------------------------------------------------------
# Generation time comparison
# ------------------------------------------------------------

st.markdown("### Average Generation Time")

generation_time_data = filtered_summary.pivot(
    index="k",
    columns="method",
    values="mean_generation_time",
).sort_index()

st.bar_chart(
    generation_time_data,
    x_label="Number of retrieved passages (K)",
    y_label="Generation Time (seconds)",
)

# ------------------------------------------------------------
# Input token comparison
# ------------------------------------------------------------

st.markdown("### Average Input Tokens")

input_token_data = filtered_summary.pivot(
    index="k",
    columns="method",
    values="mean_input_tokens",
).sort_index()

st.line_chart(
    input_token_data,
    x_label="Number of retrieved passages (K)",
    y_label="Average Input Tokens",
)

# ------------------------------------------------------------
# Model call comparison
# ------------------------------------------------------------

st.markdown("### Total Model Calls")

model_call_data = filtered_summary.pivot(
    index="k",
    columns="method",
    values="total_model_calls",
).sort_index()

st.bar_chart(
    model_call_data,
    x_label="Number of retrieved passages (K)",
    y_label="Total Model Calls",
)

# ------------------------------------------------------------
# Detailed computational cost table
# ------------------------------------------------------------

st.markdown("### Computational Cost Details")

cost_columns = [
    "method",
    "k",
    "mean_retrieval_time",
    "mean_generation_time",
    "mean_input_tokens",
    "mean_generated_tokens",
    "total_model_calls",
]

st.dataframe(
    filtered_summary[cost_columns],
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 9. BEST CONFIGURATION
# ============================================================

st.divider()

st.subheader("Best Performing Configuration")

best_row = summary.loc[
    summary["exact_match_pct"].idxmax()
]

best_col1, best_col2, best_col3 = st.columns(3)

with best_col1:

    st.metric(
        "Best Fusion Method",
        best_row["method"],
    )

with best_col2:

    st.metric(
        "Best Retrieval K",
        int(best_row["k"]),
    )

with best_col3:

    st.metric(
        "Best Exact Match",
        f"{best_row['exact_match_pct']:.2f}%",
    )

st.info(
    "The best configuration is selected using "
    "Exact Match accuracy across all six configurations."
)


# ============================================================
# 10. DETAILED ACCURACY TABLE
# ============================================================

st.divider()

st.subheader("Detailed Evaluation Results")

display_columns = [
    "method",
    "k",
    "questions",
    "exact_match_pct",
    "token_f1_pct",
    "retrieval_hit_pct",
    "prepared_evidence_support_pct",
    "abstention_rate_pct",
]

st.dataframe(
    filtered_summary[display_columns],
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 10.1 FAILURE ANALYSIS DASHBOARD
# ============================================================

st.divider()

st.subheader("Failure Analysis")

st.caption(
    "Inspect non-exact answers and compare automated "
    "evaluation with manual assessments."
)

# ------------------------------------------------------------
# 1. Select the best configuration
# ------------------------------------------------------------

full_evaluation = datasets["full"].copy()

best_configuration = full_evaluation[
    (full_evaluation["method"] == "concatenation")
    & (full_evaluation["k"] == 1)
].copy()

# ------------------------------------------------------------
# 2. Identify non-exact answers
# ------------------------------------------------------------

non_exact = best_configuration[
    best_configuration["exact_match"] == 0
].copy()

# ------------------------------------------------------------
# 3. Merge manual assessments
# ------------------------------------------------------------

manual_assessment = datasets["manual"].copy()

failure_analysis = non_exact.merge(
    manual_assessment,
    on="question_id",
    how="left",
    validate="one_to_one",
)

# ------------------------------------------------------------
# 4. Failure summary metrics
# ------------------------------------------------------------

total_questions = len(best_configuration)
exact_matches = int(best_configuration["exact_match"].sum())
non_exact_count = len(non_exact)
abstentions = int(best_configuration["abstained"].sum())

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Evaluated Questions", total_questions)

with metric2:
    st.metric("Exact Matches", exact_matches)

with metric3:
    st.metric("Non-Exact Answers", non_exact_count)

with metric4:
    st.metric("Abstentions", abstentions)

# ------------------------------------------------------------
# 5. Manual assessment distribution
# ------------------------------------------------------------

st.markdown("### Manual Assessment Distribution")

assessment_counts = (
    failure_analysis["assessment"]
    .fillna("not_assessed")
    .value_counts()
    .rename_axis("Assessment")
    .to_frame("Questions")
)

st.bar_chart(
    assessment_counts,
    x_label="Manual Assessment",
    y_label="Number of Questions",
)

# ------------------------------------------------------------
# 6. Detailed failure table
# ------------------------------------------------------------

st.markdown("### Non-Exact Answer Details")

failure_columns = [
    "question_id",
    "question",
    "reference_answer",
    "answer",
    "token_f1",
    "original_evidence_support",
    "prepared_evidence_support",
    "abstained",
    "assessment",
    "reason",
]

st.dataframe(
    failure_analysis[failure_columns],
    use_container_width=True,
    hide_index=True,
)

# ------------------------------------------------------------
# 7. Select an individual failed question
# ------------------------------------------------------------

st.markdown("### Inspect an Individual Failure")

selected_question_id = st.selectbox(
    "Select question",
    options=failure_analysis["question_id"].tolist(),
)

selected_failure = failure_analysis[
    failure_analysis["question_id"] == selected_question_id
].iloc[0]

st.write("**Question:**", selected_failure["question"])

st.write(
    "**Reference Answer:**",
    selected_failure["reference_answer"],
)

st.write(
    "**Generated Answer:**",
    selected_failure["answer"],
)

st.write(
    "**Token F1:**",
    round(float(selected_failure["token_f1"]), 4),
)

st.write(
    "**Manual Assessment:**",
    selected_failure["assessment"],
)

st.write(
    "**Assessment Reason:**",
    selected_failure["reason"],
)

# ------------------------------------------------------------
# 8. Evidence support inspection
# ------------------------------------------------------------

st.markdown("### Evidence Availability")

evidence_col1, evidence_col2 = st.columns(2)

with evidence_col1:
    st.metric(
        "Original Evidence Supports Answer",
        "Yes"
        if selected_failure["original_evidence_support"] == 1
        else "No",
    )

with evidence_col2:
    st.metric(
        "Prepared Evidence Supports Answer",
        "Yes"
        if selected_failure["prepared_evidence_support"] == 1
        else "No",
    )

if selected_failure["prepared_evidence_support"] == 0:

    st.warning(
        "The reference answer is not supported by the "
        "prepared evidence. This may indicate retrieval "
        "or evidence-preparation limitations."
    )

else:

    st.info(
        "The prepared evidence supports the reference answer. "
        "The non-exact result may involve answer generation "
        "or differences in wording."
    )

# ============================================================
# 11. ACADEMIC INTERPRETATION
# ============================================================

with st.expander("Understanding the Results"):

    st.markdown(
        """
        ### Why is K = 1 better?

        Increasing K retrieves additional passages.

        Additional evidence may help when relevant,
        but can also introduce distracting information.

        **Example: Diana Richburg**

        - K = 1 retrieves relevant information about
          an American middle-distance runner.
        - K = 3 additionally retrieves information about
          Diana Baciu, a Moldovan chess player.
        - The generator can mistakenly use the occupation
          of Diana Baciu when answering about Richburg.

        This demonstrates that:

        **More retrieved evidence does not always
        improve answer correctness.**

        ### Why compare two fusion methods?

        **Concatenation**

        Prepared passages are combined into one evidence
        context before answer generation.

        **Generate from evidence**

        Evidence passages are processed through additional
        generation steps before producing the final answer.

        The comparison helps identify the approach that
        balances accuracy and computational cost.
        """
    )

st.caption(
    "Offline Wikipedia RAG — IIT Bombay "
    "Deep Learning / Generative AI Project"
)