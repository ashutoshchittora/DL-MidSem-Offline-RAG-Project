# ============================================================
# Stage A5.4.3 — Streamlit RAG Visualization Application
#
# Project: Mid-Sem-Project-RAG-UI
# File: app.py
#
# Purpose:
# Visualize the existing Offline Wikipedia RAG system.
#
# The core RAG project is reused without modification.
# ============================================================

import json
from pathlib import Path

import streamlit as st

from services.rag_service import load_rag_pipeline


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Offline Wikipedia RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. RESOLVE PROJECT LOCATIONS
# ============================================================

UI_PROJECT_ROOT = Path(__file__).resolve().parent

RAG_PROJECT_ROOT = (
    UI_PROJECT_ROOT.parent
    / "Mid-Sem-Project-RAG"
)

RAG_CONFIG_PATH = (
    RAG_PROJECT_ROOT
    / "config"
    / "rag_config.json"
)


# ============================================================
# 3. LOAD EXISTING RAG CONFIGURATION
# ============================================================

@st.cache_data
def load_rag_config(config_path: str):
    """
    Load configuration from the existing RAG project.
    """

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


if not RAG_CONFIG_PATH.is_file():

    st.error(
        "RAG configuration file was not found. "
        "Verify the main RAG project location."
    )

    st.stop()


config = load_rag_config(str(RAG_CONFIG_PATH))


# ============================================================
# 4. CACHE THE EXISTING RAG PIPELINE
# ============================================================

@st.cache_resource(
    show_spinner="Loading Wikipedia RAG pipeline..."
)
def get_cached_rag_pipeline():
    """
    Initialize the existing RAG pipeline once.

    Streamlit reuses the loaded model and retriever
    across application reruns.
    """

    return load_rag_pipeline()


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ RAG Configuration")

    st.write("**Retriever:** TF-IDF")

    st.write("**Similarity:** Cosine")

    st.write(
        "**Generator:**",
        config["generator"]["model_name"],
    )

    st.write(
        "**Evidence limit:**",
        config["evidence"]["max_tokens_per_passage"],
        "tokens",
    )

    st.divider()

    st.success("RAG configuration loaded")

    # --------------------------------------------------------
    # Initialize engine manually
    # --------------------------------------------------------

    if st.button("Initialize RAG Engine"):

        pipeline = get_cached_rag_pipeline()

        st.success(
            "RAG engine initialized successfully!"
        )

        st.write(
            "Retriever:",
            type(pipeline.retriever).__name__,
        )

        st.write(
            "Generator:",
            type(pipeline.generator).__name__,
        )

        st.write(
            "Device:",
            str(pipeline.generator.device),
        )

    st.divider()

    st.caption(
        "This interface reuses the existing "
        "Offline Wikipedia RAG project."
    )


# ============================================================
# 6. APPLICATION TITLE
# ============================================================

st.title("📚 Offline Wikipedia RAG")

st.write(
    "An interactive visualization of Retrieval-Augmented "
    "Generation using Wikipedia, TF-IDF retrieval, "
    "and Flan-T5-small."
)


# ============================================================
# 7. PROJECT OVERVIEW
# ============================================================

st.subheader("Project Overview")

overview_col1, overview_col2, overview_col3, overview_col4 = (
    st.columns(4)
)

with overview_col1:

    st.metric(
        "Indexed Passages",
        f"{config['data']['num_passages']:,}",
    )

with overview_col2:

    st.metric(
        "TF-IDF Vocabulary",
        f"{config['retrieval']['max_features']:,}",
    )

with overview_col3:

    st.metric(
        "Best Retrieval K",
        config["retrieval"]["top_k"],
    )

with overview_col4:

    st.metric(
        "Maximum Input Tokens",
        config["generator"]["max_input_tokens"],
    )


# ============================================================
# 8. QUESTION ANSWERING INTERFACE
# ============================================================

st.divider()

st.subheader("Ask Wikipedia")

# ------------------------------------------------------------
# Allow other Streamlit pages to suggest a question
# ------------------------------------------------------------

if "rag_question" not in st.session_state:
    st.session_state["rag_question"] = "Who is Diana Richburg?"

question = st.text_input(
    "Enter your question",
    key="rag_question",
    placeholder="Ask a question about Wikipedia...",
)

top_k = st.selectbox(
    "Number of retrieved passages (K)",
    options=[1, 3, 5],
    index=0,
)

generate_clicked = st.button(
    "Generate Answer",
    type="primary",
)


# ============================================================
# 9. EXECUTE THE EXISTING RAG PIPELINE
# ============================================================

if generate_clicked:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        pipeline = get_cached_rag_pipeline()

        with st.spinner(
            "Retrieving Wikipedia evidence "
            "and generating an answer..."
        ):

            result = pipeline.answer(
                question=question.strip(),
                top_k=top_k,
            )

        # Save the most recent result in this browser session.
        st.session_state["rag_result"] = result


# ============================================================
# 10. DISPLAY ANSWER AND EVIDENCE
# ============================================================

if "rag_result" in st.session_state:

    result = st.session_state["rag_result"]

    st.divider()

    # --------------------------------------------------------
    # 10.1 GENERATED ANSWER
    # --------------------------------------------------------

    st.subheader("Generated Answer")

    st.write(
        "**Question:**",
        result["question"],
    )

    st.write(
        "**Retrieval K:**",
        result["top_k"],
    )

    st.success(
        result["answer"]
    )

    # --------------------------------------------------------
    # 10.2 PERFORMANCE METRICS
    # --------------------------------------------------------

    st.subheader("Performance Metrics")

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    with metric_col1:

        st.metric(
            "Retrieval Time",
            f"{result['retrieval_time_seconds']:.4f} s",
        )

    with metric_col2:

        st.metric(
            "Generation Time",
            f"{result['generation_time_seconds']:.4f} s",
        )

    with metric_col3:

        st.metric(
            "Input Tokens",
            result["input_tokens"],
        )

    with metric_col4:

        st.metric(
            "Generated Tokens",
            result["generated_tokens"],
        )

    st.caption(
        "Total pipeline time: "
        f"{result['total_time_seconds']:.4f} seconds"
    )

    # --------------------------------------------------------
    # 10.3 EVIDENCE INSPECTION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Wikipedia Evidence Inspection"
    )

    st.caption(
        "Compare the original retrieved passages with "
        "the shortened evidence provided to Flan-T5."
    )

    st.write(
        "**Evidence preparation method:**",
        "Prefix shortening",
    )

    st.write(
        "**Maximum evidence tokens per passage:**",
        result["max_evidence_tokens"],
    )

    st.write(
        "**Fusion method:**",
        result["fusion_method"],
    )

    # --------------------------------------------------------
    # 10.4 LOAD ORIGINAL AND PREPARED SOURCES
    # --------------------------------------------------------

    original_sources = result["sources"]

    prepared_sources = result["prepared_sources"]

    if len(original_sources) != len(prepared_sources):

        st.error(
            "Original and prepared evidence counts "
            "do not match."
        )

        st.stop()

    # Reuse the tokenizer from the already loaded model.
    tokenizer = (
        get_cached_rag_pipeline()
        .generator
        .tokenizer
    )

    evidence_token_budget = result["max_evidence_tokens"]

    if evidence_token_budget <= 0:

        st.error(
            "The configured evidence token budget "
            "must be greater than zero."
        )

        st.stop()

    # --------------------------------------------------------
    # 10.5 DISPLAY EACH RETRIEVED PASSAGE
    # --------------------------------------------------------

    for rank, (source, prepared_source) in enumerate(
        zip(original_sources, prepared_sources),
        start=1,
    ):

        article_title = source["title"]

        similarity = source["score"]

        with st.expander(
            f"Rank {rank}: {article_title} "
            f"(Similarity: {similarity:.4f})",
            expanded=(rank == 1),
        ):

            # ------------------------------------------------
            # Passage metadata
            # ------------------------------------------------

            st.write(
                "**Article:**",
                article_title,
            )

            st.write(
                "**Chunk ID:**",
                source["chunk_id"],
            )

            st.write(
                "**Similarity Score:**",
                round(similarity, 4),
            )

            st.write(
                "**Wikipedia URL:**",
                source["url"],
            )

            # ------------------------------------------------
            # Side-by-side evidence comparison
            # ------------------------------------------------

            original_col, prepared_col = st.columns(2)

            with original_col:

                st.markdown(
                    "#### Original Retrieved Passage"
                )

                st.info(
                    source["text"]
                )

            with prepared_col:

                st.markdown(
                    "#### Prepared Evidence"
                )

                st.success(
                    prepared_source["text"]
                )

            # ------------------------------------------------
            # Word counts
            # ------------------------------------------------

            original_words = len(
                source["text"].split()
            )

            prepared_words = len(
                prepared_source["text"].split()
            )

            st.caption(
                f"Original words: {original_words} | "
                f"Prepared words: {prepared_words}"
            )

            # ------------------------------------------------
            # Token counts using the actual Flan-T5 tokenizer
            # ------------------------------------------------

            original_tokens = len(
                tokenizer.encode(
                    source["text"],
                    add_special_tokens=False,
                )
            )

            prepared_tokens = len(
                tokenizer.encode(
                    prepared_source["text"],
                    add_special_tokens=False,
                )
            )

            # ------------------------------------------------
            # Display token statistics
            # ------------------------------------------------

            st.markdown(
                "#### Evidence Token Statistics"
            )

            token_col1, token_col2, token_col3 = (
                st.columns(3)
            )

            with token_col1:

                st.metric(
                    "Original Tokens",
                    original_tokens,
                )

            with token_col2:

                st.metric(
                    "Prepared Tokens",
                    prepared_tokens,
                )

            with token_col3:

                st.metric(
                    "Token Budget",
                    evidence_token_budget,
                )

            # ------------------------------------------------
            # Token budget utilization
            # ------------------------------------------------

            token_utilization = (
                prepared_tokens / evidence_token_budget
            )

            st.progress(
                min(token_utilization, 1.0)
            )

            st.caption(
                "Token budget utilization: "
                f"{token_utilization * 100:.1f}%"
            )

            # ------------------------------------------------
            # Verify evidence limit
            # ------------------------------------------------

            if prepared_tokens <= evidence_token_budget:

                st.success(
                    "Prepared evidence is within "
                    "the token budget."
                )

            else:

                st.error(
                    "Prepared evidence exceeds "
                    "the token budget."
                )

    # --------------------------------------------------------
    # 10.6 EXECUTION DEVICE
    # --------------------------------------------------------

    st.caption(
        f"Model execution device: {result['device']}"
    )


# ============================================================
# 11. PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander("About this project"):

    st.write(
        """
        This application visualizes an offline
        Retrieval-Augmented Generation (RAG) system.

        **Architecture**

        1. Wikipedia articles are divided into passages.
        2. Passages are indexed using TF-IDF.
        3. Cosine similarity retrieves relevant passages.
        4. Evidence is shortened within a token budget.
        5. Prepared evidence is supplied to Flan-T5-small.
        6. Flan-T5-small generates an answer.
        7. Answers are evaluated against reference answers.

        **Academic Evaluation**

        The project evaluates two evidence-fusion approaches
        across K = 1, 3, and 5.

        The best measured configuration is:

        - Retrieval K: 1
        - Fusion: Concatenation
        - Exact Match: 75%
        - Token F1: 85.62%
        """
    )

    st.caption(
        "IIT Bombay — Deep Learning / Generative AI Project"
    )