# ============================================================
# Stage A5.7.2 — Knowledge Base Explorer
#
# Project: Mid-Sem-Project-RAG-UI
# File: pages/2_Knowledge_Base_Explorer.py
#
# Purpose:
# Help users discover which Wikipedia articles are indexed.
#
# This page reads the existing dataset.
# It does not modify the main RAG project.
# ============================================================

from pathlib import Path

import pandas as pd
import streamlit as st

from datasets import load_from_disk


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Knowledge Base Explorer",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# 2. RESOLVE PROJECT DIRECTORIES
# ============================================================

UI_PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAG_PROJECT_ROOT = (
    UI_PROJECT_ROOT.parent
    / "Mid-Sem-Project-RAG"
)

PASSAGE_DIRECTORY = (
    RAG_PROJECT_ROOT
    / "data"
    / "wiki_chunks_100w_20overlap"
)


# ============================================================
# 3. LOAD ARTICLE METADATA
# ============================================================

@st.cache_data(
    show_spinner="Loading indexed Wikipedia topics..."
)
def load_article_catalogue(dataset_path):
    """
    Load indexed article metadata.

    Returns
    -------
    pandas.DataFrame
        One row per Wikipedia article.

    pandas.DataFrame
        Individual passage records.
    """

    passages = load_from_disk(dataset_path)

    passage_df = passages.to_pandas()

    article_catalogue = (
        passage_df
        .groupby(
            ["article_id", "title", "url"],
            as_index=False
        )
        .agg(
            passage_count=("chunk_id", "count")
        )
        .sort_values("title")
        .reset_index(drop=True)
    )

    return article_catalogue, passage_df


# ============================================================
# 4. INITIALIZE CATALOGUE
# ============================================================

if not PASSAGE_DIRECTORY.is_dir():

    st.error(
        "Wikipedia passage dataset not found. "
        "Verify the main RAG project location."
    )

    st.stop()


try:

    articles_df, passages_df = load_article_catalogue(
        str(PASSAGE_DIRECTORY)
    )

except Exception as error:

    st.error(
        f"Unable to load Wikipedia passages: {error}"
    )

    st.stop()


# ============================================================
# 5. PAGE TITLE
# ============================================================

st.title("🔎 Knowledge Base Explorer")

st.write(
    "Discover the Wikipedia articles available in "
    "this Offline RAG system."
)

st.info(
    "This application searches a selected collection "
    "of Wikipedia articles, not all of Wikipedia. "
    "An indexed article does not guarantee that every "
    "question about it can be answered correctly."
)


# ============================================================
# 6. KNOWLEDGE BASE METRICS
# ============================================================

st.subheader("Knowledge Base Overview")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Indexed Wikipedia Articles",
        f"{len(articles_df):,}"
    )

with col2:

    st.metric(
        "Indexed Passages",
        f"{len(passages_df):,}"
    )


# ============================================================
# 7. SEARCH AVAILABLE ARTICLES
# ============================================================

st.divider()

st.subheader("Search Available Topics")

search_text = st.text_input(
    "Search Wikipedia article titles",
    placeholder="Example: Diana Richburg, football, chess..."
)

# Case-insensitive title search
if search_text.strip():

    filtered_articles = articles_df[
        articles_df["title"].str.contains(
            search_text.strip(),
            case=False,
            regex=False,
            na=False
        )
    ].copy()

else:

    filtered_articles = articles_df.copy()


st.write(
    f"**Matching articles:** {len(filtered_articles):,}"
)


# ============================================================
# 8. DISPLAY MATCHING ARTICLES
# ============================================================

if filtered_articles.empty:

    st.warning(
        "No indexed article titles match your search. "
        "Try a different keyword."
    )

else:

    display_df = filtered_articles[
        ["title", "passage_count", "url"]
    ].rename(
        columns={
            "title": "Article",
            "passage_count": "Indexed Passages",
            "url": "Wikipedia URL"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 9. ARTICLE INSPECTION
# ============================================================

st.divider()

st.subheader("Inspect an Indexed Article")

if filtered_articles.empty:

    st.info(
        "Search for another article to inspect its passages."
    )

else:

    selected_article_id = st.selectbox(
        "Select an article",
        options=filtered_articles["article_id"].tolist(),
        format_func=lambda article_id: (
            filtered_articles.loc[
                filtered_articles["article_id"] == article_id,
                "title"
            ].iloc[0]
        )
    )

    selected_article = filtered_articles[
        filtered_articles["article_id"] == selected_article_id
    ].iloc[0]

    st.write(
        "**Article:**",
        selected_article["title"]
    )

    st.write(
        "**Indexed passages:**",
        int(selected_article["passage_count"])
    )

    st.write(
        "**Wikipedia URL:**",
        selected_article["url"]
    )

    # --------------------------------------------------------
    # Retrieve passages belonging to the selected article
    # --------------------------------------------------------

    article_passages = passages_df[
        passages_df["article_id"] == selected_article_id
    ].sort_values(
        "chunk_number"
    )

    # --------------------------------------------------------
    # Show a preview of the first passage
    # --------------------------------------------------------

    if not article_passages.empty:

        first_passage = article_passages.iloc[0]

        st.markdown("### First Indexed Passage")

        st.write(
            "**Chunk ID:**",
            first_passage["chunk_id"]
        )

        st.info(
            first_passage["text"]
        )

        st.caption(
            "This is a preview of stored Wikipedia evidence. "
            "No answer generation is performed on this page."
        )


# ============================================================
# 9.1 SUGGESTED QUESTIONS
# ============================================================

st.divider()

st.subheader("Try a Suggested Question")

st.write(
    "Select a question from the project's evaluation "
    "dataset and send it to the Ask Wikipedia page."
)

# ------------------------------------------------------------
# Resolve the evaluation question file
# ------------------------------------------------------------

QUESTION_FILE = (
    RAG_PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_evaluation_20.csv"
)


# ------------------------------------------------------------
# Load the saved questions
# ------------------------------------------------------------

@st.cache_data
def load_suggested_questions(filepath):
    """
    Load the existing academic evaluation questions.
    """

    questions_df = pd.read_csv(filepath)

    required_columns = {
        "question_id",
        "question",
        "source_article",
    }

    missing_columns = (
        required_columns - set(questions_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing question columns: {sorted(missing_columns)}"
        )

    return questions_df


if not QUESTION_FILE.is_file():

    st.warning(
        "Suggested questions are currently unavailable."
    )

else:

    try:

        questions_df = load_suggested_questions(
            str(QUESTION_FILE)
        )

    except (ValueError, pd.errors.ParserError) as error:

        st.error(
            f"Unable to load suggested questions: {error}"
        )

    else:

        st.write(
            "**Available example questions:**",
            len(questions_df)
        )

        # ----------------------------------------------------
        # Select a question
        # ----------------------------------------------------

        selected_question = st.selectbox(
            "Choose a question",
            options=questions_df["question"].tolist(),
        )

        # ----------------------------------------------------
        # Show the source article
        # ----------------------------------------------------

        selected_record = questions_df[
            questions_df["question"] == selected_question
        ].iloc[0]

        st.caption(
            "Source article: "
            f"{selected_record['source_article']}"
        )

        # ----------------------------------------------------
        # Send the question to the main application
        # ----------------------------------------------------

        if st.button(
            "Ask This Question",
            type="primary",
        ):

            st.session_state["rag_question"] = (
                selected_question
            )

            st.switch_page("app.py")

# ============================================================
# 10. PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander("What does 'indexed' mean?"):

    st.markdown(
        """
        **Indexed** means that an article's text has been
        divided into passages and included in the project's
        TF-IDF search index.

        When a user asks a question:

        1. TF-IDF converts the question into a numerical vector.
        2. Cosine similarity compares the question with passages.
        3. The most relevant passages are retrieved.
        4. Flan-T5 uses prepared evidence to generate an answer.

        Not every article is guaranteed to be retrieved for
        every question, and not every generated answer will
        necessarily be correct.
        """
    )


st.caption(
    "Offline Wikipedia RAG — IIT Bombay "
    "Deep Learning / Generative AI Project"
)