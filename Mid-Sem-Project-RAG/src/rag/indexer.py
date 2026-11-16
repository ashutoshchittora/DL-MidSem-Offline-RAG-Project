"""
TF-IDF indexing utilities for the IIT Bombay Offline RAG Project.

This module transforms text passages into sparse numerical
representations suitable for information retrieval.
"""

import json
from pathlib import Path

import joblib
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_index(
    passages: list[str],
    max_features: int = 30000
):
    """
    Construct a TF-IDF index from text passages.

    Parameters
    ----------
    passages : list[str]
        Collection of text passages.

    max_features : int
        Maximum vocabulary size.

    Returns
    -------
    tuple
        Fitted TF-IDF vectorizer and sparse document-term matrix.
    """

    if not passages:
        raise ValueError("Passage collection cannot be empty.")

    if max_features <= 0:
        raise ValueError("max_features must be positive.")

    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        norm="l2"
    )

    # Learn vocabulary and calculate TF-IDF weights
    tfidf_matrix = vectorizer.fit_transform(passages)

    return vectorizer, tfidf_matrix

def load_tfidf_index(index_dir: str | Path):
    """
    Load a previously saved TF-IDF index from local storage.

    Parameters
    ----------
    index_dir : str or Path
        Directory containing the saved index artifacts.

    Returns
    -------
    tuple
        Fitted vectorizer, sparse matrix, and metadata dictionary.

    Raises
    ------
    FileNotFoundError
        If an expected artifact is missing.

    ValueError
        If the index dimensions disagree with metadata.
    """

    index_dir = Path(index_dir)

    vectorizer_path = index_dir / "vectorizer.joblib"
    matrix_path = index_dir / "tfidf_matrix.npz"
    metadata_path = index_dir / "index_metadata.json"

    # Verify required files are present
    for path in [
        vectorizer_path,
        matrix_path,
        metadata_path
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing TF-IDF index artifact: {path}"
            )

    # Load only locally generated, trusted Joblib files
    vectorizer = joblib.load(vectorizer_path)

    # Load sparse document-term matrix
    matrix = load_npz(matrix_path)

    # Load index configuration
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Validate the stored dimensions
    if matrix.shape[0] != metadata["num_passages"]:
        raise ValueError(
            "Passage count does not match index metadata."
        )

    if matrix.shape[1] != metadata["vocabulary_size"]:
        raise ValueError(
            "Vocabulary size does not match index metadata."
        )

    if matrix.shape[1] != len(vectorizer.vocabulary_):
        raise ValueError(
            "Vocabulary size does not match fitted vectorizer."
        )

    return vectorizer, matrix, metadata