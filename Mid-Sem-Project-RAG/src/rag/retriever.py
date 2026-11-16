# ============================================================
# Stage 4.2 — Reusable TF-IDF Passage Retriever
# ============================================================

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """
    Retrieve relevant passages using TF-IDF and cosine similarity.
    """

    def __init__(self, vectorizer, tfidf_matrix, passages):
        """
        Initialize the retriever with an existing TF-IDF index.

        Parameters
        ----------
        vectorizer:
            Previously fitted TF-IDF vectorizer.

        tfidf_matrix:
            TF-IDF matrix containing indexed passage vectors.

        passages:
            Dataset containing passage text and metadata.
        """

        if tfidf_matrix.shape[0] != len(passages):
            raise ValueError(
                "Number of TF-IDF rows must match passage count."
            )

        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.passages = passages

    def retrieve(self, question, top_k=3):
        """
        Return the top-k passages ranked by cosine similarity.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        # Step 1: Convert the user question into a TF-IDF vector.
        # IMPORTANT: transform(), not fit_transform().
        question_vector = self.vectorizer.transform([question])

        # Step 2: Compare the question with every indexed passage.
        similarity_scores = cosine_similarity(
            question_vector,
            self.tfidf_matrix
        ).ravel()

        # Step 3: Select the highest-scoring passage indices.
        top_k = min(top_k, len(self.passages))

        top_indices = np.argsort(
            -similarity_scores,
            kind="stable"
        )[:top_k]

        # Step 4: Assemble retrieved passages and their metadata.
        results = []

        for rank, passage_index in enumerate(top_indices, start=1):

            passage = self.passages[int(passage_index)]

            results.append({
                "rank": rank,
                "score": float(similarity_scores[passage_index]),
                "chunk_id": passage["chunk_id"],
                "article_id": passage["article_id"],
                "title": passage["title"],
                "url": passage["url"],
                "text": passage["text"]
            })

        return results