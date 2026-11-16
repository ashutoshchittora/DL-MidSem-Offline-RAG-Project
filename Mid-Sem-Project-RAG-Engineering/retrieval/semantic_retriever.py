"""
Stage C11.6.4 — Semantic Retriever

Retrieves Wikipedia passages using MiniLM embeddings
and the existing persistent ChromaDB index.

Designed to match the original TfidfRetriever interface.

Original IIT RAG files are not modified.
"""

from pathlib import Path

import numpy as np
import torch
import chromadb

from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# 1. Engineering configuration
# ------------------------------------------------------------

ENGINEERING_ROOT = Path(__file__).resolve().parents[1]

CHROMA_DIRECTORY = (
    ENGINEERING_ROOT
    / "data"
    / "chroma_minilm"
)

COLLECTION_NAME = "wikipedia_minilm_384"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EXPECTED_DIMENSIONS = 384


# ------------------------------------------------------------
# 2. Semantic retriever
# ------------------------------------------------------------

class SemanticRetriever:
    """
    Retrieve Wikipedia passages using dense embeddings.

    Returns records compatible with the original
    TfidfRetriever output structure.
    """

    def __init__(self, device=None):

        print("\n=== INITIALIZING SEMANTIC RETRIEVER ===")

        # ----------------------------------------------------
        # Validate persistent database location
        # ----------------------------------------------------

        if not CHROMA_DIRECTORY.is_dir():
            raise FileNotFoundError(
                f"ChromaDB index not found: {CHROMA_DIRECTORY}"
            )

        # ----------------------------------------------------
        # Open existing ChromaDB collection
        # ----------------------------------------------------

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIRECTORY)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        self.index_size = self.collection.count()

        if self.index_size <= 0:
            raise ValueError("Semantic index is empty.")

        metadata = self.collection.metadata or {}

        if metadata.get("embedding_model") != MODEL_NAME:
            raise ValueError(
                "Semantic index uses an unexpected embedding model."
            )

        if (
            metadata.get("embedding_dimensions")
            != EXPECTED_DIMENSIONS
        ):
            raise ValueError(
                "Semantic index has unexpected dimensions."
            )

        print("Collection:", self.collection.name)
        print("Indexed passages:", self.index_size)

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=self.device,
        )

        self.model.max_seq_length = 256

        dimensions = self.model.get_embedding_dimension()

        if dimensions != EXPECTED_DIMENSIONS:
            raise ValueError(
                f"Unexpected model dimensions: {dimensions}"
            )

        print("Embedding model:", MODEL_NAME)
        print("Device:", self.device)
        print("Dimensions:", dimensions)

        print("Semantic retriever initialized successfully.")

    # --------------------------------------------------------
    # 3. Retrieve relevant passages
    # --------------------------------------------------------

    def retrieve(self, question, top_k=3):
        """
        Return top-k passages ranked by cosine similarity.

        Output fields match the original TfidfRetriever:
        rank, score, chunk_id, article_id, title, url, text.
        """

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question must be a non-empty string."
            )

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        # ----------------------------------------------------
        # Encode the user question
        # ----------------------------------------------------

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if query_embedding.shape != (1, EXPECTED_DIMENSIONS):
            raise ValueError(
                "Unexpected query embedding dimensions."
            )

        if not np.isfinite(query_embedding).all():
            raise ValueError(
                "Query embedding contains non-finite values."
            )

        # ----------------------------------------------------
        # Search ChromaDB
        # ----------------------------------------------------

        effective_k = min(top_k, self.index_size)

        response = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=effective_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        if not (
            len(ids)
            == len(documents)
            == len(metadatas)
            == len(distances)
        ):
            raise ValueError(
                "Inconsistent ChromaDB query result lengths."
            )

        # ----------------------------------------------------
        # Convert results to original retriever format
        # ----------------------------------------------------

        results = []

        for rank, (
            chunk_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):

            # ChromaDB cosine distance = 1 - cosine similarity.
            similarity = 1.0 - float(distance)

            results.append({
                "rank": rank,
                "score": similarity,
                "chunk_id": chunk_id,
                "article_id": metadata["article_id"],
                "title": metadata["title"],
                "url": metadata["url"],
                "text": document,
            })

        return results