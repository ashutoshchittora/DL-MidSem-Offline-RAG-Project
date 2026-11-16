# ============================================================
# Stage C11.3.3 — First Semantic Embedding Experiment
#
# Project: Mid-Sem-Project-RAG-Engineering
# File: scripts/embedding_smoke_test.py
#
# Purpose:
# Generate sentence embeddings and compare cosine similarity.
#
# The original RAG implementation is not modified.
# ============================================================

import numpy as np
import torch

from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# 1. Experiment configuration
# ------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUESTION = "Who is Diana Richburg?"

PASSAGES = [
    "Diana Richburg was an American middle-distance runner.",
    "Diana Baciu was a Moldovan chess player.",
    "The athlete competed in track and field events."
]


# ------------------------------------------------------------
# 2. Load embedding model
# ------------------------------------------------------------

print("\n=== SEMANTIC EMBEDDING EXPERIMENT ===")

print("\nLoading embedding model...")

print("Model:", MODEL_NAME)
print("Device:", DEVICE)

model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

print("Embedding model loaded successfully.")


# ------------------------------------------------------------
# 3. Generate question embedding
# ------------------------------------------------------------

question_embedding = model.encode(
    [QUESTION],
    convert_to_numpy=True,
    normalize_embeddings=True
)


# ------------------------------------------------------------
# 4. Generate passage embeddings
# ------------------------------------------------------------

passage_embeddings = model.encode(
    PASSAGES,
    convert_to_numpy=True,
    normalize_embeddings=True
)


# ------------------------------------------------------------
# 5. Inspect embedding dimensions
# ------------------------------------------------------------

print("\n=== EMBEDDING DIMENSIONS ===")

print(
    "Question embedding shape:",
    question_embedding.shape
)

print(
    "Passage embeddings shape:",
    passage_embeddings.shape
)

print(
    "Embedding dimensions:",
    passage_embeddings.shape[1]
)


# ------------------------------------------------------------
# 6. Calculate cosine similarity
# ------------------------------------------------------------

# Because embeddings are L2-normalized,
# their dot product equals cosine similarity.

similarity_scores = (
    question_embedding @ passage_embeddings.T
).ravel()


# ------------------------------------------------------------
# 7. Rank passages
# ------------------------------------------------------------

ranking = np.argsort(-similarity_scores)

print("\n=== SEMANTIC RETRIEVAL RANKING ===")

print("Question:", QUESTION)

for rank, index in enumerate(ranking, start=1):

    print(f"\nRank {rank}")

    print("Passage:", PASSAGES[index])

    print(
        "Cosine similarity:",
        round(float(similarity_scores[index]), 4)
    )


# ------------------------------------------------------------
# 8. Inspect vector normalization
# ------------------------------------------------------------

print("\n=== VECTOR NORMALIZATION ===")

question_norm = np.linalg.norm(
    question_embedding[0]
)

passage_norm = np.linalg.norm(
    passage_embeddings[0]
)

print("Question vector norm:", round(float(question_norm), 6))

print("First passage vector norm:", round(float(passage_norm), 6))


# ------------------------------------------------------------
# 9. Final verification
# ------------------------------------------------------------

print("\n=== EXPERIMENT VERIFICATION ===")

checks = {
    "Question embedding created":
        question_embedding.shape == (1, 384),

    "Three passage embeddings created":
        passage_embeddings.shape == (3, 384),

    "All similarity scores finite":
        bool(np.isfinite(similarity_scores).all()),

    "Question vector normalized":
        bool(np.isclose(question_norm, 1.0, atol=1e-4)),

    "Passage vector normalized":
        bool(np.isclose(passage_norm, 1.0, atol=1e-4))
}

for name, passed in checks.items():

    print(
        name,
        "PASSED" if passed else "FAILED"
    )

if all(checks.values()):

    print("\nStage C11.3.3: PASSED")

else:

    print("\nStage C11.3.3: FAILED")