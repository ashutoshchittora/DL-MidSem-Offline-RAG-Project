# ============================================================
# Stage C11.3.4 — Embedding Token Coverage Inspection
#
# Project: Mid-Sem-Project-RAG-Engineering
# File: scripts/embedding_token_inspection.py
#
# Purpose:
# Measure whether Wikipedia passages fit within
# the selected embedding model's input limit.
#
# READ-ONLY for original RAG data.
# ============================================================

from pathlib import Path

import numpy as np

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# 1. Resolve existing dataset
# ------------------------------------------------------------

ENGINEERING_ROOT = Path(__file__).resolve().parent.parent

RAG_ROOT = (
    ENGINEERING_ROOT.parent
    / "Mid-Sem-Project-RAG"
)

PASSAGES_PATH = (
    RAG_ROOT
    / "data"
    / "wiki_chunks_100w_20overlap"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------
# 2. Load existing Wikipedia passages
# ------------------------------------------------------------

print("\n=== EMBEDDING TOKEN COVERAGE INSPECTION ===")

passages = load_from_disk(
    str(PASSAGES_PATH)
)

texts = passages["text"]

print("Indexed passages:", len(texts))


# ------------------------------------------------------------
# 3. Load embedding tokenizer
# ------------------------------------------------------------

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
    local_files_only=True
)

tokenizer = model.tokenizer

max_length = model.max_seq_length

print("Model:", MODEL_NAME)
print("Maximum sequence length:", max_length)


# ------------------------------------------------------------
# 4. Measure passage token lengths
# ------------------------------------------------------------

token_lengths = []

for text in texts:

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False
    )

    token_lengths.append(
        len(encoded["input_ids"])
    )

token_lengths = np.asarray(token_lengths)


# ------------------------------------------------------------
# 5. Calculate statistics
# ------------------------------------------------------------

over_limit = int(
    np.sum(token_lengths > max_length)
)

percentage_over_limit = (
    100 * over_limit / len(token_lengths)
)

print("\n=== TOKEN LENGTH STATISTICS ===")

print("Minimum:", int(token_lengths.min()))
print("Mean:", round(float(token_lengths.mean()), 2))
print("Median:", float(np.median(token_lengths)))
print("Maximum:", int(token_lengths.max()))

print("\n=== TOKEN LIMIT ANALYSIS ===")

print("Model token limit:", max_length)
print("Passages exceeding limit:", over_limit)

print(
    "Percentage exceeding limit:",
    round(percentage_over_limit, 2),
    "%"
)

print("\nToken coverage inspection completed.")