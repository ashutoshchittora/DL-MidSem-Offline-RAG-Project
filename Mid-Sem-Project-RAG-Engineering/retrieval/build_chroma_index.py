"""
C11.5.5 — Build a persistent MiniLM + ChromaDB index.

Reads the original IIT RAG passages without modifying them.

Usage:
    python -m retrieval.build_chroma_index --limit 10

Later:
    python -m retrieval.build_chroma_index
"""

import argparse
from pathlib import Path

import chromadb
import numpy as np
import torch

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# 1. Project configuration
# ------------------------------------------------------------

ENGINEERING_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_RAG_ROOT = (
    ENGINEERING_ROOT.parent / "Mid-Sem-Project-RAG"
)

PASSAGE_DIRECTORY = (
    ORIGINAL_RAG_ROOT
    / "data"
    / "wiki_chunks_100w_20overlap"
)

CHROMA_DIRECTORY = (
    ENGINEERING_ROOT
    / "data"
    / "chroma_minilm"
)

COLLECTION_NAME = "wikipedia_minilm_384"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EXPECTED_PASSAGES = 8191
EXPECTED_DIMENSIONS = 384


# ------------------------------------------------------------
# 2. Build or resume the semantic index
# ------------------------------------------------------------

def build_index(limit=None, batch_size=32):

    print("\n=== SEMANTIC INDEX BUILDER ===")

    # Validate inputs before loading the model.
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive.")

    # --------------------------------------------------------
    # Load original passages
    # --------------------------------------------------------

    print("\n=== LOADING ORIGINAL WIKIPEDIA PASSAGES ===")

    if not PASSAGE_DIRECTORY.is_dir():
        raise FileNotFoundError(PASSAGE_DIRECTORY)

    passages = load_from_disk(str(PASSAGE_DIRECTORY))

    if len(passages) != EXPECTED_PASSAGES:
        raise ValueError(
            "Original passage count differs from expectation."
        )

    required_fields = {
        "chunk_id",
        "article_id",
        "title",
        "url",
        "chunk_number",
        "text",
        "word_count",
    }

    if not required_fields.issubset(passages.column_names):
        raise ValueError("Required dataset fields are missing.")

    all_ids = list(passages["chunk_id"])

    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate chunk IDs found.")

    print("Original passages:", len(passages))

    selected_count = (
        len(passages)
        if limit is None
        else min(limit, len(passages))
    )

    print("Selected for this run:", selected_count)

    # --------------------------------------------------------
    # Load MiniLM
    # --------------------------------------------------------

    print("\n=== LOADING EMBEDDING MODEL ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    model.max_seq_length = 256

    dimensions = model.get_sentence_embedding_dimension()

    if dimensions != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Unexpected embedding dimensions: {dimensions}"
        )

    print("Model:", MODEL_NAME)
    print("Device:", device)
    print("Embedding dimensions:", dimensions)

    # --------------------------------------------------------
    # Initialize persistent ChromaDB collection
    # --------------------------------------------------------

    print("\n=== INITIALIZING CHROMADB ===")

    CHROMA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIRECTORY)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
        metadata={
            "embedding_model": MODEL_NAME,
            "embedding_dimensions": EXPECTED_DIMENSIONS,
            "source_passages": EXPECTED_PASSAGES,
        },
    )

    if collection.metadata is None:
        raise ValueError("Collection metadata is missing.")

    if (
        collection.metadata.get("embedding_model")
        != MODEL_NAME
    ):
        raise ValueError("Existing collection uses another model.")

    if (
        collection.metadata.get("embedding_dimensions")
        != EXPECTED_DIMENSIONS
    ):
        raise ValueError("Existing collection has another dimension.")

    print("Collection:", collection.name)
    print("Existing records:", collection.count())

    # --------------------------------------------------------
    # Process passages in batches
    # --------------------------------------------------------

    print("\n=== INDEXING PASSAGES ===")

    inserted = 0
    skipped = 0

    for start in range(0, selected_count, batch_size):

        stop = min(start + batch_size, selected_count)

        batch = passages.select(range(start, stop))

        batch_ids = list(batch["chunk_id"])

        # Check which records already exist.
        existing = collection.get(
            ids=batch_ids,
            include=[],
        )

        existing_ids = set(existing["ids"])

        new_records = [
            record
            for record in batch
            if record["chunk_id"] not in existing_ids
        ]

        skipped += len(batch) - len(new_records)

        if not new_records:
            print(
                f"Batch {start}:{stop} — already indexed"
            )
            continue

        texts = [record["text"] for record in new_records]

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError("Empty passage encountered.")

        # Generate normalized embeddings using MiniLM.
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if embeddings.shape != (
            len(new_records),
            EXPECTED_DIMENSIONS,
        ):
            raise ValueError("Embedding shape mismatch.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Non-finite embedding values.")

        metadata = [
            {
                "article_id": str(record["article_id"]),
                "title": str(record["title"]),
                "url": str(record["url"]),
                "chunk_number": int(record["chunk_number"]),
                "word_count": int(record["word_count"]),
            }
            for record in new_records
        ]

        collection.add(
            ids=[
                str(record["chunk_id"])
                for record in new_records
            ],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
        )

        inserted += len(new_records)

        print(
            f"Batch {start}:{stop} — "
            f"inserted={len(new_records)}"
        )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n=== INDEX VERIFICATION ===")

    stored_count = collection.count()

    print("Inserted this run:", inserted)
    print("Already present:", skipped)
    print("Stored records:", stored_count)
    print("Expected total:", EXPECTED_PASSAGES)

    if stored_count > EXPECTED_PASSAGES:
        raise ValueError(
            "Collection contains more than expected."
        )

    if stored_count < selected_count:
        raise ValueError(
            "Collection contains fewer records than selected."
        )

    if limit is None and stored_count != EXPECTED_PASSAGES:
        raise ValueError(
            "Full index count verification failed."
        )

    print("\nSemantic indexing verification: PASSED")

    if limit is not None:
        print("Mode: PARTIAL TEST")
    else:
        print("Mode: FULL CORPUS")

    return stored_count


# ------------------------------------------------------------
# 3. Command-line entry point
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of passages to process.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of passages per embedding batch.",
    )

    args = parser.parse_args()

    build_index(
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()