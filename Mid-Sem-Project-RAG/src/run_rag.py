# ============================================================
# Stage 9.7.3 — Configuration-Driven Standalone RAG Application
#
# File: src/run_rag.py
#
# Enhancement:
# Optional local Flan-T5 model loading with backward compatibility.
#
# Original retrieval, fusion, generation and evaluation behavior
# is preserved.
# ============================================================

import argparse
import json

from pathlib import Path

from datasets import load_from_disk

from rag.indexer import load_tfidf_index
from rag.retriever import TfidfRetriever
from rag.generator import FlanT5Generator
from rag.pipeline import RAGPipeline


# ------------------------------------------------------------
# 1. Resolve project root and configuration path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "rag_config.json"
)


# ------------------------------------------------------------
# 2. Load configuration
# ------------------------------------------------------------

def load_config(config_path=CONFIG_PATH):

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"RAG configuration not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    required_sections = [
        "data",
        "retrieval",
        "evidence",
        "generator"
    ]

    for section in required_sections:

        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    return config


# ------------------------------------------------------------
# 3. Initialize RAG application
# ------------------------------------------------------------

def initialize_rag(config):

    print("\n=== INITIALIZING CONFIGURATION-DRIVEN RAG ===")

    # --------------------------------------------------------
    # Resolve local artifact paths
    # --------------------------------------------------------

    passage_dir = (
        PROJECT_ROOT
        / config["data"]["passage_directory"]
    )

    index_dir = (
        PROJECT_ROOT
        / config["data"]["index_directory"]
    )

    # --------------------------------------------------------
    # Load saved Wikipedia passages
    # --------------------------------------------------------

    print("\nLoading Wikipedia passages...")

    passages = load_from_disk(
        str(passage_dir)
    )

    print("Passages loaded:", len(passages))

    # --------------------------------------------------------
    # Load TF-IDF index
    # --------------------------------------------------------

    print("\nLoading TF-IDF index...")

    vectorizer, tfidf_matrix, metadata = (
        load_tfidf_index(index_dir)
    )

    print("TF-IDF matrix:", tfidf_matrix.shape)

    # --------------------------------------------------------
    # Validate passage/index counts
    # --------------------------------------------------------

    expected_passages = config["data"]["num_passages"]

    if len(passages) != expected_passages:
        raise ValueError(
            "Loaded passage count differs from configuration."
        )

    if len(passages) != tfidf_matrix.shape[0]:
        raise ValueError(
            "Passage count does not match TF-IDF matrix."
        )

    if (
        tfidf_matrix.shape[1]
        != config["retrieval"]["max_features"]
    ):
        raise ValueError(
            "TF-IDF feature count differs from configuration."
        )

    if config["retrieval"]["method"] != "tfidf":
        raise ValueError(
            "Only TF-IDF retrieval is currently supported."
        )

    if config["retrieval"]["similarity"] != "cosine":
        raise ValueError(
            "Only cosine similarity is currently supported."
        )

    # --------------------------------------------------------
    # Initialize retriever
    # --------------------------------------------------------

    print("\nInitializing retriever...")

    retriever = TfidfRetriever(
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        passages=passages
    )

    # --------------------------------------------------------
    # Initialize generator once
    # --------------------------------------------------------

    print("\nLoading generator...")

    generator_config = config["generator"]

    configured_device = generator_config["device"]

    # None lets FlanT5Generator select CUDA or CPU
    device = (
        None
        if configured_device == "auto"
        else configured_device
    )

    # ========================================================
    # NEW: Resolve optional local model path
    # ========================================================

    local_model_config = generator_config.get(
        "local_model_path"
    )

    local_model_path = (
        str(PROJECT_ROOT / local_model_config)
        if local_model_config
        else None
    )

    # --------------------------------------------------------
    # Initialize generator
    #
    # When a valid local model exists, generator.py loads it.
    #
    # When the local directory is missing, generator.py uses
    # the original Hugging Face model-loading behavior.
    #
    # Existing constructor defaults remain compatible.
    # --------------------------------------------------------

    generator = FlanT5Generator(
        model_name=generator_config["model_name"],
        max_input_tokens=generator_config["max_input_tokens"],
        max_new_tokens=generator_config["max_new_tokens"],
        device=device,
        local_model_path=local_model_path
    )

    # ========================================================
    # NEW: Display actual model source
    # ========================================================

    print("\n=== MODEL SOURCE ===")

    print(
        "Model loading mode:",
        "Local model"
        if generator.using_local_model
        else "Hugging Face model"
    )

    print(
        "Model source:",
        generator.model_source
    )

    # --------------------------------------------------------
    # Initialize final pipeline
    # --------------------------------------------------------

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        max_evidence_tokens=config["evidence"][
            "max_tokens_per_passage"
        ],
        fusion_method=config["evidence"]["fusion_method"]
    )

    print("\nRAG application initialized successfully.")

    return pipeline


# ------------------------------------------------------------
# 4. Execute question answering
# ------------------------------------------------------------

def ask_question(pipeline, question, top_k):

    result = pipeline.answer(
        question=question,
        top_k=top_k
    )

    print("\n=== GENERATED ANSWER ===")

    print("Question:", result["question"])
    print("Answer:", result["answer"])

    print("\n=== RETRIEVED SOURCES ===")

    for source in result["sources"]:

        print("Article:", source["title"])
        print("Chunk ID:", source["chunk_id"])
        print("Similarity:", round(source["score"], 4))
        print("URL:", source["url"])

    print("\n=== PERFORMANCE METRICS ===")

    print(
        "Retrieval time:",
        round(result["retrieval_time_seconds"], 4),
        "seconds"
    )

    print(
        "Generation time:",
        round(result["generation_time_seconds"], 4),
        "seconds"
    )

    print(
        "Total pipeline time:",
        round(result["total_time_seconds"], 4),
        "seconds"
    )

    print("Input tokens:", result["input_tokens"])
    print("Generated tokens:", result["generated_tokens"])
    print("Device:", result["device"])

    return result


# ------------------------------------------------------------
# 5. Main application
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Configuration-Driven Offline Wikipedia RAG"
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask the RAG application"
    )

    args = parser.parse_args()

    if not args.question.strip():
        parser.error("--question cannot be empty")

    # --------------------------------------------------------
    # Load saved configuration
    # --------------------------------------------------------

    config = load_config()

    print("=== RAG CONFIGURATION ===")

    print(
        "Retrieval K:",
        config["retrieval"]["top_k"]
    )

    print(
        "Fusion:",
        config["evidence"]["fusion_method"]
    )

    print(
        "Evidence token limit:",
        config["evidence"]["max_tokens_per_passage"]
    )

    print(
        "Generator:",
        config["generator"]["model_name"]
    )

    # --------------------------------------------------------
    # Initialize application
    # --------------------------------------------------------

    pipeline = initialize_rag(config)

    # --------------------------------------------------------
    # Execute configured retrieval
    # --------------------------------------------------------

    ask_question(
        pipeline=pipeline,
        question=args.question,
        top_k=config["retrieval"]["top_k"]
    )


if __name__ == "__main__":
    main()