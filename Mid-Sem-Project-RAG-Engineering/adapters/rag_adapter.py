# ============================================================
# Stage C10.4.3 — Engineering RAG Adapter
#
# Project: Mid-Sem-Project-RAG-Engineering
# File: adapters/rag_adapter.py
#
# Purpose:
# Reuse the original academic RAG pipeline without
# modifying its source files.
# ============================================================

import importlib
import sys

from functools import lru_cache
from pathlib import Path


# ------------------------------------------------------------
# 1. Resolve project locations
# ------------------------------------------------------------

ENGINEERING_ROOT = Path(__file__).resolve().parent.parent

CODE_ROOT = ENGINEERING_ROOT.parent

RAG_PROJECT_ROOT = CODE_ROOT / "Mid-Sem-Project-RAG"

RAG_SRC_DIR = RAG_PROJECT_ROOT / "src"

RAG_CONFIG_PATH = (
    RAG_PROJECT_ROOT
    / "config"
    / "rag_config.json"
)


# ------------------------------------------------------------
# 2. Verify original RAG project
# ------------------------------------------------------------

def inspect_original_rag():
    """
    Verify that the original academic RAG project
    contains the files needed by this adapter.
    """

    required_files = [
        RAG_SRC_DIR / "run_rag.py",
        RAG_SRC_DIR / "rag" / "pipeline.py",
        RAG_SRC_DIR / "rag" / "generator.py",
        RAG_CONFIG_PATH
    ]

    missing_files = [
        path
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:

        raise FileNotFoundError(
            "Missing original RAG files: "
            + ", ".join(str(path) for path in missing_files)
        )

    return {
        "engineering_root": str(ENGINEERING_ROOT),
        "rag_project_root": str(RAG_PROJECT_ROOT),
        "rag_config_path": str(RAG_CONFIG_PATH),
        "required_files_present": True
    }


# ------------------------------------------------------------
# 3. Import original RAG application
# ------------------------------------------------------------

def get_original_rag_module():
    """
    Import the existing run_rag module.

    Importing the module does not invoke its main()
    function.
    """

    inspect_original_rag()

    rag_src = str(RAG_SRC_DIR)

    if rag_src not in sys.path:
        sys.path.insert(0, rag_src)

    module = importlib.import_module("run_rag")

    return module


# ------------------------------------------------------------
# 4. Initialize and cache the RAG pipeline
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def get_rag_pipeline():
    """
    Initialize the original RAG pipeline and reuse it.

    The cache prevents sequential calls from repeatedly
    loading the model and TF-IDF index.

    Concurrent first-time initialization will be
    reviewed during the performance engineering stage.
    """

    rag_module = get_original_rag_module()

    config = rag_module.load_config(
        config_path=RAG_CONFIG_PATH
    )

    pipeline = rag_module.initialize_rag(config)

    return pipeline


# ------------------------------------------------------------
# 5. Execute question answering
# ------------------------------------------------------------

def answer_question(question: str, top_k: int = 1):
    """
    Execute question answering through the existing
    academic RAG pipeline.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k not in (1, 3, 5):
        raise ValueError("top_k must be 1, 3, or 5.")

    pipeline = get_rag_pipeline()

    result = pipeline.answer(
        question=question.strip(),
        top_k=top_k
    )

    return result