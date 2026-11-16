# ============================================================
# Step A5.2.1 — RAG Service Bridge
#
# File: services/rag_service.py
#
# Purpose:
# Connect Streamlit UI to the existing RAG application.
# ============================================================

import sys
from pathlib import Path


# ------------------------------------------------------------
# 1. Resolve project paths
# ------------------------------------------------------------

UI_PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAG_PROJECT_ROOT = (
    UI_PROJECT_ROOT.parent
    / "Mid-Sem-Project-RAG"
)

RAG_SRC_DIR = RAG_PROJECT_ROOT / "src"


# ------------------------------------------------------------
# 2. Verify main RAG application
# ------------------------------------------------------------

def inspect_rag_project():
    """
    Verify that the existing RAG project is accessible.
    """

    required_files = [
        RAG_SRC_DIR / "run_rag.py",
        RAG_SRC_DIR / "rag" / "pipeline.py",
        RAG_SRC_DIR / "rag" / "generator.py",
        RAG_PROJECT_ROOT / "config" / "rag_config.json",
    ]

    missing_files = [
        path
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required RAG files are missing: "
            + ", ".join(str(path) for path in missing_files)
        )

    return {
        "ui_project_root": str(UI_PROJECT_ROOT),
        "rag_project_root": str(RAG_PROJECT_ROOT),
        "rag_src_directory": str(RAG_SRC_DIR),
        "required_files_present": True
    }


# ------------------------------------------------------------
# 3. Import existing RAG application
# ------------------------------------------------------------

def get_rag_application_module():
    """
    Import the existing RAG application without executing main().
    """

    inspect_rag_project()

    # Allow Python to locate modules in the main project.
    if str(RAG_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_SRC_DIR))

    # Importing run_rag does not call main(),
    # because main() is protected by:
    #
    # if __name__ == "__main__":
    #     main()

    import run_rag

    return run_rag

# ------------------------------------------------------------
# 4. Initialize existing RAG pipeline
# ------------------------------------------------------------

def load_rag_pipeline():
    """
    Load the existing configuration-driven RAG pipeline.

    Reuses the implementation from the main RAG project.
    """

    # Import the existing RAG application
    rag_application = get_rag_application_module()

    # Load the main project's configuration
    config = rag_application.load_config()

    # Initialize retriever, generator and pipeline
    pipeline = rag_application.initialize_rag(config)

    return pipeline