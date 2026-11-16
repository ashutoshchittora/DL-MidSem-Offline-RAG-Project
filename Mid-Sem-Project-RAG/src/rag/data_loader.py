"""
Local dataset loading utilities for the IIT Bombay Offline RAG Project.

This module loads previously saved Hugging Face datasets
without downloading data from the Hugging Face Hub.
"""

from pathlib import Path

from datasets import Dataset, load_from_disk


def load_local_dataset(
    project_root: str | Path,
    dataset_name: str
) -> Dataset:
    """
    Load a locally saved Hugging Face dataset.

    Parameters
    ----------
    project_root : str or Path
        Root directory of the RAG project.

    dataset_name : str
        Name of the saved dataset directory inside data/.

    Returns
    -------
    Dataset
        Locally loaded Hugging Face Dataset.

    Raises
    ------
    FileNotFoundError
        If the specified dataset directory does not exist.

    ValueError
        If the loaded object is not a single Dataset.
    """

    # Convert the project root into a Path object
    project_root = Path(project_root)

    # Construct the local dataset directory
    dataset_path = project_root / "data" / dataset_name

    # Verify that the directory exists
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Local dataset directory not found: {dataset_path}"
        )

    # Load the dataset from local storage
    dataset = load_from_disk(str(dataset_path))

    # Ensure the result is a single Dataset
    if not isinstance(dataset, Dataset):
        raise ValueError(
            "Expected a single Hugging Face Dataset."
        )

    return dataset