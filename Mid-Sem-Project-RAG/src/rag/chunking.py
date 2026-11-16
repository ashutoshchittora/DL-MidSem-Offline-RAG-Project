"""
Text chunking utilities for the IIT Bombay Offline RAG Project.

This module converts long documents into overlapping word-based
passages suitable for information retrieval.
"""


def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Parameters
    ----------
    text : str
        Original document text.

    chunk_size : int
        Maximum number of words in each chunk.

    overlap : int
        Number of words shared by consecutive chunks.

    Returns
    -------
    list[str]
        List of text chunks.

    Example
    -------
    >>> chunk_text("one two three four five", chunk_size=3, overlap=1)
    ['one two three', 'three four five']
    """

    # Validate input
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    # Convert text into a sequence of words
    words = text.split()

    if not words:
        return []

    # Number of words to advance between chunks
    stride = chunk_size - overlap

    chunks = []

    # Generate overlapping chunks
    for start in range(0, len(words), stride):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        # Stop after reaching the document end
        if end == len(words):
            break

    return chunks