# ============================================================
# Stage 6.4.1 — Reusable Evidence Preparation Module
# File: src/rag/evidence.py
# ============================================================

def shorten_passage(text, tokenizer, max_tokens=80):
    """
    Shorten a passage to a maximum number of tokenizer tokens.
    """

    if not isinstance(text, str):
        raise TypeError("Passage text must be a string.")

    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer.")

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False
    )["input_ids"]

    shortened_ids = token_ids[:max_tokens]

    shortened_text = tokenizer.decode(
        shortened_ids,
        skip_special_tokens=True
    )

    return shortened_text


def prepare_evidence(passages, tokenizer, max_tokens=80):
    """
    Prepare retrieved passages for the generator.

    Shortens passage text while preserving retrieval metadata.
    """

    if passages is None:
        raise ValueError("Passages cannot be None.")

    prepared_passages = []

    for passage in passages:

        prepared_passage = passage.copy()

        prepared_passage["text"] = shorten_passage(
            text=passage["text"],
            tokenizer=tokenizer,
            max_tokens=max_tokens
        )

        prepared_passages.append(prepared_passage)

    return prepared_passages