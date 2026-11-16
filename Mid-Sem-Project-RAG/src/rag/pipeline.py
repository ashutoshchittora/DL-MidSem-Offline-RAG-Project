# ============================================================
# Stage 7.1.2 — Final RAG Pipeline Integration
# File: src/rag/pipeline.py
# ============================================================

import time

from rag.fusion import concatenate_evidence


class RAGPipeline:
    """
    Coordinate TF-IDF retrieval, evidence preparation,
    concatenation fusion, and Flan-T5 answer generation.

    Existing retriever and generator instances are reused.
    """

    def __init__(
        self,
        retriever,
        generator,
        max_evidence_tokens=80,
        fusion_method="concatenation"
    ):

        if (
            not isinstance(max_evidence_tokens, int)
            or isinstance(max_evidence_tokens, bool)
            or max_evidence_tokens <= 0
        ):
            raise ValueError(
                "max_evidence_tokens must be a positive integer."
            )

        if fusion_method != "concatenation":
            raise ValueError(
                "Final pipeline currently supports "
                "fusion_method='concatenation'."
            )

        self.retriever = retriever
        self.generator = generator

        self.max_evidence_tokens = max_evidence_tokens
        self.fusion_method = fusion_method

    def answer(self, question, top_k=1):
        """
        Retrieve evidence and generate a grounded answer.
        """

        pipeline_start = time.perf_counter()

        # ----------------------------------------------------
        # Step 1: Retrieve passages
        # ----------------------------------------------------

        retrieval_start = time.perf_counter()

        retrieved_passages = self.retriever.retrieve(
            question=question,
            top_k=top_k
        )

        retrieval_time = (
            time.perf_counter() - retrieval_start
        )

        # ----------------------------------------------------
        # Step 2: Prepare and concatenate evidence
        # ----------------------------------------------------

        prepared_passages, combined_evidence = (
            concatenate_evidence(
                passages=retrieved_passages,
                tokenizer=self.generator.tokenizer,
                max_tokens=self.max_evidence_tokens
            )
        )

        # ----------------------------------------------------
        # Step 3: Generate answer
        # ----------------------------------------------------

        generation_result = self.generator.generate(
            question=question,
            passages=combined_evidence
        )

        # ----------------------------------------------------
        # Step 4: Calculate total time
        # ----------------------------------------------------

        total_time = (
            time.perf_counter() - pipeline_start
        )

        # ----------------------------------------------------
        # Step 5: Return structured results
        # ----------------------------------------------------

        return {
            "question": question,
            "answer": generation_result["answer"],

            "top_k": top_k,
            "fusion_method": self.fusion_method,
            "max_evidence_tokens": self.max_evidence_tokens,

            "sources": retrieved_passages,
            "prepared_sources": prepared_passages,

            "retrieval_time_seconds": retrieval_time,

            "generation_time_seconds":
                generation_result["generation_time_seconds"],

            "total_time_seconds": total_time,

            "input_tokens": generation_result["input_tokens"],
            "generated_tokens": generation_result["generated_tokens"],

            "device": generation_result["device"]
        }