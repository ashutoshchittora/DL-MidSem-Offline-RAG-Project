# Why separate the generator from the retriever?
# Suppose we later replace TF-IDF retrieval with a vector database.
# The generator should still work without modification because its input remains a collection of retrieved passages.
# That's the benefit of designing reusable components.


# ============================================================
# Stage 5.3 — Reusable Flan-T5 Generator
# ============================================================

from pathlib import Path
import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


class FlanT5Generator:
    """
    Generate evidence-grounded answers using Flan-T5.
    """
    ### pre offline model load 

    # def __init__(
    #     self,
    #     model_name="google/flan-t5-small",
    #     max_input_tokens=512,
    #     max_new_tokens=64,
    #     device=None
    # ):
    #     """
    #     Initialize tokenizer and pretrained model.
    #     """

    #     self.model_name = model_name
    #     self.max_input_tokens = max_input_tokens
    #     self.max_new_tokens = max_new_tokens

    #     # Select GPU when available
    #     self.device = torch.device(
    #         device if device is not None
    #         else ("cuda" if torch.cuda.is_available() else "cpu")
    #     )

    #     # Load pretrained tokenizer
    #     self.tokenizer = AutoTokenizer.from_pretrained(
    #         self.model_name
    #     )

    #     # Load pretrained model
    #     self.model = AutoModelForSeq2SeqLM.from_pretrained(
    #         self.model_name
    #     )

    #     # Move model to selected device
    #     self.model = self.model.to(self.device)

    #     # Enable evaluation behaviour
    #     self.model.eval()

    def __init__(
    self,
    model_name="google/flan-t5-small",
    max_input_tokens=512,
    max_new_tokens=64,
    device=None,
    local_model_path=None
):
        """
        Initialize tokenizer and pretrained model.

        Loading priority:
        1. Valid local model directory, if supplied.
        2. Existing Hugging Face model-loading behavior.

        Existing callers remain backward-compatible.
        """

        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

        # ----------------------------------------------------
        # Step 1: Select execution device
        # ----------------------------------------------------

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ----------------------------------------------------
        # Step 2: Determine model loading source
        # ----------------------------------------------------

        model_source = self.model_name
        using_local_model = False

        if local_model_path is not None:

            local_path = Path(local_model_path)

            if local_path.exists():

                if not local_path.is_dir():
                    raise ValueError(
                        f"Local model path is not a directory: "
                        f"{local_path}"
                    )

                required_files = [
                    "config.json",
                    "tokenizer_config.json",
                ]

                missing_files = [
                    filename
                    for filename in required_files
                    if not (local_path / filename).is_file()
                ]

                has_model_weights = any(
                    (local_path / filename).is_file()
                    for filename in [
                        "model.safetensors",
                        "pytorch_model.bin"
                    ]
                )

                has_tokenizer = any(
                    (local_path / filename).is_file()
                    for filename in [
                        "tokenizer.json",
                        "spiece.model"
                    ]
                )

                if missing_files or not has_model_weights or not has_tokenizer:

                    raise ValueError(
                        "Local model directory is incomplete: "
                        f"{local_path}. "
                        f"Missing required files: {missing_files}. "
                        f"Model weights present: {has_model_weights}. "
                        f"Tokenizer present: {has_tokenizer}."
                    )

                model_source = str(local_path.resolve())
                using_local_model = True

        # ----------------------------------------------------
        # Step 3: Load tokenizer and model
        # ----------------------------------------------------

        if using_local_model:

            print(f"Loading local model: {model_source}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                local_files_only=True
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_source,
                local_files_only=True
            )

        else:

            print(f"Loading Hugging Face model: {self.model_name}")

            # Original behavior preserved
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name
            )

        # ----------------------------------------------------
        # Step 4: Move model to execution device
        # ----------------------------------------------------

        self.model = self.model.to(self.device)

        # Enable evaluation behavior
        self.model.eval()

        # Record actual source for verification
        self.model_source = model_source
        self.using_local_model = using_local_model

    def build_prompt(self, question, passages):
        """
        Construct a grounded question-answering prompt.

        passages: list of dictionaries containing a 'text' field.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be non-empty.")

        if not passages:
            raise ValueError("At least one evidence passage is required.")

        evidence = "\n\n".join(
            passage["text"]
            for passage in passages
        )

        prompt = (
            "Answer the question using only the evidence below. "
            "If the evidence does not contain the answer, say "
            "'Insufficient evidence'.\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        return prompt

    def generate(self, question, passages):
        """
        Generate an answer using retrieved evidence.
        """

        # Step 1: Construct prompt
        prompt = self.build_prompt(
            question,
            passages
        )

        # Step 2: Tokenize WITHOUT truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False
        )

        input_token_count = inputs["input_ids"].shape[1]

        # Step 3: Enforce input token budget
        if input_token_count > self.max_input_tokens:
            raise ValueError(
                f"Prompt requires {input_token_count} tokens, "
                f"but maximum is {self.max_input_tokens}. "
                "Reduce retrieval size or evidence length."
            )

        # Step 4: Move inputs to model device
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Step 5: Generate answer
        self.model.eval()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        start_time = time.perf_counter()

        with torch.inference_mode():

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        generation_time = time.perf_counter() - start_time

        # Step 6: Decode model output
        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return {
            "question": question,
            "answer": answer,
            "input_tokens": input_token_count,
            "generated_tokens": int(output_ids.shape[1]),
            "generation_time_seconds": generation_time,
            "device": str(self.device),
            "evidence_count": len(passages)
        }