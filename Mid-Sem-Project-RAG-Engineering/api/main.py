# ============================================================
# Stage C10.5 — FastAPI RAG Backend
#
# Project: Mid-Sem-Project-RAG-Engineering
# File: api/main.py
#
# Purpose:
# Expose the existing academic RAG engine through HTTP.
#
# Original academic RAG and Streamlit files are unchanged.
# ============================================================

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from adapters.rag_adapter import answer_question


# ------------------------------------------------------------
# 1. Initialize FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="Offline Wikipedia RAG Engineering API",
    description=(
        "Production Engineering API for the existing "
        "Offline Wikipedia RAG system."
    ),
    version="0.1.0"
)

from api.observability import install_observability
from api.security import install_security

install_security(app)
install_observability(app)


# ------------------------------------------------------------
# 2. Request schema
# ------------------------------------------------------------

class AnswerRequest(BaseModel):
    """
    Validate an incoming RAG question.
    """

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Question to ask the Wikipedia RAG system."
    )

    top_k: Literal[1, 3, 5] = Field(
        default=1,
        description="Number of passages to retrieve."
    )

    @field_validator("top_k", mode="before")
    @classmethod
    def reject_boolean_top_k(cls, value):
        if isinstance(value, bool):
            raise ValueError(
                "top_k must be an integer, not a boolean"
            )

        return value

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        return question


# ------------------------------------------------------------
# 3. Response schema
# ------------------------------------------------------------

class AnswerResponse(BaseModel):
    """
    Structured API response containing the answer
    and selected execution metrics.
    """

    question: str
    answer: str
    top_k: int

    retrieval_time_seconds: float
    generation_time_seconds: float
    total_time_seconds: float

    input_tokens: int
    generated_tokens: int

    device: str
    retrieved_passages: int


# ------------------------------------------------------------
# 4. API health endpoint
# ------------------------------------------------------------

@app.get(
    "/health",
    tags=["System"]
)
def health_check():
    """
    Verify that the API process is responding.

    Does not initialize or verify the RAG model.
    """

    return {
        "status": "healthy",
        "service": "offline-wikipedia-rag-engineering",
        "version": "0.1.0"
    }


# ------------------------------------------------------------
# 5. RAG question-answering endpoint
# ------------------------------------------------------------

@app.post(
    "/answer",
    response_model=AnswerResponse,
    tags=["RAG"]
)
def generate_answer(request: AnswerRequest):
    """
    Execute question answering using the existing
    academic RAG pipeline through the Engineering adapter.
    """

    result = answer_question(
        question=request.question,
        top_k=request.top_k
    )

    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        top_k=result["top_k"],

        retrieval_time_seconds=result[
            "retrieval_time_seconds"
        ],

        generation_time_seconds=result[
            "generation_time_seconds"
        ],

        total_time_seconds=result[
            "total_time_seconds"
        ],

        input_tokens=result["input_tokens"],
        generated_tokens=result["generated_tokens"],

        device=str(result["device"]),

        retrieved_passages=len(result["sources"])
    )