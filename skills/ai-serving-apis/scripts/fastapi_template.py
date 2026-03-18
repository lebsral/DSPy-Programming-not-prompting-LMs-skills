"""FastAPI template for serving a DSPy program as a REST API.

Usage:
    1. Copy this file into your project
    2. Replace `build_program()` with your actual DSPy program
    3. Run: uvicorn fastapi_template:app --reload

Or use as a reference when building your own API wrapper.
"""

from contextlib import asynccontextmanager

import dspy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# --- Configure your DSPy program here ---

def build_program() -> dspy.Module:
    """Build and return your DSPy program.

    Replace this with your actual program setup.
    """
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    program = dspy.ChainOfThought("question -> answer")
    # To load an optimized program:
    # program.load("path/to/optimized_program.json")
    return program


# --- API setup ---

program = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global program
    program = build_program()
    yield


app = FastAPI(
    title="DSPy API",
    description="REST API serving a DSPy program",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Request body — add your input fields here."""
    question: str


class PredictResponse(BaseModel):
    """Response body — add your output fields here."""
    answer: str


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run the DSPy program on the input."""
    try:
        result = program(**request.model_dump())
        return PredictResponse(answer=result.answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "program_loaded": program is not None}
