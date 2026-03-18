"""FastAPI app template for serving DSPy programs.

Copy this file into your project and customize:
1. Update the signature and program in build_program()
2. Update request/response models to match your signature
3. Run: pip install fastapi uvicorn dspy && uvicorn app:app --reload
"""

from contextlib import asynccontextmanager

import dspy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def build_program() -> dspy.Module:
    """Replace with your DSPy program setup."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)
    return dspy.ChainOfThought("question -> answer")


program = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global program
    program = build_program()
    yield


app = FastAPI(title="DSPy API", lifespan=lifespan)


# --- Customize these models to match your signature ---

class Request(BaseModel):
    question: str


class Response(BaseModel):
    answer: str


@app.post("/predict", response_model=Response)
async def predict(req: Request):
    try:
        result = program(**req.model_dump())
        return Response(answer=result.answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "program_loaded": program is not None}
