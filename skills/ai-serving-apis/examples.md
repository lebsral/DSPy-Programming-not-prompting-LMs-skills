# Serving APIs — Worked Examples

## Example 1: RAG API (document search behind FastAPI)

Three-layer separation: DSPy program → Pydantic models → FastAPI routes.

### `program.py` — DSPy logic (no API code here)

```python
import dspy

class AnswerFromDocs(dspy.Signature):
    """Answer the question based on the given context."""
    context: list[str] = dspy.InputField(desc="Relevant passages")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Answer grounded in the context")

class RAGProgram(dspy.Module):
    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.answer = dspy.ChainOfThought(AnswerFromDocs)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)
```

### `models.py` — Request/response schemas

```python
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    num_passages: int = Field(3, ge=1, le=10)
    model: str | None = None

class SearchResponse(BaseModel):
    answer: str
    passages: list[str]

class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    optimized: bool
```

### `server.py` — FastAPI routes

```python
from contextlib import asynccontextmanager
import dspy
from fastapi import FastAPI, HTTPException

from program import RAGProgram
from models import SearchRequest, SearchResponse, HealthResponse

MODEL_NAME = "openai/gpt-4o-mini"
PROGRAM_PATH = "optimized.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    lm = dspy.LM(MODEL_NAME)
    dspy.configure(lm=lm)

    app.state.program = RAGProgram()
    app.state.optimized = False
    try:
        app.state.program.load(PROGRAM_PATH)
        app.state.optimized = True
    except FileNotFoundError:
        pass
    yield

app = FastAPI(title="Document Search API", lifespan=lifespan)

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        program = app.state.program
        if request.model:
            with dspy.context(lm=dspy.LM(request.model)):
                result = program(question=request.question)
        else:
            result = program(question=request.question)
        return SearchResponse(
            answer=result.answer,
            passages=result.completions.context if hasattr(result, "completions") else [],
        )
    except Exception as e:
        if "rate limit" in str(e).lower():
            raise HTTPException(429, "Rate limited")
        raise HTTPException(500, "Search failed")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(model=MODEL_NAME, optimized=app.state.optimized)
```

Run it:

```bash
uvicorn server:app --reload --port 8000
# POST http://localhost:8000/search {"question": "How do I reset my password?"}
```

---

## Example 2: Classification API with batch endpoint

Sorting/classification behind FastAPI with single and batch endpoints.

### `server.py`

```python
from contextlib import asynccontextmanager
import dspy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- DSPy program ---
class ClassifyTicket(dspy.Signature):
    """Classify a support ticket into a category."""
    text: str = dspy.InputField(desc="Support ticket text")
    category: str = dspy.OutputField(desc="Category: billing, technical, account, other")
    reasoning: str = dspy.OutputField(desc="Why this category")

class Classifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(ClassifyTicket)
    def forward(self, text):
        return self.classify(text=text)

# --- Pydantic models ---
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)

class ClassifyResponse(BaseModel):
    category: str
    reasoning: str

# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    app.state.classifier = Classifier()
    try:
        app.state.classifier.load("optimized.json")
    except FileNotFoundError:
        pass
    yield

app = FastAPI(title="Classification API", lifespan=lifespan)

@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    result = app.state.classifier(text=request.text)
    return ClassifyResponse(category=result.category, reasoning=result.reasoning)

@app.post("/classify/batch", response_model=list[ClassifyResponse])
async def classify_batch(requests: list[ClassifyRequest]):
    results = []
    for req in requests:
        result = app.state.classifier(text=req.text)
        results.append(ClassifyResponse(category=result.category, reasoning=result.reasoning))
    return results
```

```bash
# Single
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for my subscription"}'

# Batch
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '[{"text": "I was charged twice"}, {"text": "App keeps crashing"}]'
```

---

## Streaming progress updates

DSPy doesn't natively stream token-by-token output. But you can stream progress for multi-step pipelines using Server-Sent Events (SSE):

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/search/stream")
async def search_stream(request: SearchRequest):
    async def generate():
        # Step 1: retrieve
        yield f"data: {json.dumps({'step': 'retrieving', 'status': 'in_progress'})}\n\n"
        passages = app.state.program.retrieve(request.question).passages
        yield f"data: {json.dumps({'step': 'retrieving', 'status': 'done', 'count': len(passages)})}\n\n"

        # Step 2: generate answer
        yield f"data: {json.dumps({'step': 'answering', 'status': 'in_progress'})}\n\n"
        result = app.state.program.answer(context=passages, question=request.question)
        yield f"data: {json.dumps({'step': 'answering', 'status': 'done', 'answer': result.answer})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

This is useful when your pipeline has multiple visible steps (retrieve → reason → answer) and you want the frontend to show progress. For single-step programs, a regular POST endpoint is simpler.
