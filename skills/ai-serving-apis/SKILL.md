---
name: ai-serving-apis
description: Put your AI behind an API. Use when you need to serve AI features as web endpoints, add AI to an existing backend, deploy AI for other services to call, wrap a DSPy program in REST/HTTP, build an AI microservice, or put a language model behind FastAPI. Covers FastAPI integration, program loading, request/response models, per-request configuration, error handling, and Docker deployment.
---

# Put Your AI Behind an API

Guide the user through wrapping a DSPy program in a web API so other services (or a frontend) can call it over HTTP. Uses FastAPI for the web layer, with clean separation between DSPy logic and API code.

## When you need this

- You built an AI feature and need to serve it as a web endpoint
- You're adding AI capabilities to an existing backend
- Other services need to call your AI over HTTP
- You want to deploy your AI with Docker

## Step 1: Understand the setup

Ask the user:
1. **What DSPy program are you serving?** (classification, RAG, extraction, pipeline, etc.)
2. **Is it optimized?** (do you have an `optimized.json` from `/ai-improving-accuracy`?)
3. **What endpoints do you need?** (single query, batch, health check, etc.)
4. **Do you have an existing web framework?** (FastAPI, Flask, Django — default to FastAPI)

## Step 2: Project structure

Recommended layout — keep DSPy logic separate from API code:

```
project/
├── program.py       # DSPy module (already exists from /ai-kickoff)
├── server.py        # FastAPI app — routes and startup
├── models.py        # Pydantic request/response schemas
├── config.py        # Environment configuration
├── optimized.json   # Saved optimized program (if available)
├── requirements.txt
├── Dockerfile
└── .env.example
```

## Step 3: Define request/response models

Use Pydantic models for all inputs and outputs. This gives you validation, documentation, and serialization for free.

```python
# models.py
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request to the AI endpoint."""
    query: str = Field(..., description="The input to process", min_length=1)
    # Optional: let callers override the model per request
    model: str | None = Field(None, description="Override the default LM")
    temperature: float | None = Field(None, ge=0, le=2, description="Override temperature")

class QueryResponse(BaseModel):
    """Response from the AI endpoint."""
    answer: str
    # Include whatever your DSPy program outputs
    # reasoning: str | None = None
    # confidence: float | None = None

class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    optimized: bool
```

Adapt fields to match your DSPy program's inputs and outputs. If your program takes `question` and returns `answer` and `reasoning`, mirror that in the schemas.

## Step 4: Load the optimized program at startup

Load the DSPy program once when the server starts, not on every request. Use FastAPI's lifespan handler:

```python
# server.py
from contextlib import asynccontextmanager
import dspy
from fastapi import FastAPI

from program import MyProgram
from config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load DSPy program once at startup."""
    # Configure the default LM
    lm = dspy.LM(settings.model_name)
    dspy.configure(lm=lm)

    # Load the program (with optimization if available)
    app.state.program = MyProgram()
    app.state.optimized = False
    try:
        app.state.program.load(settings.program_path)
        app.state.optimized = True
        print(f"Loaded optimized program from {settings.program_path}")
    except FileNotFoundError:
        print("Running unoptimized program")

    yield  # Server runs here

app = FastAPI(title="My AI API", lifespan=lifespan)
```

**Why lifespan?** Loading a program + configuring an LM takes time. Doing it once at startup means requests are fast. The `app.state` object makes the program available to all route handlers.

## Step 5: Create endpoints

### Query endpoint

```python
# server.py (continued)
from fastapi import HTTPException
from models import QueryRequest, QueryResponse, HealthResponse

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run the AI program on input."""
    program = app.state.program

    # If caller wants a different model, use dspy.context for this request only
    if request.model or request.temperature is not None:
        lm_kwargs = {}
        if request.model:
            lm_kwargs["model"] = request.model
        if request.temperature is not None:
            lm_kwargs["temperature"] = request.temperature
        override_lm = dspy.LM(**lm_kwargs) if request.model else dspy.LM(
            settings.model_name, temperature=request.temperature
        )
        with dspy.context(lm=override_lm):
            result = program(query=request.query)
    else:
        result = program(query=request.query)

    return QueryResponse(answer=result.answer)
```

### Health check

```python
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        model=settings.model_name,
        optimized=app.state.optimized,
    )
```

### Batch endpoint

For processing multiple inputs at once:

```python
@app.post("/query/batch", response_model=list[QueryResponse])
async def query_batch(requests: list[QueryRequest]):
    """Process multiple inputs."""
    program = app.state.program
    results = []
    for req in requests:
        result = program(query=req.query)
        results.append(QueryResponse(answer=result.answer))
    return results
```

## Step 6: Handle errors

Map DSPy errors to appropriate HTTP status codes:

```python
from dspy.primitives.assertions import DSPyAssertionError

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    program = app.state.program
    try:
        if request.model or request.temperature is not None:
            override_lm = dspy.LM(
                request.model or settings.model_name,
                temperature=request.temperature,
            )
            with dspy.context(lm=override_lm):
                result = program(query=request.query)
        else:
            result = program(query=request.query)
        return QueryResponse(answer=result.answer)

    except DSPyAssertionError as e:
        # AI output failed validation (from dspy.Assert)
        raise HTTPException(status_code=422, detail=f"Output validation failed: {e}")
    except Exception as e:
        error_msg = str(e).lower()
        if "rate limit" in error_msg or "429" in error_msg:
            raise HTTPException(status_code=429, detail="Rate limited by AI provider")
        if "timeout" in error_msg:
            raise HTTPException(status_code=504, detail="AI provider timed out")
        raise HTTPException(status_code=500, detail="Internal error processing request")
```

## Step 7: Environment configuration

Use pydantic-settings to manage configuration:

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = "openai/gpt-4o-mini"
    program_path: str = "optimized.json"
    api_key: str = ""  # Set via environment variable

    model_config = {"env_prefix": "AI_"}

settings = Settings()
```

```
# .env.example
AI_MODEL_NAME=openai/gpt-4o-mini
AI_PROGRAM_PATH=optimized.json
AI_API_KEY=your-api-key-here
```

## Step 8: Run and deploy

### Run locally

```bash
pip install fastapi uvicorn pydantic-settings
uvicorn server:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for auto-generated API docs.

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```
dspy>=2.5
fastapi>=0.100
uvicorn[standard]
pydantic-settings>=2.0
```

Add provider-specific packages as needed (e.g., `openai`, `anthropic`).

### Docker Compose (optional)

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./optimized.json:/app/optimized.json:ro
```

## Key patterns

- **Load once, serve many.** Load the program and LM at startup via lifespan, not per request.
- **Pydantic everything.** Request/response models give you validation, docs, and serialization.
- **`dspy.context()` for overrides.** Let callers switch models or temperature without affecting other requests.
- **Separate DSPy from API code.** Keep `program.py` independent — the same module runs in scripts, tests, and the API.
- **Map errors to HTTP codes.** Assertion failures → 422, rate limits → 429, timeouts → 504.

## Additional resources

- For worked examples (RAG API, classification API, streaming), see [examples.md](examples.md)
- Use `/ai-kickoff` to scaffold a new project (add `--api` structure with Step 2b)
- Use `/ai-searching-docs` to build the RAG program to serve
- Use `/ai-monitoring` to monitor your deployed API
- Use `/ai-cutting-costs` to optimize API costs in production
