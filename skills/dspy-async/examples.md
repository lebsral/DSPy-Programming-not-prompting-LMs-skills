# Async DSPy Examples

## Example 1: FastAPI with concurrent classification

A production endpoint that classifies multiple items concurrently:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import dspy

app = FastAPI()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

classifier = dspy.Predict("text -> label, confidence: float")
semaphore = asyncio.Semaphore(20)  # Max 20 concurrent LM calls


class BatchRequest(BaseModel):
    texts: list[str]


class ClassificationResult(BaseModel):
    label: str
    confidence: float


@app.post("/classify-batch")
async def classify_batch(request: BatchRequest):
    async def classify_one(text: str):
        async with semaphore:
            result = await classifier.aforward(text=text)
            return ClassificationResult(
                label=result.label,
                confidence=float(result.confidence),
            )

    tasks = [classify_one(text) for text in request.texts]
    results = await asyncio.gather(*tasks)
    return {"results": results}
```

## Example 2: Parallel research with timeout

Run multiple research queries concurrently with a timeout:

```python
import asyncio
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

researcher = dspy.ChainOfThought("topic -> findings, sources")


async def research_with_timeout(topics: list[str], timeout_seconds: float = 30.0):
    """Research multiple topics concurrently with a global timeout."""

    async def research_one(topic: str):
        result = await researcher.aforward(topic=topic)
        return {"topic": topic, "findings": result.findings}

    tasks = [research_one(topic) for topic in topics]

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout_seconds,
        )
        # Filter out exceptions
        return [r for r in results if isinstance(r, dict)]
    except asyncio.TimeoutError:
        return [{"error": "Research timed out"}]


# Usage
topics = ["quantum computing advances", "CRISPR applications", "fusion energy progress"]
findings = asyncio.run(research_with_timeout(topics, timeout_seconds=20.0))
for f in findings:
    print(f"{f['topic']}: {f['findings'][:100]}...")
```

## Example 3: Async pipeline with error recovery

A multi-step async pipeline that handles failures gracefully:

```python
import asyncio
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class RobustPipeline(dspy.Module):
    def __init__(self):
        self.extract = dspy.Predict("document -> entities: list[str]")
        self.enrich = dspy.ChainOfThought("entity -> description, category")

    async def aforward(self, document):
        # Step 1: Extract entities
        extraction = await self.extract.aforward(document=document)
        entities = extraction.entities

        # Step 2: Enrich each entity concurrently (with error handling)
        async def enrich_one(entity: str):
            try:
                result = await self.enrich.aforward(entity=entity)
                return {
                    "entity": entity,
                    "description": result.description,
                    "category": result.category,
                }
            except Exception as e:
                return {"entity": entity, "error": str(e)}

        enriched = await asyncio.gather(*[enrich_one(e) for e in entities])

        return dspy.Prediction(
            entities=entities,
            enriched=[e for e in enriched if "error" not in e],
            errors=[e for e in enriched if "error" in e],
        )


pipeline = RobustPipeline()
result = asyncio.run(pipeline.aforward(document="Apple announced new AI features..."))
print(f"Enriched {len(result.enriched)} entities, {len(result.errors)} errors")
```
