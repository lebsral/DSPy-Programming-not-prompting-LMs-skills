# Async DSPy API Reference

> Condensed from [dspy.ai/api/modules](https://dspy.ai/api/modules/). Verify against upstream for latest.

## aforward()

Available on all DSPy modules (Predict, ChainOfThought, ReAct, CodeAct, custom Module subclasses).

```python
result = await module.aforward(**kwargs)
```

| Method | Equivalent sync | Description |
|--------|----------------|-------------|
| `module.aforward(**kwargs)` | `module.forward(**kwargs)` | Async forward pass |
| `module.acall(**kwargs)` | `module(**kwargs)` | Alias for aforward (convenience) |

**Returns:** `dspy.Prediction` (same as sync version)

## acall()

Convenience alias for `aforward()`. Identical behavior:

```python
# These are equivalent
result = await module.acall(question="...")
result = await module.aforward(question="...")
```

## Async patterns

### Basic async call

```python
import asyncio
import dspy

module = dspy.ChainOfThought("question -> answer")

async def main():
    result = await module.aforward(question="...")
    return result.answer

asyncio.run(main())
```

### Concurrent calls (asyncio.gather)

```python
async def concurrent_calls(inputs: list[str]):
    tasks = [module.aforward(question=q) for q in inputs]
    return await asyncio.gather(*tasks)
```

### Semaphore-limited concurrency

```python
semaphore = asyncio.Semaphore(10)

async def limited_call(**kwargs):
    async with semaphore:
        return await module.aforward(**kwargs)
```

### Timeout

```python
async def with_timeout(**kwargs):
    return await asyncio.wait_for(
        module.aforward(**kwargs),
        timeout=30.0,
    )
```

## Custom async modules

```python
class MyModule(dspy.Module):
    def __init__(self):
        self.step1 = dspy.Predict("input -> intermediate")
        self.step2 = dspy.ChainOfThought("intermediate -> output")

    async def aforward(self, input):
        mid = await self.step1.aforward(input=input)
        result = await self.step2.aforward(intermediate=mid.intermediate)
        return result
```

## Batch processing

DSPy modules also have a `.batch()` method for built-in batch processing:

```python
# Sync batch
results = module.batch(
    [{"question": q} for q in questions],
    num_threads=10,
)

# For async batch processing, use asyncio.gather with semaphore
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `examples` | `list[dict]` | required | List of input kwarg dicts |
| `num_threads` | `int` | `2` | Number of concurrent threads |

## Framework integration

### FastAPI

```python
@app.post("/endpoint")
async def endpoint(input: str):
    result = await module.aforward(input=input)
    return {"output": result.output}
```

### Starlette

```python
async def homepage(request):
    result = await module.aforward(input=request.query_params["q"])
    return JSONResponse({"output": result.output})
```
