---
name: dspy-utils
description: "Use when you need DSPy infrastructure: streaming responses, caching control, debugging with inspect_history, saving/loading programs, async execution, or MCP integration. Common scenarios: enabling streaming responses in production, controlling the cache to avoid stale results, debugging with inspect_history to see raw prompts, saving and loading optimized programs, running DSPy modules asynchronously, or integrating with MCP servers. Related: ai-tracing-requests, ai-serving-apis, ai-monitoring. Also: \"dspy.inspect_history\", \"dspy.settings.configure\", \"streaming DSPy output\", \"cache control in DSPy\", \"save and load DSPy program\", \"async DSPy execution\", \"MCP integration with DSPy\", \"debug DSPy prompts\", \"see what DSPy sent to the model\", \"DSPy program serialization\", \"production DSPy utilities\", \"clear DSPy cache\", \"view prompt history\", \"async await with DSPy\", \"stream tokens from DSPy\"."
---

# DSPy Utilities: Streaming, Caching, Debugging, and More

Guide the user through DSPy's utility functions for production workflows -- streaming LM outputs, controlling caching, debugging calls, persisting optimized programs, running async, integrating MCP tools, and enforcing runtime constraints.

## 1. StreamListener and streamify -- streaming LM outputs

Use `streamify` to wrap a DSPy program so it yields output tokens incrementally instead of waiting for the full response. Use `StreamListener` to capture streaming output from specific signature fields.

### streamify

```python
from dspy.streaming import streamify, StreamListener

# Wrap any DSPy module for streaming
streaming_program = streamify(
    program,                                      # The DSPy Module to stream
    stream_listeners=[...],                       # List of StreamListener instances
    include_final_prediction_in_output_stream=True,  # Include final Prediction in stream
    is_async_program=False,                       # Set True if program is already async
    async_streaming=True,                         # True for async generator, False for sync
    status_message_provider=None,                 # Custom status messages (optional)
)
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `program` | `Module` | required | The DSPy module to enable streaming on |
| `stream_listeners` | `list[StreamListener]` | `None` | Captures streaming output from specific fields |
| `include_final_prediction_in_output_stream` | `bool` | `True` | Whether the final `Prediction` appears in the stream |
| `is_async_program` | `bool` | `False` | Set `True` if the wrapped program is already async |
| `async_streaming` | `bool` | `True` | `True` returns an async generator; `False` returns sync |
| `status_message_provider` | `StatusMessageProvider` | `None` | Custom status messages for tracking progress |

**Returns:** A callable that returns an async (or sync) generator yielding incremental outputs.

### StreamListener

```python
listener = StreamListener(
    signature_field_name="answer",   # Which output field to stream
    predict=None,                    # Predictor to monitor (auto-detected if None)
    predict_name=None,               # Name identifier for the predictor
    allow_reuse=False,               # Allow reuse across multiple streams
)
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `signature_field_name` | `str` | required | The output field name to listen to |
| `predict` | `Any` | `None` | Predictor to monitor; auto-detected if `None` |
| `predict_name` | `str \| None` | `None` | Name identifier for the predictor |
| `allow_reuse` | `bool` | `False` | Permit reuse across multiple streams (may hurt performance) |

**Key methods:**

- `receive(chunk)` -- processes incoming streaming chunks, manages buffering
- `finalize()` -- flushes remaining buffered tokens at stream end, returns final chunk with `is_last_chunk=True`
- `flush()` -- flushes all tokens in the buffer, clears it

### Minimal streaming example

```python
import dspy
from dspy.streaming import streamify, StreamListener

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question -> answer")

# Create a listener for the "answer" field
answer_listener = StreamListener(signature_field_name="answer")

# Wrap the program for streaming
streaming_qa = streamify(
    qa,
    stream_listeners=[answer_listener],
)

# Consume the async generator
async for chunk in streaming_qa(question="What is DSPy?"):
    if hasattr(chunk, "answer"):
        print(chunk.answer, end="", flush=True)
```

### Notes on streaming

- The listener buffers ~10 tokens internally to detect field boundary delimiters before yielding
- Adapter format matters: `ChatAdapter` uses `[[ ## field_name ## ]]` delimiters, `JSONAdapter` uses partial JSON parsing
- Set `allow_reuse=True` only when you need the same listener across multiple streams -- it duplicates chunk processing and can hurt performance
- The final `Prediction` object appears as the last item in the stream (unless `include_final_prediction_in_output_stream=False`)

## 2. configure_cache -- controlling cache behavior

DSPy caches LM responses by default to reduce costs and speed up development. Use `dspy.configure_cache` to control this globally.

```python
# Disable caching entirely
dspy.configure_cache(enable=False)

# Re-enable caching
dspy.configure_cache(enable=True)
```

### Per-LM cache control

You can also control caching per LM instance:

```python
# This LM never caches
lm_no_cache = dspy.LM("openai/gpt-4o-mini", cache=False)

# This LM caches (default)
lm_cached = dspy.LM("openai/gpt-4o-mini", cache=True)
```

### When to disable caching

- **Generating diverse outputs** -- when you need different responses for the same prompt (e.g., data generation)
- **Testing real latency** -- cache hits are instant, which skews benchmarks
- **Streaming** -- caching may interfere with streaming behavior in some configurations

Cache is stored locally on disk. Identical calls (same prompt, parameters, model) return cached results with no API call.

## 3. inspect_history -- debugging LM calls

`dspy.inspect_history` shows the raw prompts and responses from recent LM calls. This is the single most useful debugging tool in DSPy.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

classify = dspy.Predict("text -> label")
classify(text="Great product!")

# See what was actually sent to and received from the LM
dspy.inspect_history(n=1)  # Show last 1 call
dspy.inspect_history(n=3)  # Show last 3 calls
```

### What inspect_history shows

- The full prompt sent to the LM (including system message, few-shot demos, instructions)
- The raw LM response
- Which adapter formatted the prompt (ChatAdapter, JSONAdapter, etc.)

### Debugging workflow

1. Run your program on a failing input
2. Call `dspy.inspect_history(n=1)` to see the last LM call
3. Check if the prompt makes sense -- are the instructions clear? Are few-shot demos relevant?
4. Check the raw response -- did the LM follow the format? Did it hallucinate?
5. Adjust your signature, module, or optimization strategy based on what you see

### Verbose logging

For more detailed tracing, configure DSPy with an empty trace list:

```python
dspy.configure(lm=lm, trace=[])
```

You can also print a module to see its structure:

```python
print(my_program)  # Shows module tree with all sub-modules and signatures
```

## 4. save/load -- persisting optimized programs

After optimizing a DSPy program, save its learned state (few-shot demos, instructions) for production use.

### Save

```python
# After optimization
optimized = optimizer.compile(my_program, trainset=trainset)
optimized.save("optimized_program.json")
```

### Load

```python
# In production -- create a fresh instance, then load state
program = MyProgram()
program.load("optimized_program.json")

# Use it
result = program(question="What is DSPy?")
```

### What gets saved

- Few-shot demonstrations discovered by optimizers
- Optimized instructions (from MIPROv2, GEPA, etc.)
- Any state tracked by `dspy.Predict` modules

### What does NOT get saved

- Python logic in `forward()` -- that's your code, it must exist at load time
- Model weights (unless you used `BootstrapFinetune`)
- LM configuration -- you must call `dspy.configure()` before loading

### Production deployment pattern

```python
import dspy

class MyPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> category")
        self.respond = dspy.ChainOfThought("text, category -> response")

    def forward(self, text):
        cat = self.classify(text=text)
        return self.respond(text=text, category=cat.category)

# --- Optimization (run once) ---
# optimizer = dspy.MIPROv2(metric=metric, auto="medium")
# optimized = optimizer.compile(MyPipeline(), trainset=trainset)
# optimized.save("pipeline_v1.json")

# --- Production (run on every request) ---
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

pipeline = MyPipeline()
pipeline.load("pipeline_v1.json")

result = pipeline(text="How do I reset my password?")
```

## 5. asyncify -- running DSPy programs asynchronously

`dspy.asyncify` wraps a synchronous DSPy program so it can run as an async function, useful for web servers and concurrent workloads.

```python
import dspy
import asyncio

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question -> answer")

# Wrap for async execution
async_qa = dspy.asyncify(qa)

# Use in async code
async def main():
    result = await async_qa(question="What is DSPy?")
    print(result.answer)

asyncio.run(main())
```

### Running multiple programs concurrently

```python
async def process_batch(questions):
    async_qa = dspy.asyncify(qa)
    tasks = [async_qa(question=q) for q in questions]
    results = await asyncio.gather(*tasks)
    return [r.answer for r in results]
```

### Key details

- **Context propagation** -- `asyncify` captures the current thread's `dspy.configure` settings and propagates them to the worker thread
- **Thread-safe** -- each async call maintains its own configuration state
- **Cancel-safe** -- uses `abandon_on_cancel=True` internally for clean cancellation
- A fresh asyncified callable is created on each invocation to capture the latest context

## 6. MCP integration -- using DSPy with MCP servers

DSPy can connect to Model Context Protocol (MCP) servers, converting MCP tools into DSPy tools for use with agents like `dspy.ReAct`.

### Installation

```bash
pip install -U "dspy[mcp]"
```

### Connecting to an MCP server

```python
import dspy
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# Define the MCP server to connect to
server_params = StdioServerParameters(
    command="python",
    args=["path/to/your/mcp_server.py"],
    env=None,
)

async def main():
    # Connect to the MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            response = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # Use with a ReAct agent
            agent = dspy.ReAct(
                signature="question -> answer",
                tools=dspy_tools,
                max_iters=5,
            )
            result = await agent.acall(question="What files are in the project?")
            print(result.answer)
```

### Remote MCP servers (HTTP)

```python
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        # Same tool conversion pattern as above
```

### What from_mcp_tool preserves

- Tool name and description
- Parameter schemas and types
- Async execution support

## 7. dspy.Assert and dspy.Suggest -- runtime constraints

Use assertions inside `forward()` to enforce output constraints. DSPy automatically retries on failures.

```python
class SafeQA(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        result = self.generate(question=question)

        # Hard constraint -- raises error, triggers retry
        dspy.Assert(result.answer != "I don't know", "Must provide a substantive answer")

        # Soft constraint -- adds feedback to prompt on retry, doesn't fail
        dspy.Suggest(len(result.answer.split()) >= 10, "Answer should be detailed")

        return result
```

- **`dspy.Assert(condition, message)`** -- hard constraint. If `False`, DSPy retries the prediction (up to a limit). Use for requirements that must be met.
- **`dspy.Suggest(condition, message)`** -- soft constraint. Adds feedback but doesn't raise an error. Use for quality preferences.

Assertions work with optimizers -- the optimizer learns to avoid triggering them. For more detail on using assertions in modules, see `/dspy-modules`.

## Cross-references

- **`/dspy-lm`** -- Configure language models, per-LM caching, `inspect_history` on LM instances
- **`/dspy-modules`** -- Build composable programs with `dspy.Module`, assertions in modules, save/load patterns
- **`/ai-tracing-requests`** -- Production observability and tracing for DSPy programs
- **`/ai-serving-apis`** -- Serve DSPy programs as web APIs (pairs well with streaming and asyncify)
- For worked examples, see [examples.md](examples.md)
