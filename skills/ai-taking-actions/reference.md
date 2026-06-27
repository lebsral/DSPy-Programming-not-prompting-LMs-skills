> Condensed from [dspy.ai/api/modules/ReAct](https://dspy.ai/api/modules/ReAct/) and [dspy.ai/api/primitives/Tool](https://dspy.ai/api/primitives/Tool). Verify against upstream for latest.

# DSPy API Reference for Action-Taking AI

## dspy.ReAct

[API docs](https://dspy.ai/api/modules/ReAct/)

```python
dspy.ReAct(signature, tools, max_iters=20)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Input/output contract — e.g. `"question -> answer"` |
| `tools` | `list[Callable \| dspy.Tool]` | required | Functions the agent can call; accepts plain callables, `dspy.Tool`, or converted LangChain tools |
| `max_iters` | `int` | `20` | Max Thought-Action-Observation cycles; lower to 5-10 to control cost |

A "finish" tool is added automatically — the agent calls it to signal task completion.

**Return value:** `dspy.Prediction` with output fields matching your signature (e.g. `.answer`) plus `.trajectory` — a dict of the full Thought-Action-Observation trace.

## dspy.CodeAct

```python
dspy.CodeAct(signature, tools, max_iters=5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Input/output contract |
| `tools` | `list[Callable]` | required | **Pure functions only** — no callable objects, class instances, or functions with undeclared dependencies |
| `max_iters` | `int` | `5` | Max code-write-and-execute cycles |

Use CodeAct when the task is inherently code-centric (math, data transforms). Use ReAct for everything else.

## dspy.Tool

[API docs](https://dspy.ai/api/primitives/Tool)

```python
dspy.Tool(func, name=None, desc=None, args=None, arg_types=None, arg_desc=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | required | Function to wrap |
| `name` | `str \| None` | `None` | Tool name (auto-inferred from `func.__name__`) |
| `desc` | `str \| None` | `None` | Description (auto-inferred from docstring) |
| `arg_desc` | `dict \| None` | `None` | Per-argument descriptions |

Most tools don't need explicit wrapping — pass plain callables directly to `dspy.ReAct`. Use `dspy.Tool` only when you need to override the name, description, or argument schemas that DSPy infers automatically.

### Conversion class methods

```python
dspy.Tool.from_langchain(tool: BaseTool) -> dspy.Tool
dspy.Tool.from_mcp_tool(session: mcp.ClientSession, tool: mcp.types.Tool) -> dspy.Tool  # async
```

## dspy.PythonInterpreter

```python
dspy.PythonInterpreter().execute(expression)  # returns the evaluated result
```

Used inside tool functions to safely evaluate mathematical expressions. Requires [Deno](https://docs.deno.com/runtime/getting_started/installation/).

Wrap in a tool function rather than calling directly from agent setup:

```python
def evaluate_math(expression: str) -> float:
    """Evaluate a mathematical expression and return the result."""
    return dspy.PythonInterpreter().execute(expression)
```

## Accessing .trajectory

```python
result = agent(question="What is 9 * 7?")
print(result.trajectory)  # dict with Thought/Action/Observation steps
```

Use `.trajectory` to debug why the agent took a particular path or to log agentic behavior.

## Tool requirements

Any callable passed to `dspy.ReAct` must have:
- Type hints on all parameters and the return type
- A docstring describing what the tool does and when to use it — this is the agent's only guidance for tool selection
- A return value that converts cleanly to `str` — returning dicts, lists, or custom objects causes misinterpretation

## Quick-reference

| | ReAct | CodeAct |
|---|---|---|
| **Default `max_iters`** | `20` | `5` |
| **Tool types accepted** | Any callable, `dspy.Tool`, LangChain tools | Pure functions only |
| **Best for** | APIs, search, databases — general-purpose agents | Math, data transforms — tasks where writing code is natural |
| **Start here?** | Yes | When the task is inherently code-centric |

## Install

```bash
pip install -U dspy
pip install langchain-community  # optional — for LangChain tools (DuckDuckGo, Wikipedia, SQL, etc.)
```

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)  # must be called before instantiating any agent
```
