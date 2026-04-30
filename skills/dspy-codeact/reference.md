> Condensed from [dspy.ai/api/modules/CodeAct/](https://dspy.ai/api/modules/CodeAct/). Verify against upstream for latest.

# dspy.CodeAct — API Reference

## Constructor

```python
dspy.CodeAct(
    signature,          # str | type[Signature] (required)
    tools,              # list[Callable] (required)
    max_iters=5,        # int
    interpreter=None,   # PythonInterpreter | None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str | type[Signature]` | required | Defines input/output fields for the agent. |
| `tools` | `list[Callable]` | required | Pure functions the agent can call in generated code. Must be plain functions — not callable objects, class methods, or lambdas. |
| `max_iters` | `int` | `5` | Maximum generate-execute cycles before stopping. |
| `interpreter` | `PythonInterpreter | None` | `None` | Sandboxed Python executor. Auto-instantiated (Deno-based) if omitted. |

## Inheritance

`CodeAct` inherits from both `ReAct` and `ProgramOfThought`, combining tool-based agentic reasoning with code generation.

## Key methods

All standard `dspy.Module` methods are available:

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `agent(**inputs)` | Run the agent on the given inputs. Returns a `Prediction`. |
| `batch` | `agent.batch(examples, ...)` | Run on multiple inputs in parallel. |
| `save` | `agent.save(path)` | Save optimized agent to JSON. |
| `load` | `agent.load(path)` | Load a previously saved agent. |
| `set_lm` | `agent.set_lm(lm)` | Override the LM for this module. |

## Tool requirements

1. Must be **plain functions** — not callable objects (`__call__`), not bound methods, not lambdas
2. Must have **type hints** and a **docstring** (the agent reads these to understand how to call them)
3. Cannot import external libraries (numpy, pandas, requests, etc.) in the **generated glue code** — but tool functions themselves run in your normal Python process and can use any library
4. All logic must be self-contained — no references to external classes or global state from generated code
5. Dependent functions must be explicitly passed as tools if the agent needs to call them directly

## Execution model

1. **Generate** — the LM writes a Python code snippet using the available tools as functions
2. **Execute** — code runs in a sandboxed Deno-based Python interpreter (not your system Python)
3. **Observe** — the agent sees stdout/stderr output
4. **Repeat** — if not done, writes more code incorporating previous results (up to `max_iters`)

### Sandbox constraints

- No filesystem, network, or environment variable access from generated code
- Standard library only in generated code — no pip packages
- Tool functions execute in your normal Python process with full privileges
- Variables persist across iterations within a single agent call

## PythonInterpreter

```python
dspy.PythonInterpreter()
```

The sandboxed code executor used by CodeAct. Created automatically if not provided. Uses Deno under the hood for isolation.
