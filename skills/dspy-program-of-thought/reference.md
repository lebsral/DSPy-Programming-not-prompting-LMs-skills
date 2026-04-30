# ProgramOfThought API Reference

> Condensed from [dspy.ai/api/modules/ProgramOfThought](https://dspy.ai/api/modules/ProgramOfThought/). Verify against upstream for latest.

## Constructor

```python
dspy.ProgramOfThought(
    signature,          # str | type[Signature] -- required
    max_iters=3,        # int -- max code generation/retry attempts
    interpreter=None,   # PythonInterpreter | None -- custom sandbox config
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Defines input/output contract |
| `max_iters` | `int` | `3` | Max retry attempts for code generation and execution |
| `interpreter` | `PythonInterpreter \| None` | `None` | Custom sandbox configuration. Creates a new PythonInterpreter if not provided |

## Key Methods

| Method | Description |
|--------|-------------|
| `forward(**kwargs) -> dspy.Prediction` | Generate code, execute it, return result |
| `__call__(**kwargs) -> dspy.Prediction` | Entry point (calls forward with callbacks) |
| `acall(**kwargs) -> dspy.Prediction` | Async variant |

## How It Works

1. **Code generation** — uses ChainOfThought internally to generate Python code
2. **Execution** — runs generated code in Deno/Pyodide WASM sandbox
3. **Error recovery** — if code raises an exception, regenerates with traceback context (up to `max_iters`)
4. **Output generation** — produces final answer from execution result

## Inherited Module Methods

| Method | Description |
|--------|-------------|
| `batch(examples, num_threads, max_errors, ...)` | Parallel processing |
| `save(path)` | Persist learned state |
| `load(path)` | Load state into fresh instance |
| `set_lm(lm)` | Override LM for this module |
| `named_predictors()` | Access internal Predict instances |

## PythonInterpreter

```python
from dspy import PythonInterpreter

interp = PythonInterpreter(
    deno_command=None,              # list[str] | None -- custom Deno command
    enable_read_paths=None,         # list[PathLike | str] | None -- allowed read paths
    enable_write_paths=None,        # list[PathLike | str] | None -- allowed write paths
    enable_env_vars=None,           # list[str] | None -- allowed environment variables
    enable_network_access=None,     # list[str] | None -- allowed domains/IPs
    sync_files=True,                # bool -- sync sandbox changes back to host
    tools=None,                     # dict[str, Callable] | None -- host-side tools callable from sandbox
    output_fields=None,             # list[dict] | None -- typed output fields
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deno_command` | `list[str] \| None` | `None` | Custom command to launch Deno |
| `enable_read_paths` | `list[PathLike \| str] \| None` | `None` | Files/directories the sandbox can read |
| `enable_write_paths` | `list[PathLike \| str] \| None` | `None` | Files/directories the sandbox can write |
| `enable_env_vars` | `list[str] \| None` | `None` | Environment variables accessible in sandbox |
| `enable_network_access` | `list[str] \| None` | `None` | Domains/IPs the sandbox can reach |
| `sync_files` | `bool` | `True` | Sync file changes from sandbox back to host |
| `tools` | `dict[str, Callable] \| None` | `None` | Host-side functions callable from sandbox code |

### Usage with ProgramOfThought

```python
from dspy import PythonInterpreter

# Custom interpreter with file access
interp = PythonInterpreter(
    enable_read_paths=["./data/"],
    enable_network_access=["api.example.com"],
)

solver = dspy.ProgramOfThought("question -> answer", interpreter=interp)
```

### Context Manager

PythonInterpreter can be used as a context manager for standalone code execution:

```python
with PythonInterpreter() as interp:
    result = interp("print(1 + 2)")  # Returns "3"
```

## Prerequisites

**Deno must be installed** for ProgramOfThought to work:

```bash
# macOS
brew install deno

# Linux / Windows
curl -fsSL https://deno.land/install.sh | sh
```

The first execution downloads and caches Pyodide (~30s). Subsequent runs are fast.
