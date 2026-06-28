# Tool API Reference

> Condensed from [dspy.Tool](https://dspy.ai/api/primitives/Tool/), [dspy.PythonInterpreter](https://dspy.ai/api/tools/PythonInterpreter/), and [dspy.ToolCalls](https://dspy.ai/api/primitives/ToolCalls/). Verify against upstream for latest.

## dspy.Tool

```python
dspy.Tool(
    func,             # Callable -- required
    name=None,        # str | None -- inferred from func.__name__
    desc=None,        # str | None -- inferred from docstring
    args=None,        # dict | None -- JSON schemas, inferred from type hints
    arg_types=None,   # dict | None -- type mappings, inferred from type hints
    arg_desc=None,    # dict[str, str] | None -- per-argument descriptions
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | required | Function to wrap as a tool |
| `name` | `str \| None` | `None` | Tool name (auto-inferred from `func.__name__`) |
| `desc` | `str \| None` | `None` | Tool description (auto-inferred from docstring) |
| `args` | `dict \| None` | `None` | Argument JSON schemas (auto-inferred from type hints) |
| `arg_types` | `dict \| None` | `None` | Argument type mappings (auto-inferred) |
| `arg_desc` | `dict[str, str] \| None` | `None` | Per-argument descriptions |

### Class Methods

```python
# Convert a LangChain tool
dspy.Tool.from_langchain(tool: BaseTool) -> Tool

# Convert an MCP tool (returns async callable)
dspy.Tool.from_mcp_tool(session: mcp.ClientSession, tool: mcp.types.Tool) -> Tool
```

### Instance Methods

```python
tool(**kwargs)                           # Synchronous call
await tool.acall(**kwargs)               # Async call
tool.format()                            # String representation
tool.format_as_litellm_function_call()  # OpenAI-compatible function schema dict
```

## dspy.PythonInterpreter

```python
dspy.PythonInterpreter(
    deno_command=None,            # list[str] | None -- custom Deno launch command
    enable_read_paths=None,       # list[PathLike | str] | None -- readable paths
    enable_write_paths=None,      # list[PathLike | str] | None -- writable paths
    enable_env_vars=None,         # list[str] | None -- exposed env vars
    enable_network_access=None,   # list[str] | None -- allowed network domains
    sync_files=True,              # bool -- sync file changes back to host
    tools=None,                   # dict[str, Callable[..., str]] | None -- host-side tools
    output_fields=None,           # list[dict] | None -- output field definitions
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deno_command` | `list[str] \| None` | `None` | Custom Deno launch command |
| `enable_read_paths` | `list[PathLike \| str] \| None` | `None` | Paths the sandbox can read |
| `enable_write_paths` | `list[PathLike \| str] \| None` | `None` | Paths the sandbox can write |
| `enable_env_vars` | `list[str] \| None` | `None` | Environment variables to expose |
| `enable_network_access` | `list[str] \| None` | `None` | Allowed network domains |
| `sync_files` | `bool` | `True` | Sync sandbox file changes back to host |
| `tools` | `dict[str, Callable[..., str]] \| None` | `None` | Host-side tool functions (run outside sandbox, must return str) |
| `output_fields` | `list[dict] \| None` | `None` | Output field definitions for typed SUBMIT signatures |

### Key Methods

```python
interp(code: str, variables: dict | None = None) -> Any   # __call__ — execute code string
interp.execute(code: str, variables: dict | None = None) -> Any  # alias for __call__
interp.start() -> None    # Initialize Deno/Pyodide sandbox (called automatically by __enter__)
interp.shutdown() -> None # Terminate subprocess (called automatically by __exit__)
```

Use as a context manager: `with PythonInterpreter() as interp:`

Requires [Deno](https://docs.deno.com/runtime/getting_started/installation/).

## dspy.ToolCalls

> **Note:** `dspy.ToolCalls` is in DSPy 3.3.0 (beta). Not yet available in the stable 3.2.1 release.

Structured type for tool-calling output. Use as an `OutputField` type in signatures when you want the LM to plan tool calls directly.

```python
# Create from dicts
ToolCalls.from_dict_list([{"name": "search", "args": {"query": "..."}}])
```

Adapts to native LM function-calling API when the configured LM supports it.
