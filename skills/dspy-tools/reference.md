# Tool API Reference

> Condensed from [dspy.ai/api/primitives/Tool](https://dspy.ai/api/primitives/Tool). Verify against upstream for latest.

## dspy.Tool

```python
dspy.Tool(
    func,             # Callable -- required
    name=None,        # str | None -- inferred from func.__name__
    desc=None,        # str | None -- inferred from docstring
    args=None,        # dict | None -- JSON schemas, inferred from type hints
    arg_types=None,   # dict | None -- type mappings, inferred from type hints
    arg_desc=None,    # dict | None -- per-argument descriptions
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | required | Function to wrap as a tool |
| `name` | `str \| None` | `None` | Tool name (auto-inferred) |
| `desc` | `str \| None` | `None` | Tool description (auto-inferred from docstring) |
| `args` | `dict \| None` | `None` | Argument JSON schemas (auto-inferred from type hints) |
| `arg_types` | `dict \| None` | `None` | Argument type mappings (auto-inferred) |
| `arg_desc` | `dict \| None` | `None` | Per-argument descriptions |

### Class Methods

```python
# Convert a LangChain tool
dspy.Tool.from_langchain(tool: BaseTool) -> Tool

# Convert an MCP tool (async)
dspy.Tool.from_mcp_tool(session: mcp.ClientSession, tool: mcp.types.Tool) -> Tool
```

## dspy.PythonInterpreter

```python
dspy.PythonInterpreter(
    deno_command=None,            # list[str] | None -- custom Deno launch command
    enable_read_paths=None,       # list[str] | None -- readable paths
    enable_write_paths=None,      # list[str] | None -- writable paths
    enable_env_vars=None,         # list[str] | None -- exposed env vars
    enable_network_access=None,   # list[str] | None -- allowed network domains
    sync_files=True,              # sync file changes back to host
    tools=None,                   # dict[str, Callable] | None -- host-side tools
    output_fields=None,           # list[dict] | None -- output field definitions
)
```

Use as a context manager: `with PythonInterpreter() as interp:`

Requires [Deno](https://docs.deno.com/runtime/getting_started/installation/).

## dspy.ToolCalls

Structured type for tool-calling output. Use as an `OutputField` type.

```python
# Create from dicts
ToolCalls.from_dict_list([{"name": "search", "args": {"query": "..."}}])
```

Adapts to native LM function-calling API when supported.
