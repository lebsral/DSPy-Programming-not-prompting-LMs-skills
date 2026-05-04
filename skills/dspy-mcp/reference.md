# MCP Integration API Reference

> Condensed from [dspy.ai/tutorials/agents/mcp](https://dspy.ai/tutorials/agents/mcp/) and [modelcontextprotocol.io](https://modelcontextprotocol.io/). Verify against upstream for latest.

## dspy.Tool.from_mcp_tool()

```python
dspy.Tool.from_mcp_tool(
    session,    # mcp.ClientSession -- active MCP session
    tool,       # mcp.types.Tool -- tool descriptor from list_tools()
) -> dspy.Tool
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session` | `mcp.ClientSession` | Active, initialized MCP client session |
| `tool` | `mcp.types.Tool` | Tool descriptor returned by `session.list_tools()` |

**Returns:** A `dspy.Tool` instance that can be passed to `dspy.ReAct` or `dspy.CodeAct`.

The returned tool is async -- it calls `session.call_tool()` internally. Use `acall()` on the agent.

## MCP Client Setup

### StdioServerParameters

```python
from mcp import StdioServerParameters

params = StdioServerParameters(
    command="npx",              # str -- executable to run
    args=["-y", "server-pkg"], # list[str] -- command arguments
    env=None,                  # dict | None -- environment variables
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `command` | `str` | required | Command to launch the MCP server |
| `args` | `list[str]` | `[]` | Arguments passed to the command |
| `env` | `dict` | `None` | Environment variables for the server process |

### stdio_client

```python
from mcp.client.stdio import stdio_client

async with stdio_client(server_params) as (read_stream, write_stream):
    # read_stream and write_stream are asyncio streams
    pass
```

Returns a tuple of `(read_stream, write_stream)` for creating a `ClientSession`.

### ClientSession

```python
from mcp import ClientSession

async with ClientSession(read_stream, write_stream) as session:
    await session.initialize()
    # Session is now ready
    pass
```

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `None` | Handshake with server (must call first) |
| `list_tools()` | `ListToolsResult` | Discover available tools |
| `call_tool(name, arguments)` | `CallToolResult` | Execute a tool by name |

### ListToolsResult

```python
result = await session.list_tools()
for tool in result.tools:
    print(tool.name, tool.description)
```

Each `tool` in `result.tools` has:
- `name` -- tool identifier
- `description` -- what the tool does
- `inputSchema` -- JSON Schema for the tool arguments

## Connection pattern

```python
import asyncio
import dspy
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(command="...", args=[...])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, t) for t in tools.tools
            ]

            agent = dspy.ReAct("task -> answer", tools=dspy_tools)
            result = await agent.acall(task="...")
            return result

asyncio.run(main())
```
