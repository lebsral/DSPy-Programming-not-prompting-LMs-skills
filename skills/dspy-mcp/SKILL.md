---
name: dspy-mcp
description: Use when you need to connect DSPy agents to external MCP tool servers — databases, file systems, APIs, or any MCP-compatible service. Common scenarios - wiring MCP tools into a ReAct or CodeAct agent, discovering tools from an MCP server at runtime, converting MCP tools to DSPy tools, or building agents that use tools hosted on remote servers. Related - dspy-tools, dspy-react, dspy-codeact, ai-taking-actions. Also used for dspy.Tool.from_mcp_tool, MCP with DSPy, connect DSPy to MCP server, use MCP tools in DSPy agent, model context protocol DSPy, DSPy agent with external tools via MCP, MCP tool integration, StdioServerParameters, ClientSession, stdio_client, mcp tool discovery, async MCP connection, acall with MCP tools.
---

# Connect DSPy Agents to MCP Tool Servers

Guide the user through connecting DSPy agents to MCP (Model Context Protocol) servers, discovering tools at runtime, and wiring them into ReAct or CodeAct agents.

## What is MCP integration in DSPy

DSPy can consume tools from any MCP-compatible server using `dspy.Tool.from_mcp_tool()`. This lets your agents use tools hosted externally -- databases, file systems, web APIs, or custom services -- without writing Python wrappers for each one. The MCP server handles execution; DSPy handles reasoning.

## When to use MCP

| Use MCP when... | Use plain dspy.Tool when... |
|-----------------|----------------------------|
| Tools are hosted on a separate process or server | You have a simple Python function |
| You want to reuse tools across multiple AI systems | The tool is specific to this DSPy program |
| Tools need isolation (file system access, DB connections) | No isolation needed |
| An MCP server already exists for your use case | You are building tools from scratch |
| You need runtime tool discovery (tools change dynamically) | Tool set is fixed at development time |

## Step 1: Install dependencies

```bash
pip install dspy mcp
```

## Step 2: Connect to an MCP server

MCP servers communicate via stdio. Use `StdioServerParameters` to configure the server command and `stdio_client` to establish the connection:

```python
import dspy
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Configure the MCP server to connect to
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"],
)
```

## Step 3: Discover and convert tools

Inside an async context, connect to the server, list available tools, and convert them to DSPy tools:

```python
import asyncio

async def build_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the MCP connection
            await session.initialize()

            # Discover available tools
            mcp_tools = await session.list_tools()
            print(f"Found {len(mcp_tools.tools)} tools")

            # Convert MCP tools to DSPy tools
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in mcp_tools.tools
            ]

            # Build a ReAct agent with the discovered tools
            agent = dspy.ReAct(
                "question -> answer",
                tools=dspy_tools,
            )

            # Use acall() for async tool execution
            result = await agent.acall(question="List all Python files in the project")
            print(result.answer)

asyncio.run(build_agent())
```

## Step 4: Wire into ReAct or CodeAct

The converted tools work exactly like native DSPy tools:

```python
async def run_agent_with_mcp():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in mcp_tools.tools
            ]

            # ReAct agent
            agent = dspy.ReAct(
                "task -> result",
                tools=dspy_tools,
                max_iters=10,
            )

            # Must use acall() because MCP tools are async
            result = await agent.acall(task="Find the largest file and summarize it")
            return result
```

**Important:** Use `await agent.acall()` (not `agent()`) because MCP tool calls are async operations.

## Step 5: Combining MCP tools with local tools

Mix MCP-discovered tools with locally-defined Python tools:

```python
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # use a safe evaluator in production

async def build_hybrid_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in mcp_tools.tools
            ]

            # Combine MCP tools with local tools
            all_tools = dspy_tools + [calculate]

            agent = dspy.ReAct("question -> answer", tools=all_tools)
            result = await agent.acall(question="How many lines in main.py divided by 3?")
            return result
```

## Step 6: Error handling

MCP connections can fail. Wrap the connection in proper error handling:

```python
from mcp import McpError

async def safe_agent_call(question: str):
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                mcp_tools = await session.list_tools()
                dspy_tools = [
                    dspy.Tool.from_mcp_tool(session, tool)
                    for tool in mcp_tools.tools
                ]

                agent = dspy.ReAct("question -> answer", tools=dspy_tools)
                return await agent.acall(question=question)

    except McpError as e:
        print(f"MCP server error: {e}")
        return None
    except ConnectionError:
        print("Could not connect to MCP server")
        return None
```

## Gotchas

1. **Claude forgets `acall()` and uses synchronous `agent()`.** MCP tools are async -- you must use `await agent.acall()` or you get runtime errors. DSPy cannot call async MCP tools from a sync context.
2. **The entire agent must run inside the `async with` block.** The MCP session is only valid inside the context manager. If you build the tools inside and call the agent outside, the session is closed and tool calls fail.
3. **Claude hardcodes tool lists instead of discovering them.** The point of MCP is runtime discovery. Always use `session.list_tools()` to get the current tool set -- MCP servers can add/remove tools dynamically.
4. **MCP server must be running before you connect.** `StdioServerParameters` launches the server process. If the command fails (e.g., `npx` not installed, package not found), you get a cryptic connection error. Test the server command manually first.
5. **Claude nests too many async context managers.** Keep the pattern flat -- one `stdio_client` context and one `ClientSession` context. Do not add extra wrappers.

## Additional resources

- [DSPy MCP tutorial](https://dspy.ai/tutorials/agents/mcp/)
- [MCP specification](https://modelcontextprotocol.io/)
- For API details, see [reference.md](reference.md)
- For worked examples, see [examples.md](examples.md)

## Cross-references

> Install any skill: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill <name>`

- **Defining tools** from Python functions -- see `/dspy-tools`
- **ReAct agents** that use tools -- see `/dspy-react`
- **CodeAct agents** for code execution -- see `/dspy-codeact`
- **Action-taking AI** from a problem-first perspective -- see `/ai-taking-actions`
- **Async execution** patterns -- see `/dspy-async`
- **Install `/ai-do` if you do not have it** -- it routes any AI problem to the right skill and is the fastest way to work: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do`
