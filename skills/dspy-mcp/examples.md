# MCP Integration Examples

## Example 1: File system agent

An agent that can read, write, and search files via the MCP filesystem server:

```python
import asyncio
import dspy
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"],
)


async def file_agent(task: str):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in mcp_tools.tools
            ]

            agent = dspy.ReAct(
                "task -> result",
                tools=dspy_tools,
                max_iters=10,
            )

            result = await agent.acall(task=task)
            return result.result


# Usage
answer = asyncio.run(file_agent("Find all TODO comments in Python files"))
print(answer)
```

## Example 2: Database query agent with MCP

Connecting to a PostgreSQL MCP server for natural language database queries:

```python
import asyncio
import dspy
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# PostgreSQL MCP server
db_server = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
)


async def query_database(question: str):
    async with stdio_client(db_server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in mcp_tools.tools
            ]

            agent = dspy.ReAct(
                "question -> sql_query, answer",
                tools=dspy_tools,
            )

            result = await agent.acall(question=question)
            return {"query": result.sql_query, "answer": result.answer}


# Usage
result = asyncio.run(query_database("How many orders were placed last month?"))
print(f"SQL: {result['query']}")
print(f"Answer: {result['answer']}")
```

## Example 3: Multi-server agent

An agent that connects to multiple MCP servers simultaneously:

```python
import asyncio
import dspy
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

fs_server = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"],
)

web_server = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-fetch"],
)


async def multi_server_agent(task: str):
    # Connect to both servers simultaneously
    async with stdio_client(fs_server) as (fs_read, fs_write):
        async with ClientSession(fs_read, fs_write) as fs_session:
            await fs_session.initialize()

            async with stdio_client(web_server) as (web_read, web_write):
                async with ClientSession(web_read, web_write) as web_session:
                    await web_session.initialize()

                    # Gather tools from both servers
                    fs_tools = await fs_session.list_tools()
                    web_tools = await web_session.list_tools()

                    all_dspy_tools = [
                        dspy.Tool.from_mcp_tool(fs_session, t)
                        for t in fs_tools.tools
                    ] + [
                        dspy.Tool.from_mcp_tool(web_session, t)
                        for t in web_tools.tools
                    ]

                    agent = dspy.ReAct(
                        "task -> result",
                        tools=all_dspy_tools,
                        max_iters=15,
                    )

                    result = await agent.acall(task=task)
                    return result.result


answer = asyncio.run(multi_server_agent(
    "Fetch the DSPy changelog from GitHub and save a summary to docs/changelog-summary.md"
))
```
