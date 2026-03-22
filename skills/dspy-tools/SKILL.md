---
name: dspy-tools
description: "Use when you need to give DSPy agents tool-calling abilities — wrapping Python functions as tools, building tool-using pipelines, or setting up code execution environments."
---

# Give Agents Tool-Calling Abilities with dspy.Tool

Guide the user through wrapping functions as DSPy tools, using `dspy.PythonInterpreter` for sandboxed code execution, and wiring tools into agents with `dspy.ReAct` and `dspy.CodeAct`.

## What is dspy.Tool

`dspy.Tool` wraps a Python function so DSPy agents can call it. It automatically extracts the function's name, docstring, parameter types, and descriptions to build the tool schema that the LM sees.

You can pass plain functions directly to `dspy.ReAct` or `dspy.CodeAct` and DSPy wraps them for you. Use `dspy.Tool` explicitly when you need to override the inferred metadata, convert tools from LangChain or MCP, or inspect the generated schema.

```python
import dspy

# Implicit -- pass a function directly (DSPy wraps it automatically)
agent = dspy.ReAct("question -> answer", tools=[my_search_function])

# Explicit -- wrap it yourself for control over name, description, etc.
tool = dspy.Tool(my_search_function, name="search", desc="Search the knowledge base")
agent = dspy.ReAct("question -> answer", tools=[tool])
```

## dspy.Tool constructor

```python
dspy.Tool(
    func,                   # Callable -- the function to wrap
    name=None,              # str | None -- tool name (inferred from func.__name__ if omitted)
    desc=None,              # str | None -- description (inferred from docstring if omitted)
    args=None,              # dict | None -- argument JSON schemas (inferred from type hints)
    arg_types=None,         # dict | None -- argument type mappings (inferred from type hints)
    arg_desc=None,          # dict | None -- per-argument descriptions
)
```

All parameters except `func` are optional. DSPy infers them from the function signature and docstring. Override them when the inferred values are wrong or when you want a different name or description.

```python
def foo(x: int, y: str = "hello"):
    """Combine a number and a string."""
    return str(x) + y

tool = dspy.Tool(foo)
print(tool.name)    # "foo"
print(tool.desc)    # "Combine a number and a string."
print(tool.args)    # {'x': {'type': 'integer'}, 'y': {'type': 'string', 'default': 'hello'}}
```

## Wrapping functions as tools

The quality of your tools depends on three things: type hints, docstrings, and focused scope.

### Type hints tell the agent what to pass

DSPy reads type hints to build the JSON schema the LM uses for tool calling. Always annotate every parameter and the return type.

```python
# Good -- fully typed
def search(query: str, max_results: int = 5) -> str:
    """Search the knowledge base for documents matching the query."""
    ...

# Bad -- no type hints, agent won't know what to pass
def search(query, max_results=5):
    """Search the knowledge base."""
    ...
```

### Docstrings tell the agent when to use it

The docstring becomes the tool description. Write it from the perspective of someone deciding whether to call this tool.

```python
# Good -- explains what the tool does and when to use it
def lookup_user(email: str) -> str:
    """Look up a user account by email address. Returns name, plan, and join date."""
    ...

# Bad -- vague, doesn't help the agent decide
def lookup_user(email: str) -> str:
    """Get user info."""
    ...
```

### One tool, one job

Keep tools focused. A tool that searches and summarizes is harder for the agent to use than two separate tools.

```python
# Good -- single responsibility
def search(query: str) -> str:
    """Search for documents matching the query."""
    ...

def summarize(text: str) -> str:
    """Summarize a long piece of text into key points."""
    ...

# Bad -- does two things
def search_and_summarize(query: str) -> str:
    """Search for documents and summarize the results."""
    ...
```

### Return strings

Tool return values become the Observation the agent sees. Return a string (or something that converts to string cleanly).

```python
import json

def check_order(order_id: str) -> str:
    """Check the status of an order by its ID."""
    order = db.get_order(order_id)
    if order:
        return json.dumps(order)
    return f"No order found with ID {order_id}."
```

### Per-argument descriptions

For complex tools, add per-argument descriptions using `arg_desc`:

```python
tool = dspy.Tool(
    search,
    arg_desc={
        "query": "The search query -- use keywords, not full sentences",
        "max_results": "Maximum number of results to return (1-20)",
    },
)
```

## dspy.PythonInterpreter

`dspy.PythonInterpreter` runs Python code in a sandboxed Deno + Pyodide environment. By default, the sandbox has no filesystem, network, or environment access. You selectively enable what you need.

### Constructor

```python
dspy.PythonInterpreter(
    deno_command=None,          # list[str] | None -- custom Deno launch command
    enable_read_paths=None,     # list[str] | None -- paths the sandbox can read
    enable_write_paths=None,    # list[str] | None -- paths the sandbox can write
    enable_env_vars=None,       # list[str] | None -- environment variables to expose
    enable_network_access=None, # list[str] | None -- allowed network domains
    sync_files=True,            # bool -- sync file changes back to host
    tools=None,                 # dict[str, Callable] | None -- host-side tool functions
    output_fields=None,         # list[dict] | None -- output field definitions
)
```

**Prerequisites:** Deno must be installed. See https://docs.deno.com/runtime/getting_started/installation/

### Basic execution

```python
from dspy import PythonInterpreter

with PythonInterpreter() as interp:
    result = interp("print(1 + 2)")  # Returns "3"
```

### With host-side tools

Tools passed to `PythonInterpreter` run in your normal Python process (not the sandbox). The sandbox calls them via JSON-RPC. This lets tools access databases, APIs, and libraries that aren't available inside the sandbox.

```python
def fetch_price(ticker: str) -> str:
    """Fetch the current stock price for a ticker symbol."""
    import requests
    resp = requests.get(f"https://api.example.com/price/{ticker}")
    return resp.json()["price"]

with PythonInterpreter(tools={"fetch_price": fetch_price}) as interp:
    result = interp("price = fetch_price(ticker='AAPL')\nprint(f'Price: {price}')")
```

### Selective permissions

```python
# Allow reading from a data directory and accessing one API
interp = PythonInterpreter(
    enable_read_paths=["./data"],
    enable_network_access=["api.example.com"],
)
```

### Using PythonInterpreter with CodeAct

`dspy.CodeAct` creates a `PythonInterpreter` automatically if you don't pass one. Pass your own when you need custom permissions:

```python
import dspy

interp = dspy.PythonInterpreter(
    enable_read_paths=["./data"],
    enable_network_access=["api.example.com"],
)

agent = dspy.CodeAct(
    "question -> answer",
    tools=[search, calculate],
    interpreter=interp,
    max_iters=5,
)
```

## ToolCalls type

`dspy.ToolCalls` is a structured type representing tool-calling information -- tool names and their arguments in JSON format. Use it in signatures when you want the LM to output tool calls directly (without the ReAct loop).

```python
import dspy

class PlanActions(dspy.Signature):
    """Given a user request, plan which tools to call."""
    request: str = dspy.InputField()
    actions: dspy.ToolCalls = dspy.OutputField()

planner = dspy.Predict(PlanActions)
result = planner(request="Look up the weather in Paris and convert to Celsius")
print(result.actions)  # ToolCalls with name and args for each tool call
```

### Creating ToolCalls from dicts

```python
from dspy import ToolCalls

tool_calls = ToolCalls.from_dict_list([
    {"name": "search", "args": {"query": "weather in Paris"}},
    {"name": "convert_temp", "args": {"value": 72, "from_unit": "F", "to_unit": "C"}},
])
```

### ToolCalls with native LM tool calling

When the configured LM supports native tool calling (most modern LMs do), `ToolCalls` automatically adapts to use the LM's native function-calling API rather than generating JSON as text. This improves reliability.

## Using tools with ReAct

`dspy.ReAct` is the standard choice for tool-using agents. Pass tools as a list of functions or `dspy.Tool` objects:

```python
import dspy

def search(query: str) -> str:
    """Search for information about a topic."""
    return "DSPy is a framework for programming language models."

def calculate(expression: str) -> float:
    """Evaluate a math expression and return the result."""
    return eval(expression)

agent = dspy.ReAct(
    "question -> answer",
    tools=[search, calculate],
    max_iters=5,
)

result = agent(question="What is 2^10 plus the year DSPy was released?")
print(result.answer)
```

The agent decides which tools to call, in what order, and when to stop. See `/dspy-react` for the full guide.

## Using tools with CodeAct

`dspy.CodeAct` agents write Python code that calls your tools. Tools must be pure functions (not callable objects). The agent can chain calls, use loops, and manipulate data in code:

```python
import dspy

def factorial(n: int) -> int:
    """Calculate the factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

agent = dspy.CodeAct(
    "question -> answer",
    tools=[factorial],
    max_iters=5,
)

result = agent(question="What is factorial(10) + factorial(5)?")
print(result.answer)
```

CodeAct tools have stricter requirements than ReAct tools: they must be plain functions, cannot import external libraries, and cannot reference global state. See `/dspy-codeact` for the full guide.

## Converting LangChain tools

`dspy.Tool.from_langchain()` converts any LangChain tool to a DSPy tool:

```python
import dspy
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

search = dspy.Tool.from_langchain(DuckDuckGoSearchRun())
wikipedia = dspy.Tool.from_langchain(
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

agent = dspy.ReAct("question -> answer", tools=[search, wikipedia])
```

Install LangChain tools with `pip install langchain-community`.

## Converting MCP tools

`dspy.Tool.from_mcp_tool()` converts Model Context Protocol tools into DSPy tools. It preserves the tool's name, description, and input schema, and creates an async callable that invokes the tool through the MCP session.

Install the MCP extra:

```bash
pip install -U "dspy[mcp]"
```

### Remote server (Streamable HTTP)

```python
import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("https://mcp.example.com/sse") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        mcp_tools = await session.list_tools()
        tools = [dspy.Tool.from_mcp_tool(session, t) for t in mcp_tools.tools]

        agent = dspy.ReAct("question -> answer", tools=tools)
        result = await agent.aforward(question="What files are in the repo?")
```

### Local server (stdio)

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "./data"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        mcp_tools = await session.list_tools()
        tools = [dspy.Tool.from_mcp_tool(session, t) for t in mcp_tools.tools]

        agent = dspy.ReAct("question -> answer", tools=tools)
        result = await agent.aforward(question="List the files")
```

### Key details

- **DSPy doesn't manage the server connection** — you set up and tear down the `ClientSession` yourself using the `mcp` library.
- **Tools are async** — `from_mcp_tool` creates an async callable, so use `await agent.aforward()` or run inside an async context.
- **Schema is preserved** — the tool's name, description, and JSON schema for arguments are carried over from the MCP server.

## Tool type hints and docstrings checklist

Good tools make good agents. Before passing a tool to an agent, check:

| Check | Why it matters |
|-------|---------------|
| All parameters have type hints | DSPy generates the JSON schema from them |
| Return type is annotated | Helps the agent know what to expect |
| Docstring explains what the tool does | The agent reads this to decide when to call it |
| Docstring mentions required input format | e.g., "Pass repo as 'owner/name'" |
| Parameters have sensible defaults | Reduces the number of decisions the agent makes |
| Errors return useful strings, not exceptions | The agent sees the error as an Observation and can retry |

```python
# A well-documented tool
def get_github_repo(repo: str) -> str:
    """Get information about a GitHub repository.

    Pass the full repository name like 'stanfordnlp/dspy'.
    Returns name, description, stars, and language.
    """
    try:
        response = requests.get(f"https://api.github.com/repos/{repo}", timeout=10)
        response.raise_for_status()
        data = response.json()
        return f"Name: {data['full_name']}, Stars: {data['stargazers_count']}"
    except requests.RequestException as e:
        return f"Error: {str(e)}"
```

## Error handling in tools

Tools should catch exceptions and return error strings. When a tool returns an error string, the agent sees it as an Observation and can retry with different arguments or try a different tool.

```python
def search(query: str) -> str:
    """Search for information."""
    try:
        response = requests.get("https://api.example.com/search", params={"q": query}, timeout=5)
        response.raise_for_status()
        return response.json()["results"]
    except requests.Timeout:
        return "Error: Search timed out. Try a shorter or simpler query."
    except requests.HTTPError as e:
        return f"Error: Search failed with status {e.response.status_code}."
    except Exception as e:
        return f"Error: {str(e)}"
```

## Cross-references

- **ReAct agents** -- see `/dspy-react`
- **CodeAct agents** -- see `/dspy-codeact`
- **Action-taking AI** from a problem-first perspective -- see `/ai-taking-actions`
- **Signatures** for defining agent inputs/outputs -- see `/dspy-signatures`
- **Modules** for composing agents -- see `/dspy-modules`
- For worked examples, see [examples.md](examples.md)
