# dspy-tools -- Worked Examples

## Example 1: Wrapping custom functions as DSPy tools

Demonstrates wrapping functions with `dspy.Tool`, inspecting the inferred schema, overriding metadata, and using `arg_desc` for richer descriptions.

```python
import dspy


# --- Define functions ---

def search_docs(query: str, max_results: int = 5) -> str:
    """Search the documentation for articles matching the query."""
    # Simulated search
    docs = {
        "setup": "Install DSPy with pip install -U dspy. Configure an LM with dspy.configure(lm=...).",
        "signatures": "Signatures declare input/output behavior: 'question -> answer' or class-based.",
        "modules": "Modules wrap signatures with inference strategies: Predict, ChainOfThought, ReAct.",
        "tools": "Tools are Python functions with type hints and docstrings that agents can call.",
    }
    results = []
    for key, value in docs.items():
        if key in query.lower() or any(w in value.lower() for w in query.lower().split()):
            results.append(value)
    return "\n".join(results[:max_results]) if results else "No results found."


def get_user(user_id: int) -> str:
    """Look up a user by their numeric ID. Returns name and role."""
    users = {
        1: {"name": "Alice", "role": "admin"},
        2: {"name": "Bob", "role": "viewer"},
        3: {"name": "Carol", "role": "editor"},
    }
    user = users.get(user_id)
    if user:
        return f"Name: {user['name']}, Role: {user['role']}"
    return f"No user found with ID {user_id}."


# --- Implicit wrapping (pass functions directly) ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# DSPy wraps these automatically when passed to an agent
agent = dspy.ReAct("question -> answer", tools=[search_docs, get_user])
result = agent(question="What are DSPy signatures?")
print(result.answer)


# --- Explicit wrapping with dspy.Tool ---

# Inspect the auto-inferred schema
tool = dspy.Tool(search_docs)
print(f"Name: {tool.name}")        # "search_docs"
print(f"Desc: {tool.desc}")        # "Search the documentation for articles matching the query."
print(f"Args: {tool.args}")        # {'query': {'type': 'string'}, 'max_results': {'type': 'integer', 'default': 5}}

# Override name and description
tool = dspy.Tool(
    search_docs,
    name="docs_search",
    desc="Search DSPy documentation. Use this for any question about how DSPy works.",
)
print(f"Name: {tool.name}")        # "docs_search"

# Add per-argument descriptions
tool = dspy.Tool(
    search_docs,
    arg_desc={
        "query": "Keywords to search for -- use short phrases, not full questions",
        "max_results": "How many results to return (1-10)",
    },
)

# Use the explicitly wrapped tool in an agent
agent = dspy.ReAct("question -> answer", tools=[tool, get_user])
result = agent(question="Look up user 1 and tell me their role")
print(result.answer)


# --- Format as OpenAI-compatible function call schema ---

schema = tool.format_as_litellm_function_call()
print(schema)
# {
#   'type': 'function',
#   'function': {
#     'name': 'search_docs',
#     'description': '...',
#     'parameters': {'type': 'object', 'properties': {...}, 'required': ['query']}
#   }
# }
```

Key points:
- Pass functions directly to agents for the common case -- `dspy.Tool` is needed only when you want to override metadata
- `dspy.Tool` auto-infers name, description, and argument schemas from the function
- Use `arg_desc` to add richer per-argument descriptions for complex tools
- `format_as_litellm_function_call()` returns the OpenAI-compatible schema if you need to inspect it


## Example 2: Building a multi-tool agent

A research agent with three tools: web search, database lookup, and a calculator. Demonstrates tool selection, chaining, and wrapping in a custom module.

```python
import dspy
import json
from typing import Literal


# --- Tools ---

PRODUCTS_DB = {
    "P-100": {"name": "Widget Pro", "price": 29.99, "stock": 150, "category": "hardware"},
    "P-200": {"name": "Gadget Plus", "price": 49.99, "stock": 0, "category": "hardware"},
    "P-300": {"name": "SaaS Starter", "price": 9.99, "stock": None, "category": "software"},
    "P-400": {"name": "Enterprise Suite", "price": 199.99, "stock": None, "category": "software"},
}


def search_products(query: str) -> str:
    """Search the product catalog by name or category. Returns matching products with IDs."""
    query_lower = query.lower()
    matches = []
    for pid, product in PRODUCTS_DB.items():
        if query_lower in product["name"].lower() or query_lower in product["category"]:
            matches.append(f"{pid}: {product['name']} (${product['price']}, category: {product['category']})")
    if matches:
        return "\n".join(matches)
    return f"No products found matching '{query}'."


def get_product_details(product_id: str) -> str:
    """Get full details for a product by its ID (e.g., P-100). Returns price, stock, and category."""
    product = PRODUCTS_DB.get(product_id.upper())
    if product:
        stock_info = f"{product['stock']} units" if product["stock"] is not None else "unlimited (digital)"
        return json.dumps({
            "id": product_id.upper(),
            "name": product["name"],
            "price": product["price"],
            "stock": stock_info,
            "category": product["category"],
        })
    return f"No product found with ID {product_id}."


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, and parentheses."""
    try:
        # Only allow safe math operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters. Use only numbers and +, -, *, /, **."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# --- Signature ---

class ProductAnswer(dspy.Signature):
    """Answer questions about products using the available tools."""
    question: str = dspy.InputField(desc="A question about products, pricing, or availability")
    answer: str = dspy.OutputField(desc="A helpful answer with specific product details")
    has_data: bool = dspy.OutputField(desc="Whether the answer is based on real data from tools")


# --- Agent module ---

class ProductAgent(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct(
            ProductAnswer,
            tools=[search_products, get_product_details, calculate],
            max_iters=5,
        )

    def forward(self, question: str):
        return self.agent(question=question)


def product_data_reward(args, pred):
    """Reward answers grounded in real tool data."""
    if not pred.has_data:
        return 0.5
    return 1.0


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

agent = dspy.Refine(module=ProductAgent(), N=3, reward_fn=product_data_reward, threshold=1.0)

questions = [
    "What hardware products do you have and which is cheaper?",
    "Is the Gadget Plus in stock?",
    "If I buy 3 Widget Pros, how much will it cost?",
    "Compare the prices of all software products.",
]

for q in questions:
    print(f"\nQ: {q}")
    result = agent(question=q)
    print(f"A: {result.answer}")
    print(f"Based on data: {result.has_data}")


# --- Optimization ---

def product_metric(example, prediction, trace=None):
    has_data = prediction.has_data
    has_detail = len(prediction.answer.strip()) > 30
    return has_data + 0.5 * has_detail

# trainset = [
#     dspy.Example(question="What hardware products are available?").with_inputs("question"),
#     dspy.Example(question="How much does Widget Pro cost?").with_inputs("question"),
#     dspy.Example(question="Is Gadget Plus in stock?").with_inputs("question"),
# ]
# optimizer = dspy.BootstrapFewShot(metric=product_metric, max_bootstrapped_demos=3)
# optimized = optimizer.compile(agent, trainset=trainset)
# optimized.save("product_agent.json")
```

Key points:
- Three tools with different purposes -- the agent picks the right one per question
- The agent chains tools: search for products, then get details, then calculate totals
- `has_data` output field lets the module check whether the agent actually used tools
- `dspy.Refine` retries when `has_data` is false, nudging the agent to look up data rather than guessing
- The calculator uses a character allowlist for safety


## Example 3: PythonInterpreter for safe code execution

Demonstrates using `dspy.PythonInterpreter` directly and as part of a CodeAct agent, with host-side tools and selective sandbox permissions.

```python
import dspy
from dspy import PythonInterpreter


# --- Basic execution ---

# Run simple code in the sandbox
with PythonInterpreter() as interp:
    result = interp("print(sum(range(1, 101)))")
    print(f"Sum 1-100: {result}")  # "5050"

    # Variables persist across calls within the same session
    interp("data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]")
    result = interp("print(sorted(data))")
    print(f"Sorted: {result}")

    # Multi-line code works
    result = interp("""
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fibonacci(20))
""")
    print(f"Fib(20): {result}")


# --- With host-side tools ---

# Tools run in your normal Python process, not the sandbox.
# This lets tools use libraries, access databases, and call APIs.

def query_database(sql: str) -> str:
    """Execute a SQL query and return the results as a formatted string."""
    # Simulated database
    if "users" in sql.lower():
        return "id,name,role\n1,Alice,admin\n2,Bob,viewer\n3,Carol,editor"
    if "orders" in sql.lower():
        return "id,user_id,total\n101,1,29.99\n102,1,49.99\n103,2,9.99"
    return "No results."


def send_notification(user_id: int, message: str) -> str:
    """Send a notification to a user by ID."""
    print(f"[Notification] User {user_id}: {message}")
    return f"Notification sent to user {user_id}."


with PythonInterpreter(tools={
    "query_database": query_database,
    "send_notification": send_notification,
}) as interp:
    # The sandbox code calls host-side tools
    result = interp("""
users = query_database(sql="SELECT * FROM users")
print(users)
""")
    print(f"Query result:\n{result}")

    result = interp("""
response = send_notification(user_id=1, message="Your order shipped!")
print(response)
""")
    print(f"Notification: {result}")


# --- With selective permissions ---

# Allow reading local files and accessing a specific API
interp = PythonInterpreter(
    enable_read_paths=["./reports"],
    enable_network_access=["api.example.com"],
)

# Use in a CodeAct agent with custom permissions
with interp:
    result = interp("""
# This code runs in the sandbox but can read from ./reports
# and make HTTP requests to api.example.com
print("Sandbox with custom permissions")
""")


# --- CodeAct agent with PythonInterpreter ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

def fetch_stats(dataset: str) -> str:
    """Fetch summary statistics for a named dataset."""
    datasets = {
        "sales": "count=1000, mean=45.50, median=38.00, std=22.10, min=5.00, max=199.99",
        "users": "count=500, mean_age=34, median_age=31, active=420, inactive=80",
    }
    return datasets.get(dataset, f"Dataset '{dataset}' not found. Available: {list(datasets.keys())}")


# Let CodeAct create its own interpreter (default)
agent = dspy.CodeAct(
    "question -> answer",
    tools=[fetch_stats],
    max_iters=5,
)

result = agent(question="Get the sales stats and calculate what percentage of the max the mean represents")
print(f"Answer: {result.answer}")

# Or pass a custom interpreter with specific permissions
custom_interp = PythonInterpreter(
    enable_read_paths=["./data"],
)

agent_with_perms = dspy.CodeAct(
    "question -> answer",
    tools=[fetch_stats],
    interpreter=custom_interp,
    max_iters=5,
)

result = agent_with_perms(question="Fetch user stats and summarize them")
print(f"Answer: {result.answer}")
```

Key points:
- `PythonInterpreter` runs code in a sandboxed Deno + Pyodide environment -- no filesystem or network by default
- Use the context manager (`with`) to start and shut down the sandbox cleanly
- Variables persist across calls within a single session -- the agent can build up state
- Host-side tools run in your normal Python process so they can access databases, APIs, and libraries
- Use `enable_read_paths`, `enable_write_paths`, `enable_network_access`, and `enable_env_vars` to grant selective permissions
- Pass a custom `PythonInterpreter` to `dspy.CodeAct` when you need specific sandbox permissions
- Deno must be installed for `PythonInterpreter` to work
