---
name: dspy-react
description: Use when the task requires calling external tools or APIs to gather information — multi-step tool use with reasoning, like searching databases, calling APIs, or combining multiple data sources. Common scenarios: building agents that search the web and synthesize results, multi-step information gathering from APIs, chatbots that look up data before answering, question answering that requires external knowledge, or any task needing interleaved reasoning and action. Related: ai-taking-actions, ai-searching-docs, dspy-codeact, dspy-tools. Also: dspy.ReAct, ReAct agent pattern, reasoning and acting loop, tool-using agent in DSPy, search then answer pattern, agent with tools, multi-step tool use, interleave thinking and acting, API-calling agent, agent that reasons about tool outputs, when to use ReAct vs CodeAct, build intelligent agent with DSPy.
---

# Build Tool-Using Agents with dspy.ReAct

Guide the user through building agents that reason step-by-step and call tools to accomplish tasks. `dspy.ReAct` implements the Reasoning-Action-Observation loop -- the agent thinks about what to do, calls a tool, observes the result, and repeats until it has an answer.

## What is ReAct

`dspy.ReAct` implements the Reasoning-Action-Observation loop as an optimizable module. The agent reasons about what to do, calls a tool, observes the result, and repeats until it has enough information to answer. DSPy handles the loop mechanics and prompt construction.

## When to use ReAct

| Use ReAct when... | Use something else when... |
|--------------------|----------------------------|
| The agent needs to call external tools (search, APIs, databases) | You just need input -> output with no tools (`dspy.ChainOfThought`) |
| Multi-step reasoning with real-world data | The task is purely computational / code-heavy (`dspy.CodeAct`) |
| You want the agent to decide which tools to call and in what order | You have a fixed pipeline of steps (`dspy.Module` with sub-modules) |
| You need an interpretable trace of reasoning + actions | You need agents coordinating with each other (see `/ai-coordinating-agents`) |

## Defining tools

Tools are Python functions with type hints and docstrings. DSPy uses the function signature and docstring to tell the agent what each tool does and how to call it.

```python
def search(query: str) -> str:
    """Search the web for information about a topic."""
    # Your search implementation here
    return "search results..."

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression and return the result."""
    return eval(expression)  # use a safe evaluator in production

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Your weather API call here
    return f"72°F and sunny in {city}"
```

**Tool requirements:**
- **Type hints** on all parameters and the return type -- DSPy uses these to generate the tool schema
- **Docstring** explaining what the tool does -- the agent reads this to decide when to use it
- **Return a string** (or something that converts to string) -- the result becomes the Observation

Keep tools focused on one thing. A `search` tool should search, not search-and-summarize.

## Basic ReAct agent

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

def search(query: str) -> str:
    """Search for information about a topic."""
    return "DSPy is a framework for programming language models."

agent = dspy.ReAct("question -> answer", tools=[search])
result = agent(question="What is DSPy?")
print(result.answer)
```

That's it. The agent will:
1. Read the question
2. Decide whether to call `search`
3. Use the search result to formulate an answer

## Constructor parameters

```python
dspy.ReAct(
    signature,      # str | Signature -- required, defines inputs/outputs
    tools,          # list[Callable | dspy.Tool] -- required, available tools
    max_iters=20,   # int -- max reasoning-action cycles
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Input/output contract (inline or class-based) |
| `tools` | `list[Callable \| dspy.Tool]` | required | Functions the agent can call. DSPy wraps plain functions automatically |
| `max_iters` | `int` | `20` | Max Thought-Action-Observation cycles before forcing an answer |

## The max_iters parameter

`max_iters` controls how many Thought-Action-Observation cycles the agent can take before it must produce an answer:

```python
# Simple lookup -- 1-2 tool calls usually enough
agent = dspy.ReAct("question -> answer", tools=[search], max_iters=3)

# Complex research -- may need many tool calls
agent = dspy.ReAct("question -> answer", tools=[search, lookup], max_iters=8)
```

Guidelines:
- **Default is 20** -- usually fine for most tasks
- **Set it lower** (2-5) for simple lookups where one or two tool calls suffice
- **Keep it at 20** for complex multi-step research tasks
- If the agent hits `max_iters` without finishing, it returns its best answer so far

## Multi-tool agents

Give the agent multiple tools and it decides which to use and when:

```python
import dspy

def search(query: str) -> str:
    """Search the web for general information."""
    return "search results..."

def lookup_user(email: str) -> str:
    """Look up a user account by email address."""
    return '{"name": "Alice", "plan": "pro", "status": "active"}'

def check_order(order_id: str) -> str:
    """Check the status of an order by its ID."""
    return '{"order_id": "12345", "status": "shipped", "eta": "March 20"}'

agent = dspy.ReAct(
    "question -> answer",
    tools=[search, lookup_user, check_order],
    max_iters=5,
)

# The agent picks the right tool based on the question
result = agent(question="What's the status of order 12345?")
print(result.answer)  # Uses check_order

result = agent(question="What plan is alice@example.com on?")
print(result.answer)  # Uses lookup_user
```

The agent can also chain tools -- call `lookup_user` first, then use the result to call `check_order`.

## Wrapping ReAct in a custom module

For production use, wrap `dspy.ReAct` inside a `dspy.Module` to add pre-processing, context, or post-processing:

```python
class SupportAgent(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct(
            "question, context -> answer",
            tools=[search, lookup_user, check_order],
            max_iters=6,
        )

    def forward(self, question):
        context = (
            "You are a customer support agent. "
            "Use lookup_user for account questions, "
            "check_order for order questions, "
            "and search for general questions."
        )
        result = self.agent(question=question, context=context)

        dspy.Suggest(
            len(result.answer) > 20,
            "Provide a detailed, helpful response",
        )

        return result
```

This pattern lets you:
- Pass extra context or instructions to the agent
- Add assertions and quality constraints
- Optimize the agent with DSPy optimizers (they tune the inner ReAct module)
- Save and load the optimized state

## Using class-based signatures

For agents with typed inputs and outputs, use a class-based signature:

```python
from typing import Literal

class ResearchTask(dspy.Signature):
    """Research a topic and provide a comprehensive answer with sources."""
    question: str = dspy.InputField(desc="The research question")
    answer: str = dspy.OutputField(desc="A thorough answer to the question")
    confidence: Literal["high", "medium", "low"] = dspy.OutputField(
        desc="Confidence level based on the sources found"
    )

agent = dspy.ReAct(ResearchTask, tools=[search], max_iters=5)
result = agent(question="What are the main features of DSPy?")
print(result.answer)
print(result.confidence)
```

## ReAct vs CodeAct

Both are agent modules, but they act differently:

| | ReAct | CodeAct |
|---|-------|---------|
| **How it acts** | Calls tools by name with arguments | Writes and executes Python code |
| **Best for** | API calls, database lookups, search | Data manipulation, calculations, file I/O |
| **Interpretability** | Clear tool call trace | Full code trace |
| **Tool style** | Function calls | Python expressions |
| **Use when** | You have specific tools to call | The task is better solved by writing code |

```python
# ReAct -- calls tools
agent = dspy.ReAct("question -> answer", tools=[search, calculate])

# CodeAct -- writes code
agent = dspy.CodeAct("question -> answer", tools=[search, calculate])
```

If you're unsure, start with ReAct. Switch to CodeAct if the agent needs to do math, string manipulation, or data transformations between tool calls.

## Error handling

Tools can fail. Handle errors inside your tools so the agent gets a useful message instead of a crash:

```python
import requests

def search(query: str) -> str:
    """Search the web for information."""
    try:
        response = requests.get(
            "https://api.example.com/search",
            params={"q": query},
            timeout=5,
        )
        response.raise_for_status()
        return response.json()["results"]
    except requests.Timeout:
        return "Error: Search timed out. Try a simpler query."
    except requests.HTTPError as e:
        return f"Error: Search failed with status {e.response.status_code}."
    except Exception as e:
        return f"Error: {str(e)}"
```

When a tool returns an error string, the agent sees it as an Observation and can decide to retry with different arguments, try a different tool, or give a partial answer.

For module-level error handling, wrap the agent call:

```python
class SafeAgent(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct("question -> answer", tools=[search], max_iters=5)
        self.fallback = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        try:
            return self.agent(question=question)
        except Exception:
            # Fall back to answering without tools
            return self.fallback(question=question)
```

## Optimizing ReAct agents

ReAct agents are optimizable like any DSPy module. The optimizer tunes the reasoning prompts so the agent makes better tool-calling decisions:

```python
def answer_metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# BootstrapFewShot for quick optimization
optimizer = dspy.BootstrapFewShot(metric=answer_metric, max_bootstrapped_demos=4)
optimized_agent = optimizer.compile(agent, trainset=trainset)

# MIPROv2 for better prompt optimization
optimizer = dspy.MIPROv2(metric=answer_metric, auto="medium")
optimized_agent = optimizer.compile(agent, trainset=trainset)

# Save and load
optimized_agent.save("optimized_agent.json")
```

## Debugging

Inspect what the agent is doing:

```python
# See the last few LM calls (thoughts, tool calls, observations)
dspy.inspect_history(n=5)

# Print the module structure
print(agent)
```

`inspect_history` shows you the full Thought-Action-Observation trace, which is invaluable for understanding why the agent called certain tools or gave a wrong answer.

## Gotchas

1. **Claude sets `max_iters=5` but the default is 20.** Claude habitually passes `max_iters=5` which cuts off complex multi-step tasks too early. The actual default is 20. Only lower it when you want to constrain simple tasks (2-3 for lookups). For complex research, the default of 20 is appropriate.
2. **Tool errors are passed back as observations** -- make your error messages informative so the agent can recover (e.g., "No user found with that email" not just "Error").
3. **ReAct is slow by design** -- each iteration is a separate LM call. Use `CodeAct` for computation-heavy tasks where the agent can do work in code between tool calls.
4. **Tool function docstrings become part of the prompt** -- write clear, concise docstrings. Verbose docstrings waste tokens every iteration.
5. **Claude ignores the `trajectory` in the return value.** `ReAct` returns a `dspy.Prediction` with a `.trajectory` dict containing the full Thought-Action-Observation trace. Access `result.trajectory` to log or debug the agent's reasoning path. Claude often discards this and only uses `result.answer`.

## Additional resources

- [dspy.ReAct API docs](https://dspy.ai/api/modules/ReAct/)
- For API details, see [reference.md](reference.md)
- For worked examples, see [examples.md](examples.md)

## Cross-references

- **Defining tools** in detail -- see `/dspy-tools`
- **CodeAct** for code-based agents -- see `/dspy-codeact`
- **Building custom modules** to wrap ReAct -- see `/dspy-modules`
- **Action-taking AI** from a problem-first perspective -- see `/ai-taking-actions`
- **Multi-agent coordination** -- see `/ai-coordinating-agents`
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
