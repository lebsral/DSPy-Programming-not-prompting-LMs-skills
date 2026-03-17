---
name: dspy-codeact
description: "Use DSPy's CodeAct module to build agents that write and execute code to complete tasks. Use when you want to use dspy.CodeAct, need an agent that solves problems by writing Python, or prefer code execution over tool-calling for complex data manipulation."
---

# Build Agents That Write and Execute Code with dspy.CodeAct

Guide the user through building DSPy agents that solve problems by generating and running Python code, rather than calling tools through a fixed interface.

## What is CodeAct

`dspy.CodeAct` is a DSPy module that creates agents which write Python code to accomplish tasks. Instead of selecting from a predefined set of tool calls (like ReAct), CodeAct generates executable code snippets that use provided tools as Python functions.

The agent works in a loop:

1. **Generate** -- the LM writes a Python code snippet using the available tools
2. **Execute** -- the code runs in a sandboxed interpreter
3. **Observe** -- the agent sees the output and decides whether the task is done
4. **Repeat** -- if not done, writes more code incorporating previous results

CodeAct inherits from both ReAct and ProgramOfThought, combining reasoning-and-acting with code generation.

## How CodeAct differs from ReAct

| | ReAct | CodeAct |
|---|---|---|
| **How it acts** | Calls tools by name with arguments | Writes Python code that calls tools |
| **Composition** | One tool call per step | Can chain multiple tool calls, use loops, variables, conditionals in a single step |
| **Data manipulation** | Limited to what tools return | Can transform, filter, aggregate data in code |
| **Best for** | Simple tool orchestration | Complex computation, data processing, multi-step logic |
| **Overhead** | Lower -- just picks a tool | Higher -- generates and executes code |

**Rule of thumb:** If your agent needs to do math, transform data, or chain several operations together, CodeAct is a better fit. If it just needs to look things up and combine results, ReAct is simpler.

## When to use CodeAct

Use CodeAct when the agent needs to:

- **Do computation** -- math, aggregations, statistics, string processing
- **Transform data** -- reshape, filter, combine results from multiple tool calls
- **Write multi-step logic** -- loops, conditionals, variable assignment between steps
- **Solve problems programmatically** -- tasks where the approach itself needs to be figured out

Avoid CodeAct when:

- Simple tool calls are sufficient (use ReAct instead)
- You need tight control over exactly which tools are called and in what order
- The execution environment cannot support sandboxed code execution
- You need to use external libraries like numpy or pandas inside tools (CodeAct tools cannot import external libraries)

## Basic usage

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define tools as pure functions with type hints and docstrings
def factorial(n: int) -> int:
    """Calculate the factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Create the CodeAct agent
agent = dspy.CodeAct(
    "question -> answer",
    tools=[factorial, fibonacci],
    max_iters=5,
)

result = agent(question="What is the factorial of 10 plus the 15th Fibonacci number?")
print(result.answer)
```

## Constructor parameters

```python
dspy.CodeAct(
    signature,          # str or dspy.Signature -- defines input/output fields
    tools,              # list[callable] -- pure functions the agent can call in code
    max_iters=5,        # int -- max generate-execute cycles before stopping
    interpreter=None,   # PythonInterpreter or None -- custom interpreter (creates one if None)
)
```

### Parameter details

- **`signature`** -- same as any DSPy module. Defines what the agent receives and what it should produce.
- **`tools`** -- a list of Python functions. Must be **pure functions** (not callable objects or class instances). All dependencies must be self-contained within each function -- tools cannot import external libraries or reference outside state.
- **`max_iters`** -- safety limit on how many code-generation-and-execution cycles the agent runs. Default is 5. Increase for complex multi-step tasks, decrease for simple ones.
- **`interpreter`** -- optionally pass a pre-configured `dspy.PythonInterpreter`. If `None`, CodeAct creates a fresh one. The interpreter runs code in a sandboxed Deno-based environment.

## Tool requirements

CodeAct is strict about what tools it accepts:

```python
# OK -- pure function
def search(query: str) -> str:
    """Search for information."""
    return "results..."

# OK -- function with multiple parameters
def lookup(table: str, key: str) -> str:
    """Look up a value in a table."""
    return "value..."

# NOT OK -- callable object (will be rejected)
class MyTool:
    def __call__(self, query: str) -> str:
        return "results..."

# NOT OK -- tool that imports external libraries
def analyze(data: str) -> str:
    """Analyze data."""
    import pandas as pd  # This will fail in the sandbox
    return str(pd.read_csv(data))
```

**Rules for tools:**

1. Must be plain functions (not callable objects, not class methods)
2. Must have type hints and a docstring
3. Cannot import external libraries (numpy, pandas, requests, etc.) inside the function body
4. All logic must be self-contained -- no references to external classes or global state
5. Dependencies must be explicitly passed as tools if the agent needs them

## Code execution environment

CodeAct runs generated code in a **sandboxed Deno-based Python interpreter**, not your system's Python. This means:

- **Isolation** -- code cannot access your filesystem, network, or environment variables
- **No external imports** -- standard library only within generated code; no pip packages
- **Tool access** -- the agent calls your tool functions, which execute in your normal Python environment. Only the glue code between tool calls runs in the sandbox.
- **State persistence** -- variables persist across iterations within a single agent call, so the agent can build up results incrementally

The sandbox provides security boundaries, but your tool functions themselves run in your normal Python process. If a tool accesses a database or API, that access is real.

## Safety considerations

1. **Tool functions are the trust boundary.** The sandbox constrains the generated glue code, but tool functions execute with full privileges. Keep tool functions minimal and validate their inputs.

2. **Set `max_iters` appropriately.** A runaway agent burns tokens. Start with `max_iters=5` and increase only if you see the agent running out of steps on legitimate tasks.

3. **Validate outputs.** Use `dspy.Assert` or `dspy.Suggest` in a wrapper module to check that the agent's answer meets your requirements.

4. **Don't expose dangerous operations as tools.** If you pass a tool that deletes files or sends emails, the agent can and will call it. Only expose tools you're comfortable with the agent using autonomously.

```python
class SafeCodeAgent(dspy.Module):
    def __init__(self, tools):
        self.agent = dspy.CodeAct(
            "task -> result",
            tools=tools,
            max_iters=5,
        )

    def forward(self, task):
        result = self.agent(task=task)
        dspy.Assert(
            len(result.result.strip()) > 0,
            "Agent must produce a non-empty result",
        )
        return result
```

## Using CodeAct inside a custom module

Wrap CodeAct in a `dspy.Module` to add pre-processing, post-processing, or combine it with other DSPy modules:

```python
class AnalysisAgent(dspy.Module):
    def __init__(self):
        self.planner = dspy.ChainOfThought("task -> plan")
        self.executor = dspy.CodeAct(
            "task, plan -> result",
            tools=[compute_stats, format_table],
            max_iters=8,
        )
        self.summarize = dspy.ChainOfThought("task, result -> summary")

    def forward(self, task):
        plan = self.planner(task=task)
        execution = self.executor(task=task, plan=plan.plan)
        return self.summarize(task=task, result=execution.result)
```

## Optimizing CodeAct agents

CodeAct agents are optimizable like any DSPy module:

```python
def task_metric(example, prediction, trace=None):
    return prediction.answer.strip() == example.answer.strip()

# BootstrapFewShot works well -- the agent learns from successful code traces
optimizer = dspy.BootstrapFewShot(metric=task_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(agent, trainset=trainset)

# MIPROv2 can also tune the instructions for code generation
optimizer = dspy.MIPROv2(metric=task_metric, auto="medium")
optimized = optimizer.compile(agent, trainset=trainset)

# Save and load
optimized.save("optimized_codeact.json")
```

## When to use CodeAct vs ReAct

| Scenario | Use |
|---|---|
| Look up facts and combine them | ReAct |
| Calculate, aggregate, or transform data | **CodeAct** |
| Simple API calls (weather, stock price) | ReAct |
| Multi-step data processing pipeline | **CodeAct** |
| Tasks where approach varies per input | **CodeAct** |
| Quick prototype with many tools | ReAct |
| Math-heavy or logic-heavy problems | **CodeAct** |
| Tasks needing external libraries in tools | ReAct (more flexible tool format) |

## Cross-references

- **ReAct** for tool-calling agents -- see `/ai-taking-actions`
- **Tools** and tool patterns -- see `/ai-taking-actions`
- **Multi-agent coordination** -- see `/ai-coordinating-agents`
- **Modules** for composing CodeAct with other modules -- see `/dspy-modules`
- **Signatures** for defining agent inputs/outputs -- see `/dspy-signatures`
- For worked examples, see [examples.md](examples.md)
