---
name: dspy-program-of-thought
description: Use when the task requires precise computation, math, or data manipulation — the LM writes Python code that executes in a sandbox instead of reasoning in natural language. Common scenarios: math word problems, data manipulation tasks, precise calculations the LLM gets wrong in natural language, statistical analysis, or any task where writing and executing code gives better results than reasoning in text. Related: ai-reasoning, dspy-chain-of-thought, dspy-codeact. Also: dspy.ProgramOfThought, LLM writes code to solve problem, code generation for computation, math with LLM via code, execute Python to get answer, when chain of thought gives wrong math, computation via code not text, precise calculations with LLM, data analysis by generating code, sandbox code execution, code-based reasoning, ProgramOfThought vs ChainOfThought, solve with code not words.
---

# Solve Problems by Generating and Executing Code with dspy.ProgramOfThought

Guide the user through using DSPy's `ProgramOfThought` module, which has the LM write Python code to solve a problem and then executes that code to produce the answer.

## Step 1: Understand the task

Before using ProgramOfThought, clarify:

1. **Does the task involve computation?** ProgramOfThought shines for math, data manipulation, date reasoning — anything where running code gives a more reliable answer than verbal reasoning. If the task is purely qualitative (classification, summarization), use ChainOfThought instead.
2. **Is Deno installed?** ProgramOfThought requires [Deno](https://docs.deno.com/runtime/getting_started/installation/) to run generated code in a WASM sandbox. Without it, the module will crash.
3. **What should the output look like?** A single number, a list, a formatted string? This determines your signature.

## What is ProgramOfThought

`dspy.ProgramOfThought` is a module that asks the LM to express its reasoning as executable Python code instead of natural language. The generated code runs in a sandboxed environment, and the execution result becomes the output.

This is fundamentally different from `ChainOfThought`:

- **ChainOfThought** -- the LM reasons in natural language, then produces an answer. Good for qualitative reasoning but prone to arithmetic and counting errors.
- **ProgramOfThought** -- the LM writes Python code that computes the answer. The code runs, and the result is exact. Good for anything where computation produces a more reliable answer than verbal reasoning.

Think of it as: the LM becomes a programmer that writes a small script to solve your problem, rather than trying to solve it in its head.

## When to use ProgramOfThought

Use `ProgramOfThought` when the task involves:

- **Math and arithmetic** -- compound interest, tax calculations, unit conversions, statistics
- **Counting and aggregation** -- "how many items match this condition", tallying, grouping
- **Data manipulation** -- sorting, filtering, transforming structured data
- **Date/time reasoning** -- days between dates, business day calculations, timezone math
- **Precise string operations** -- regex matching, character counting, formatting
- **Logic puzzles** -- constraint satisfaction, combinatorics, permutations

Do **not** use it when:

- The task is purely qualitative (summarization, classification, creative writing)
- No computation is needed -- use `dspy.Predict` or `dspy.ChainOfThought` instead
- You need tool use or external API calls -- use `dspy.ReAct` instead

## Setup

ProgramOfThought requires **Deno** for sandboxed code execution:

```bash
# macOS
brew install deno

# Linux / Windows
curl -fsSL https://deno.land/install.sh | sh
```

Verify: `deno --version`. The first run will download Pyodide (~30s).

## Basic usage

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Inline signature
solver = dspy.ProgramOfThought("question -> answer")

result = solver(question="What is 15% tip on a $84.50 dinner bill split 3 ways?")
print(result.answer)  # Precise computed result
```

`ProgramOfThought` works with any signature -- inline strings or class-based:

```python
class MathProblem(dspy.Signature):
    """Solve the given math problem by writing and executing Python code."""
    problem: str = dspy.InputField(desc="A math word problem")
    answer: float = dspy.OutputField(desc="The numerical answer")

solver = dspy.ProgramOfThought(MathProblem)
result = solver(problem="A store has a 20% off sale. An item costs $45. What is the sale price after 8% tax?")
print(result.answer)
```

## How it works

When you call a `ProgramOfThought` module, here is what happens:

1. **Code generation** -- the LM receives the signature and inputs, then generates Python code that computes the answer
2. **Sandbox execution** -- DSPy executes the generated code in a restricted Python environment
3. **Result extraction** -- the output of the code execution is captured and returned as the prediction

The LM does not directly produce the answer. It produces code, and the code produces the answer. This means arithmetic is done by Python (exact), not by the LM (approximate).

### What the sandbox provides

The generated code runs in a Deno/Pyodide WASM sandbox — isolated from your host filesystem, network, and environment. Pyodide includes Python's standard library (`math`, `datetime`, `collections`, `itertools`, `re`, `json`, `statistics`) plus some scientific packages. External packages not bundled with Pyodide are not available by default.

### Constructor

```python
dspy.ProgramOfThought(
    signature,                    # str | type[Signature] -- required
    max_iters=3,                  # int -- max code generation/retry attempts
    interpreter=None,             # PythonInterpreter | None -- custom sandbox config
)
```

### Retry on execution failure

If the generated code raises an exception, `ProgramOfThought` retries by generating new code. The LM sees the error traceback from the previous attempt, which helps it self-correct. Control retries with `max_iters` (default: 3).

## Using ProgramOfThought in a module

Wrap `ProgramOfThought` in a custom module to combine computation with other reasoning steps:

```python
import dspy


class FinancialAnalyzer(dspy.Module):
    def __init__(self):
        self.compute = dspy.ProgramOfThought("scenario, question -> result: float")
        self.explain = dspy.ChainOfThought("scenario, question, result -> explanation")

    def forward(self, scenario, question):
        # Step 1: Compute the exact numerical answer
        computed = self.compute(scenario=scenario, question=question)

        # Step 2: Explain the result in plain language
        explained = self.explain(
            scenario=scenario,
            question=question,
            result=str(computed.result),
        )

        return dspy.Prediction(
            result=computed.result,
            explanation=explained.explanation,
        )


lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

analyzer = FinancialAnalyzer()
result = analyzer(
    scenario="Revenue was $1.2M in Q1, $1.5M in Q2, $1.1M in Q3, $1.8M in Q4.",
    question="What is the year-over-year growth rate if last year's total was $4.8M?",
)
print(result.result)
print(result.explanation)
```

This pattern -- compute first, explain second -- gives you both precision and readability.

## Optimizing ProgramOfThought

`ProgramOfThought` modules work with DSPy optimizers just like any other module. The optimizer tunes the instructions and few-shot examples that guide code generation:

```python
def metric(example, prediction, trace=None):
    return abs(float(prediction.answer) - float(example.answer)) < 0.01

optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized_solver = optimizer.compile(solver, trainset=trainset)
```

Optimization improves the quality of the generated code by showing the LM examples of good code-generation patterns.

## Limitations

- **Requires Deno** -- the sandbox uses Deno/Pyodide WASM. Install Deno first or the module crashes.
- **Sandboxed by default** -- no host filesystem, network, or environment access. Use a custom `PythonInterpreter` with `enable_read_paths`, `enable_network_access`, etc. if needed.
- **Code generation cost** -- generating code takes more tokens than a direct answer. For trivial arithmetic (2 + 2), `ChainOfThought` is faster and cheaper.
- **LM capability matters** -- weaker models generate buggier code. Use a capable model (GPT-4o, Claude Sonnet, etc.) for complex computations.
- **First run is slow** -- Deno downloads and caches Pyodide (~30s) on the first execution. Subsequent runs are fast.

## ProgramOfThought vs ChainOfThought -- when to use which

| Scenario | Use | Why |
|----------|-----|-----|
| "What is 17% of $234.89?" | `ProgramOfThought` | Arithmetic -- code is exact |
| "Summarize this article" | `ChainOfThought` | No computation needed |
| "How many days between March 3 and November 17?" | `ProgramOfThought` | Date math -- code handles edge cases |
| "Classify this support ticket" | `ChainOfThought` | Qualitative judgment |
| "Given these 50 data points, what is the standard deviation?" | `ProgramOfThought` | Statistical computation |
| "Explain why this code has a bug" | `ChainOfThought` | Reasoning about code, not running code |
| "Sort these 20 items by priority score and return the top 5" | `ProgramOfThought` | Data manipulation |

Rule of thumb: if you would reach for a calculator or a spreadsheet, use `ProgramOfThought`.

## Gotchas

1. **Claude forgets Deno is required.** ProgramOfThought uses a Deno/Pyodide WASM sandbox — not a simple `exec()` call. Without Deno installed, the module crashes with a subprocess error. Always check `deno --version` or include Deno installation in setup instructions.
2. **Claude uses ProgramOfThought for tasks that do not need computation.** Classification, summarization, and extraction are qualitative — ProgramOfThought adds code-generation overhead with no benefit. Use ChainOfThought or Predict for non-computational tasks.
3. **Claude sets `max_iters=5` without justification.** The default is 3, which handles most retry scenarios. Only increase if you have evidence that code generation is failing due to complex logic. Higher values burn more tokens on retries.
4. **Claude ignores the `interpreter` parameter.** For tasks that need file access, network access, or environment variables, pass a custom `PythonInterpreter(enable_read_paths=[...], enable_network_access=[...])` instead of trying to work around sandbox restrictions.

## Additional resources

- [dspy.ProgramOfThought API docs](https://dspy.ai/api/modules/ProgramOfThought/)
- [Deno installation](https://docs.deno.com/runtime/getting_started/installation/)
- For API details, see [reference.md](reference.md)
- For worked examples, see [examples.md](examples.md)

## Cross-references

> Install any skill: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill <name>`

- **dspy.Predict** for simple direct LM calls -- see `/dspy-predict`
- **dspy.ChainOfThought** for natural language reasoning -- see `/dspy-chain-of-thought`
- **Building modules** that combine ProgramOfThought with other steps -- see `/dspy-modules`
- **Reasoning patterns** and when to add structured thinking -- see `/ai-reasoning`
- For worked examples, see [examples.md](examples.md)
- **Install `/ai-do` if you do not have it** — it routes any AI problem to the right skill and is the fastest way to work: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do`
