---
name: dspy-program-of-thought
description: "Use DSPy's ProgramOfThought module to solve problems by generating and executing code. Use when you want to use dspy.ProgramOfThought, need accurate math or computation, want the LM to write Python code that runs in a sandbox, or need data manipulation and counting tasks."
---

# Solve Problems by Generating and Executing Code with dspy.ProgramOfThought

Guide the user through using DSPy's `ProgramOfThought` module, which has the LM write Python code to solve a problem and then executes that code to produce the answer.

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

## Basic usage

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
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

The generated code runs with access to Python's standard library. This includes `math`, `datetime`, `collections`, `itertools`, `re`, `json`, `statistics`, and other built-in modules. External packages like `numpy` or `pandas` are not available unless they are installed in the environment.

### Retry on execution failure

If the generated code raises an exception, `ProgramOfThought` can retry by generating new code. You can control the number of retries:

```python
solver = dspy.ProgramOfThought("question -> answer", max_iters=5)
```

The default is 3 iterations. On each retry, the LM sees the error traceback from the previous attempt, which helps it self-correct.

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


lm = dspy.LM("openai/gpt-4o-mini")
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

- **Standard library only** -- generated code cannot import packages that are not installed. If your task needs `pandas` or `numpy`, ensure they are in the environment.
- **No side effects** -- the sandbox restricts file I/O, network access, and other side effects. The code is meant to compute a value, not interact with the world.
- **Code generation cost** -- generating code takes more tokens than a direct answer. For trivial arithmetic (2 + 2), `ChainOfThought` is faster and cheaper.
- **LM capability matters** -- weaker models generate buggier code. Use a capable model (GPT-4o, Claude Sonnet, etc.) for complex computations.
- **Output is the code's return value** -- the generated code must produce a result that maps to your signature's output fields.

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

## Cross-references

- **dspy.Predict** for simple direct LM calls -- see `/dspy-predict`
- **dspy.ChainOfThought** for natural language reasoning -- see `/dspy-chain-of-thought`
- **Building modules** that combine ProgramOfThought with other steps -- see `/dspy-modules`
- **Reasoning patterns** and when to add structured thinking -- see `/ai-reasoning`
- For worked examples, see [examples.md](examples.md)
