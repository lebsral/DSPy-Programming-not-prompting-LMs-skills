---
name: dspy-modules
description: "Use DSPy's Module base class to build composable AI programs. Use when you want to create a dspy.Module, implement forward(), compose sub-modules, build multi-step pipelines, or structure your DSPy program as reusable components."
---

# Build Composable AI Programs with dspy.Module

Guide the user through structuring DSPy programs as reusable, composable modules. A `dspy.Module` is the building block for all DSPy programs -- like PyTorch's `nn.Module` but for language model pipelines.

## What is dspy.Module

Every DSPy program is a module. Modules:

- **Contain sub-modules** -- `dspy.Predict`, `dspy.ChainOfThought`, `dspy.Retrieve`, or other custom modules
- **Define control flow** in `forward()` -- Python logic that wires sub-modules together
- **Are optimizable** -- DSPy optimizers discover and tune all `Predict`-based sub-modules automatically
- **Are composable** -- modules nest inside other modules, just like functions call functions
- **Track state** -- learned prompts, few-shot demos, and instructions can be saved/loaded

If you're building anything beyond a single `dspy.Predict` call, wrap it in a module.

## Basic module structure

Every module has two methods:

```python
import dspy

class MyProgram(dspy.Module):
    def __init__(self):
        # 1. Declare sub-modules here
        self.step1 = dspy.ChainOfThought("input -> intermediate")
        self.step2 = dspy.Predict("intermediate -> output")

    def forward(self, input):
        # 2. Wire them together with Python logic
        mid = self.step1(input=input)
        return self.step2(intermediate=mid.intermediate)
```

Rules:
- **`__init__`** declares sub-modules as `self.` attributes so DSPy's optimizers can find them
- **`forward()`** defines the execution logic -- call sub-modules, use conditionals, loops, whatever Python you need
- Call the module like a function: `result = MyProgram()(input="hello")`

## Simple module example -- RAG

A retrieval-augmented generation module that fetches context before answering:

```python
import dspy

class RAG(dspy.Module):
    def __init__(self, k=3):
        self.retrieve = dspy.Retrieve(k=k)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Usage
rag = RAG(k=5)
result = rag(question="What is retrieval-augmented generation?")
print(result.answer)
```

## Composing modules -- nesting modules within modules

Modules are composable. A module can use other custom modules as sub-modules:

```python
class Summarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought("text -> summary")

    def forward(self, text):
        return self.summarize(text=text)


class AnalyzeAndSummarize(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> category")
        self.summarizer = Summarizer()  # nested custom module
        self.respond = dspy.ChainOfThought("category, summary -> response")

    def forward(self, text):
        category = self.classify(text=text).category
        summary = self.summarizer(text=text).summary
        return self.respond(category=category, summary=summary)
```

DSPy optimizers traverse the full module tree. When you optimize `AnalyzeAndSummarize`, the inner `Summarizer`'s prompts get optimized too.

## Printing module structure

Use `print()` to inspect all sub-modules and their signatures:

```python
pipeline = AnalyzeAndSummarize()
print(pipeline)
```

Output shows the module tree:

```
AnalyzeAndSummarize(
  classify = Predict(text -> category)
  summarizer = Summarizer(
    summarize = ChainOfThought(text -> summary)
  )
  respond = ChainOfThought(category, summary -> response)
)
```

This is useful for verifying your module hierarchy and debugging which sub-modules exist.

## Module state -- save and load

After optimization, save the learned state (few-shot demos, instructions) and reload it later:

```python
# Save after optimization
optimized_program = optimizer.compile(my_program, trainset=trainset)
optimized_program.save("my_program.json")

# Load into a fresh instance
loaded = MyProgram()
loaded.load("my_program.json")

# Use the loaded program -- it has the optimized prompts
result = loaded(question="What is DSPy?")
```

What gets saved:
- Few-shot demonstrations discovered by optimizers
- Optimized instructions (from MIPROv2, GEPA)
- Any state that DSPy's `Predict` modules track

What does **not** get saved:
- Python logic in `forward()` -- that's your code
- Model weights (unless you used `BootstrapFinetune`)
- The LM configuration -- you must call `dspy.configure()` before loading

## Assertions in modules

Use `dspy.Assert` and `dspy.Suggest` inside `forward()` to enforce constraints on outputs:

```python
class SafeQA(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        result = self.generate(question=question)

        # Hard constraint -- raises an error and triggers retry
        dspy.Assert(
            result.answer != "I don't know",
            "Must provide a substantive answer",
        )

        # Soft constraint -- suggests improvement but doesn't fail
        dspy.Suggest(
            len(result.answer.split()) >= 10,
            "Answer should be at least 10 words for completeness",
        )

        return result
```

- **`dspy.Assert(condition, message)`** -- hard constraint. If the condition is `False`, DSPy retries the prediction (up to a limit). Use for requirements that must be met.
- **`dspy.Suggest(condition, message)`** -- soft constraint. Adds feedback to the prompt on retry but won't raise an error. Use for quality preferences.

Assertions work with optimizers -- the optimizer learns to avoid triggering them.

## Common patterns

### Conditional logic in forward()

Route to different sub-modules based on intermediate results:

```python
class ConditionalPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> category")
        self.simple_handler = dspy.Predict("text -> response")
        self.complex_handler = dspy.ChainOfThought("text -> response")

    def forward(self, text):
        category = self.classify(text=text).category

        if category in ("simple", "faq"):
            return self.simple_handler(text=text)
        else:
            return self.complex_handler(text=text)
```

### Loops in forward()

Process a list of items or iterate until a condition is met:

```python
class BatchProcessor(dspy.Module):
    def __init__(self):
        self.process_item = dspy.ChainOfThought("item -> result")

    def forward(self, items: list[str]):
        results = []
        for item in items:
            result = self.process_item(item=item)
            results.append(result.result)
        return dspy.Prediction(results=results)
```

### Iterative refinement

Keep improving until quality is sufficient:

```python
class Refiner(dspy.Module):
    def __init__(self, max_rounds=3):
        self.draft = dspy.ChainOfThought("task -> output")
        self.critique = dspy.ChainOfThought("task, output -> feedback, is_good: bool")
        self.revise = dspy.ChainOfThought("task, output, feedback -> output")
        self.max_rounds = max_rounds

    def forward(self, task):
        result = self.draft(task=task)

        for _ in range(self.max_rounds):
            check = self.critique(task=task, output=result.output)
            if check.is_good:
                break
            result = self.revise(
                task=task,
                output=result.output,
                feedback=check.feedback,
            )

        return result
```

### Error handling

Wrap sub-module calls to handle failures gracefully:

```python
class ResilientModule(dspy.Module):
    def __init__(self):
        self.primary = dspy.ChainOfThought("question -> answer")
        self.fallback = dspy.Predict("question -> answer")

    def forward(self, question):
        try:
            return self.primary(question=question)
        except Exception:
            return self.fallback(question=question)
```

### Returning custom predictions

Use `dspy.Prediction` to return structured results from `forward()`:

```python
class MultiOutput(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought("text -> sentiment, topics: list[str]")
        self.summarize = dspy.ChainOfThought("text -> summary")

    def forward(self, text):
        analysis = self.analyze(text=text)
        summary = self.summarize(text=text)

        return dspy.Prediction(
            sentiment=analysis.sentiment,
            topics=analysis.topics,
            summary=summary.summary,
        )
```

### Setting different LMs per sub-module

Assign cheaper models to simpler steps:

```python
expensive_lm = dspy.LM("openai/gpt-4o")
cheap_lm = dspy.LM("openai/gpt-4o-mini")

pipeline = MyProgram()
pipeline.classify.set_lm(cheap_lm)
pipeline.generate.set_lm(expensive_lm)
```

## Cross-references

- **Signatures** define inputs and outputs for each sub-module -- see `/dspy-signatures`
- **Predict** is the simplest sub-module for direct LM calls -- see `/dspy-predict`
- **ChainOfThought** adds step-by-step reasoning -- see `/dspy-chain-of-thought`
- **Multi-step pipelines** with real-world patterns -- see `/ai-building-pipelines`
- **Optimizing modules** to improve accuracy -- see `/ai-improving-accuracy`
- For worked examples, see [examples.md](examples.md)
