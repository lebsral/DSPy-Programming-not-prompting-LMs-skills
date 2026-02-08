# DSPy Framework Quick Reference

> Condensed API reference for DSPy. For full docs, see [dspy.ai](https://dspy.ai/).

## Setup

```bash
pip install -U dspy
```

```python
import dspy

# Configure your LM (any LiteLLM-supported provider)
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

## Signatures

Signatures declare input/output behavior as typed specs:

```python
# Inline signature
"question -> answer"

# Class-based signature (recommended for complex tasks)
class Classify(dspy.Signature):
    """Classify the sentiment of a review."""
    text: str = dspy.InputField()
    label: Literal["positive", "negative", "neutral"] = dspy.OutputField()

# Adding type constraints
"question -> answer: float"
"text -> labels: list[str]"
```

### Field options

```python
dspy.InputField(desc="description", prefix="Question:")
dspy.OutputField(desc="description", type_=Literal["yes", "no"])
```

## Modules

Modules wrap signatures with inference strategies:

| Module | Purpose | When to use |
|--------|---------|-------------|
| `dspy.Predict` | Direct LM call | Simple input -> output |
| `dspy.ChainOfThought` | Adds step-by-step reasoning | Most tasks (default choice) |
| `dspy.ProgramOfThought` | Generates + executes code | Math, computation |
| `dspy.ReAct` | Reasoning + tool use loop | Agents with tools |
| `dspy.CodeAct` | Code-based action agent | Agents that write code to act |
| `dspy.MultiChainComparison` | Runs multiple chains, picks best | When quality matters more than speed |

### Usage

```python
# Simple prediction
classify = dspy.Predict("text -> label")
result = classify(text="Great product!")
print(result.label)

# Chain of thought
classify = dspy.ChainOfThought("text -> label")
result = classify(text="Great product!")
print(result.reasoning)  # intermediate reasoning
print(result.label)

# ReAct agent with tools
def search(query: str) -> str:
    """Search for information."""
    ...

agent = dspy.ReAct("question -> answer", tools=[search])
result = agent(question="What is DSPy?")
```

### Custom modules

```python
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)
```

## Optimizers

Optimizers improve your program's prompts or weights:

| Optimizer | What it tunes | Data needed | Best for |
|-----------|--------------|-------------|----------|
| `dspy.BootstrapFewShot` | Few-shot examples | ~50 examples | Quick start, first optimization |
| `dspy.BootstrapFewShotWithRandomSearch` | Few-shot examples | ~200 examples | Better than BootstrapFewShot |
| `dspy.MIPROv2` | Instructions + few-shot | ~200 examples | Best prompt optimization |
| `dspy.GEPA` | Instructions | ~50 examples | Instruction tuning |
| `dspy.BootstrapFinetune` | LM weights | ~500+ examples | Maximum quality, smaller LMs |
| `dspy.BetterTogether` | Instructions + weights | ~500+ examples | Combining prompt + weight tuning |

### Usage

```python
# Define a metric
def metric(example, prediction, trace=None):
    return prediction.answer == example.answer

# Optimize
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized_program = optimizer.compile(my_program, trainset=trainset)

# Use optimized program
result = optimized_program(question="...")

# Save / load
optimized_program.save("optimized.json")
loaded = MyProgram()
loaded.load("optimized.json")
```

### MIPROv2 (recommended for prompt optimization)

```python
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="medium",  # "light", "medium", "heavy"
)
optimized = optimizer.compile(my_program, trainset=trainset)
```

## Evaluation

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=devset,
    metric=metric,
    num_threads=4,
    display_progress=True,
    display_table=5,
)
score = evaluator(my_program)
```

### Common metric patterns

```python
# Exact match
def exact_match(example, pred, trace=None):
    return pred.answer == example.answer

# LM-as-judge
class AssessAnswer(dspy.Signature):
    """Assess if the answer is correct and complete."""
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    is_correct: bool = dspy.OutputField()

def llm_metric(example, pred, trace=None):
    judge = dspy.Predict(AssessAnswer)
    result = judge(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=pred.answer,
    )
    return result.is_correct

# Composite metric
def composite_metric(example, pred, trace=None):
    correct = pred.answer == example.answer
    concise = len(pred.answer.split()) < 50
    return correct + 0.5 * concise  # weighted score
```

## Data Handling

```python
# Create examples
example = dspy.Example(question="What is DSPy?", answer="A framework for LM programs")
example = example.with_inputs("question")  # mark which fields are inputs

# Create dataset
trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in data]

# Load from HuggingFace
from datasets import load_dataset
dataset = load_dataset("hotpotqa", "fullwiki")
trainset = [dspy.Example(**x).with_inputs("question") for x in dataset["train"]]
```

## Retrieval

```python
# ColBERTv2
colbert = dspy.ColBERTv2(url="http://your-server:port/endpoint")
results = colbert("search query", k=3)

# Any retriever via dspy.Retrieve
self.retrieve = dspy.Retrieve(k=5)
passages = self.retrieve(query).passages
```

## Debugging & Inspection

```python
# Inspect last N LM calls
dspy.inspect_history(n=3)

# Enable verbose logging
dspy.configure(lm=lm, trace=[])

# View module structure
print(my_program)

# Check predictions
result = my_program(question="test")
print(result)  # shows all fields
print(result.reasoning)  # specific field
```

## Assertions & Constraints

```python
# Hard constraint (raises error)
dspy.Assert(condition, "Error message")

# Soft constraint (suggests retry)
dspy.Suggest(condition, "Suggestion message")

# Example in a module
class SafeAnswer(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        result = self.generate(question=question)
        dspy.Suggest(len(result.answer) > 10, "Answer should be detailed")
        dspy.Assert(result.answer != "I don't know", "Must provide an answer")
        return result
```

## Common Patterns

### Multi-stage pipeline

```python
class Pipeline(dspy.Module):
    def __init__(self):
        self.step1 = dspy.ChainOfThought("input -> intermediate")
        self.step2 = dspy.ChainOfThought("intermediate -> output")

    def forward(self, input):
        mid = self.step1(input=input)
        return self.step2(intermediate=mid.intermediate)
```

### Typed outputs

```python
from typing import Literal
from pydantic import BaseModel

class ExtractedInfo(BaseModel):
    name: str
    age: int
    city: str

class Extract(dspy.Signature):
    text: str = dspy.InputField()
    info: ExtractedInfo = dspy.OutputField()
```

### Setting different LMs per module

```python
expensive_lm = dspy.LM("openai/gpt-4o")
cheap_lm = dspy.LM("openai/gpt-4o-mini")

my_module.expensive_step.set_lm(expensive_lm)
my_module.cheap_step.set_lm(cheap_lm)
```

## Links

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [DSPy Discord](https://discord.gg/dspy)
- [Tutorials](https://dspy.ai/tutorials/)
