---
name: ai-kickoff
description: Scaffold a new AI feature powered by DSPy. Use when adding AI to your app, starting a new AI project, building an AI-powered feature, setting up a DSPy program from scratch, or bootstrapping an LLM-powered backend.
disable-model-invocation: true
---

# Start a New AI Feature

Create a project structure for building an AI-powered feature with DSPy:

```
$ARGUMENTS/
├── main.py          # Entry point — run your AI feature
├── program.py       # AI logic (DSPy module)
├── metrics.py       # How to measure if the AI is working
├── optimize.py      # Make the AI better automatically
├── evaluate.py      # Test the AI's quality
├── data.py          # Training/test data loading
└── requirements.txt # Dependencies
```

## Step 1: Gather requirements

Ask the user:
1. **What should the AI do?** (sort content, answer questions, extract data, take actions, or describe it)
2. **What goes in and what comes out?** (e.g., "customer email in, category out" or "question in, answer out")
3. **Do you have example data?** (if yes, what format — CSV, JSON, database?)
4. **Which AI provider?** (default: OpenAI — DSPy works with any provider)

## Step 2: Generate the project

### `requirements.txt`

```
dspy>=2.5
```

Add `datasets` if loading from HuggingFace. Add provider-specific packages if needed.

### `data.py`

Create dataset loading utilities:

```python
import dspy

def load_data():
    """Load and prepare training/dev data.

    Returns:
        tuple: (trainset, devset) as lists of dspy.Example
    """
    # TODO: Replace with actual data loading
    examples = [
        dspy.Example(input_field="...", output_field="...").with_inputs("input_field"),
    ]

    split = int(0.8 * len(examples))
    return examples[:split], examples[split:]
```

Adapt field names to match the user's inputs/outputs.

### `program.py`

Create the DSPy module. Choose the right module based on the task:

- **Simple input/output**: `dspy.Predict`
- **Needs reasoning**: `dspy.ChainOfThought` (most tasks)
- **Math/computation**: `dspy.ProgramOfThought`
- **Needs tools**: `dspy.ReAct`

```python
import dspy

class MySignature(dspy.Signature):
    """Describe the task here."""
    # Adapt fields to user's task
    input_field: str = dspy.InputField(desc="description")
    output_field: str = dspy.OutputField(desc="description")

class MyProgram(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought(MySignature)

    def forward(self, **kwargs):
        return self.predict(**kwargs)
```

### `metrics.py`

```python
def metric(example, prediction, trace=None):
    """Score how good the AI's output is.

    Args:
        example: Expected output (ground truth)
        prediction: What the AI actually produced
        trace: Optional trace for optimization

    Returns:
        float: Score between 0 and 1
    """
    # TODO: Implement task-specific metric
    return prediction.output_field == example.output_field
```

### `evaluate.py`

```python
import dspy
from dspy.evaluate import Evaluate
from program import MyProgram
from metrics import metric
from data import load_data

# Configure AI provider
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Load data
_, devset = load_data()

# Test quality
program = MyProgram()
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
score = evaluator(program)
print(f"Score: {score}")
```

### `optimize.py`

```python
import dspy
from program import MyProgram
from metrics import metric
from data import load_data

# Configure AI provider
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Load data
trainset, devset = load_data()

# Automatically improve the AI's prompts
program = MyProgram()
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(program, trainset=trainset)

# Check improvement
from dspy.evaluate import Evaluate
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
score = evaluator(optimized)
print(f"Optimized score: {score}")

# Save
optimized.save("optimized.json")
```

### `main.py`

```python
import dspy
from program import MyProgram

# Configure AI provider
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Load optimized version if available
program = MyProgram()
try:
    program.load("optimized.json")
    print("Loaded optimized program")
except FileNotFoundError:
    print("Running unoptimized program")

# Run
result = program(input_field="test input")
print(result)
```

## Step 3: Explain next steps

After generating the project, tell the user:

1. **Fill in `data.py`** with real training data (20+ examples)
2. **Run `evaluate.py`** to see how well the AI works now
3. **Run `optimize.py`** to automatically improve quality
4. **Run `main.py`** to use the AI

Next: `/ai-improving-accuracy` to measure and improve your AI's quality.
