# Ollama Examples

## Set up Ollama for a DSPy project

```bash
# Install and start
brew install ollama  # or: curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# Pull a model
ollama pull llama3.1:8b

# Verify it's running
ollama list
```

```python
import dspy

# Configure DSPy with Ollama
lm = dspy.LM(
    "ollama_chat/llama3.1:8b",
    api_base="http://localhost:11434",
    api_key="",
    num_ctx=8192,
)
dspy.configure(lm=lm)

# Test with a simple task
classify = dspy.Predict("email -> category: str, priority: str")
result = classify(email="Server is down, customers can't log in!")
print(f"Category: {result.category}, Priority: {result.priority}")
```

## Optimize a classifier with a local model

```python
import dspy

lm = dspy.LM(
    "ollama_chat/llama3.1:8b",
    api_base="http://localhost:11434",
    api_key="",
    num_ctx=8192,
)
dspy.configure(lm=lm)

# Training data
trainset = [
    dspy.Example(text="Server is down!", label="critical").with_inputs("text"),
    dspy.Example(text="Can't reset password", label="account").with_inputs("text"),
    dspy.Example(text="How much is the pro plan?", label="billing").with_inputs("text"),
    dspy.Example(text="Feature request: dark mode", label="feature").with_inputs("text"),
    dspy.Example(text="API returns 500 errors", label="critical").with_inputs("text"),
    dspy.Example(text="Invoice is wrong", label="billing").with_inputs("text"),
    dspy.Example(text="Want to cancel subscription", label="account").with_inputs("text"),
    dspy.Example(text="App crashes on iOS 17", label="critical").with_inputs("text"),
]

def metric(example, prediction, trace=None):
    return prediction.label.strip().lower() == example.label.strip().lower()

# Use BootstrapFewShot — fastest optimizer for local models
program = dspy.Predict("text -> label")
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
optimized = optimizer.compile(program, trainset=trainset)

# Test
result = optimized(text="Database migration failed")
print(f"Label: {result.label}")

# Save for later
optimized.save("optimized_classifier.json")
```

## Multi-model pipeline (big + small)

```python
import dspy

# Small model for simple tasks
small = dspy.LM(
    "ollama_chat/llama3.2:3b",
    api_base="http://localhost:11434",
    api_key="",
    num_ctx=4096,
)

# Big model for complex reasoning
big = dspy.LM(
    "ollama_chat/llama3.1:8b",
    api_base="http://localhost:11434",
    api_key="",
    num_ctx=8192,
)

dspy.configure(lm=small)

class SupportPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("ticket -> category, priority")
        self.draft = dspy.ChainOfThought("ticket, category, priority -> response")

    def forward(self, ticket):
        triage = self.classify(ticket=ticket)
        return self.draft(
            ticket=ticket,
            category=triage.category,
            priority=triage.priority,
        )

pipeline = SupportPipeline()
pipeline.classify.set_lm(small)  # fast classification
pipeline.draft.set_lm(big)       # better response quality

result = pipeline(ticket="I was charged twice for my subscription last month")
print(f"Category: {result.category}")
print(f"Priority: {result.priority}")
print(f"Response: {result.response}")
```
