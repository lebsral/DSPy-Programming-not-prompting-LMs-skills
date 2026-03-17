# Examples: Configuring Language Models

## Example 1: Multi-provider setup

Switch between OpenAI and Anthropic, showing how the provider string format works and how to verify your configuration.

```python
import dspy

# --- OpenAI setup ---
openai_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
dspy.configure(lm=openai_lm)

# Test it
classify = dspy.ChainOfThought("text -> sentiment: str")
result = classify(text="I love this product!")
print(f"OpenAI says: {result.sentiment}")

# --- Switch to Anthropic ---
anthropic_lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929", temperature=0.0)
dspy.configure(lm=anthropic_lm)

# Same module, different provider -- no code changes needed
result = classify(text="I love this product!")
print(f"Anthropic says: {result.sentiment}")

# --- Try Google ---
google_lm = dspy.LM("gemini/gemini-2.0-flash", temperature=0.0)
dspy.configure(lm=google_lm)

result = classify(text="I love this product!")
print(f"Google says: {result.sentiment}")

# --- Together AI (open-source) ---
together_lm = dspy.LM("together_ai/meta-llama/Llama-3-70b-chat-hf", temperature=0.0)
dspy.configure(lm=together_lm)

result = classify(text="I love this product!")
print(f"Llama 3 says: {result.sentiment}")
```

### What to notice

- The provider string format is always `"provider/model-name"`.
- Your DSPy modules and signatures stay exactly the same across providers.
- Set the API key for each provider as an environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, etc.).
- `temperature=0.0` ensures deterministic outputs for testing.

## Example 2: Cost optimization with per-module LM routing

Use an expensive model for tasks that need it and a cheap model for everything else. This pipeline classifies support tickets (easy -- use cheap model) and then generates a detailed response (hard -- use expensive model).

```python
import dspy
from typing import Literal

# --- Define two LMs at different price points ---
cheap_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)    # ~$0.15/1M input tokens
expensive_lm = dspy.LM("openai/gpt-4o", temperature=0.0)      # ~$2.50/1M input tokens

# Set the cheap model as the default
dspy.configure(lm=cheap_lm)

# --- Signatures ---
class ClassifyTicket(dspy.Signature):
    """Classify a support ticket into a category."""
    ticket_text: str = dspy.InputField()
    urgency: Literal["low", "medium", "high", "critical"] = dspy.OutputField()
    category: Literal["billing", "technical", "account", "feature_request"] = dspy.OutputField()

class DraftResponse(dspy.Signature):
    """Draft a helpful, empathetic response to a support ticket."""
    ticket_text: str = dspy.InputField()
    urgency: str = dspy.InputField()
    category: str = dspy.InputField()
    response: str = dspy.OutputField(desc="A helpful reply to the customer, 2-4 sentences")

# --- Pipeline with mixed models ---
class SupportPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(ClassifyTicket)
        self.draft = dspy.ChainOfThought(DraftResponse)

    def forward(self, ticket_text):
        classification = self.classify(ticket_text=ticket_text)
        return self.draft(
            ticket_text=ticket_text,
            urgency=classification.urgency,
            category=classification.category,
        )

pipeline = SupportPipeline()

# Route: cheap model classifies, expensive model drafts responses
pipeline.classify.set_lm(cheap_lm)
pipeline.draft.set_lm(expensive_lm)

# --- Use it ---
result = pipeline(ticket_text="I've been charged twice for my subscription and I need a refund ASAP")
print(f"Response: {result.response}")

# --- Alternative: use dspy.context for temporary overrides ---
# Everything uses cheap_lm by default, but you can override per-call:
with dspy.context(lm=expensive_lm):
    important_result = pipeline(ticket_text="Our production system is down")
```

### What to notice

- `set_lm()` is permanent for that module instance -- every call to `pipeline.classify` uses `cheap_lm`.
- `dspy.context(lm=...)` is temporary -- it only applies inside the `with` block.
- Classification is a simple routing task where a cheap model is sufficient. Response drafting benefits from a more capable model.
- You can mix and match: use `set_lm()` for the common case and `dspy.context()` for exceptions.

## Example 3: Local model setup with Ollama and vLLM

Run models on your own hardware for data privacy, zero API costs, or offline use.

### Option A: Ollama (easiest local setup)

```bash
# Install Ollama: https://ollama.ai
# Pull a model
ollama pull llama3.1
ollama pull mistral

# Ollama serves on port 11434 by default
ollama serve
```

```python
import dspy

# Connect to Ollama
lm = dspy.LM(
    "ollama_chat/llama3.1",
    api_base="http://localhost:11434",
    temperature=0.7,
    max_tokens=1000,
)
dspy.configure(lm=lm)

# Use it like any other LM
classify = dspy.ChainOfThought("text -> category: str")
result = classify(text="The API keeps returning 500 errors")
print(result.category)

# Switch to a different local model
mistral_lm = dspy.LM(
    "ollama_chat/mistral",
    api_base="http://localhost:11434",
    temperature=0.7,
)
dspy.configure(lm=mistral_lm)
```

### Option B: vLLM (high-performance serving)

```bash
# Install vLLM
pip install vllm

# Serve a model with OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b-chat-hf \
    --port 8000
```

```python
import dspy

# Connect to vLLM using the OpenAI-compatible provider
lm = dspy.LM(
    "openai/meta-llama/Llama-3-8b-chat-hf",
    api_base="http://localhost:8000/v1",
    api_key="none",  # vLLM doesn't need a real key
    temperature=0.7,
    max_tokens=1000,
)
dspy.configure(lm=lm)

# Works exactly like a cloud model
summarize = dspy.ChainOfThought("document -> summary")
result = summarize(document="DSPy is a framework for programming language models...")
print(result.summary)
```

### Option C: Mix local and cloud models

Use a local model for cheap/private tasks and a cloud model for quality-critical tasks:

```python
import dspy

# Local model for classification (free, private)
local_lm = dspy.LM(
    "ollama_chat/llama3.1",
    api_base="http://localhost:11434",
)

# Cloud model for generation (better quality)
cloud_lm = dspy.LM("openai/gpt-4o")

# Default to local
dspy.configure(lm=local_lm)

class HybridPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> category: str")
        self.generate = dspy.ChainOfThought("text, category -> response: str")

    def forward(self, text):
        classification = self.classify(text=text)
        return self.generate(text=text, category=classification.category)

pipeline = HybridPipeline()
pipeline.classify.set_lm(local_lm)    # Runs locally -- free, data stays on your machine
pipeline.generate.set_lm(cloud_lm)    # Cloud model for quality generation

result = pipeline(text="How do I reset my password?")
print(result.response)
```

### What to notice

- **Ollama** is the easiest way to run models locally. Use `"ollama_chat/model-name"` with `api_base`.
- **vLLM** gives higher throughput for production. Use `"openai/model-name"` with `api_base` since vLLM exposes an OpenAI-compatible API.
- Any server that implements the OpenAI `/v1/chat/completions` endpoint works with the `"openai/..."` provider string.
- Local models benefit the most from DSPy optimization -- run `/ai-improving-accuracy` to get better results from smaller models.
- You can freely mix local and cloud models in the same pipeline using `set_lm()`.
