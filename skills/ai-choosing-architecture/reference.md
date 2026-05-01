# ai-choosing-architecture: Reference

## Full module comparison table

| Module | When to use | Latency | Cost | Notes |
|---|---|---|---|---|
| Predict | Direct input-to-output mapping. No reasoning needed. Classification, extraction, formatting. | 1x | 1x | Fastest and cheapest. Use as the baseline. |
| ChainOfThought | Most tasks. Moderate reasoning. When you want the model to show its work. | 1.5-2x | 1.5-2x | Default choice when unsure. |
| ProgramOfThought | Math, computation, data manipulation. Tasks where code is more reliable than prose reasoning. | 2-3x | 2-3x | Generates and executes Python code internally. |
| ReAct | Tasks that require external information or actions that cannot be predetermined. | 3-10x | 3-10x | Latency and cost scale with number of tool calls. |
| CodeAct | Tasks that require writing and running code as part of the answer. | 3-10x | 3-10x | Stronger than ReAct for coding-heavy workflows. |
| MultiChainComparison | When you need the single best answer and can afford 3-5x cost. | 3-5x | 3-5x | Runs N chains and picks the best. |
| BestOfN | When you have a reward/scoring function and want to sample the best output. | Nx | Nx | Good for tasks with verifiable correctness. |
| Refine | Iterative improvement. When a single pass is not enough and you want the model to self-edit. | 2-4x | 2-4x | Each refinement pass is a full LM call. |
| RLM | RL-style reward-driven generation. Experimental use cases. | Varies | Varies | Less common; check DSPy docs for current API. |
| Parallel | Running multiple independent sub-tasks simultaneously. | 1x wall clock | Nx total | Good for fan-out patterns where sub-tasks are independent. |

### When to use each module (extended notes)

**Predict** — Use when the task is a direct mapping and reasoning would add noise. Examples: sentiment classification, named entity extraction, format conversion, label assignment from a known taxonomy. If the correct answer is deterministic given the input, Predict is the right choice.

**ChainOfThought** — Use as the default for any task where the model benefits from thinking before answering. Open-ended question answering, summarization, explanation generation, and tasks with ambiguous inputs all benefit from CoT. The reasoning trace also makes debugging easier.

**ProgramOfThought** — Use when the task involves arithmetic, counting, unit conversion, statistical computation, or any problem where writing a small Python program is more reliable than generating prose. ProgramOfThought generates executable code and runs it; the code output becomes the answer.

**ReAct** — Use when the correct answer depends on information that is not in the context at call time. Examples: answering questions about current events (needs search), looking up customer records (needs database), checking live prices (needs API). Each tool call is a round-trip LM invocation, so keep the tool set small and focused.

**CodeAct** — Use when the task itself is to write and run code, or when the reasoning loop should be expressed as code execution steps rather than natural language. More powerful than ReAct for coding-heavy workflows.

**MultiChainComparison** — Use when single-pass accuracy is measurably insufficient and cost is not the primary constraint. Runs N independent reasoning chains and selects the most consistent answer. Effective for high-stakes single answers (medical triage, legal classification).

**BestOfN** — Use when you have a reliable scoring function (a reward model, a unit test, a regex check). Samples N outputs, scores each, and returns the best. Requires a good scorer; without one, it is just random selection.

**Refine** — Use when you want the model to iteratively improve an initial draft. Effective for writing tasks where a first pass is good but a second pass makes it excellent. Each refinement is a full LM call.

**Parallel** — Use for fan-out patterns: summarizing many documents independently, classifying a batch of items, running the same task over multiple inputs simultaneously. Reduces wall-clock time when the sub-tasks are independent.

---

## Architecture templates

### Template 1: Simple classifier (Predict)

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class ClassifyTicket(dspy.Signature):
    """Classify a support ticket into exactly one category."""
    ticket_text: str = dspy.InputField(desc="The raw support ticket text")
    category: str = dspy.OutputField(
        desc="One of: billing, technical, account, other"
    )

classifier = dspy.Predict(ClassifyTicket)

# Basic usage
result = classifier(ticket_text="My invoice shows the wrong amount.")
print(result.category)  # "billing"

# With optimization
from dspy.teleprompt import BootstrapFewShot

def accuracy_metric(example, pred, trace=None):
    return example.category == pred.category

optimizer = BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(classifier, trainset=trainset)
```

### Template 2: Reasoning task (ChainOfThought)

```python
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

class AnswerQuestion(dspy.Signature):
    """Answer the question based on the provided context."""
    context: str = dspy.InputField(desc="Background information")
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="A clear, concise answer")

answerer = dspy.ChainOfThought(AnswerQuestion)

result = answerer(
    context="DSPy is a framework for programming language models...",
    question="What is DSPy used for?"
)
print(result.reasoning)  # The chain-of-thought trace
print(result.answer)

# Optimize with MIPROv2 for best quality
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(metric=your_metric, auto="medium")
optimized = optimizer.compile(answerer, trainset=trainset, valset=valset)
```

### Template 3: Tool-using agent (ReAct)

```python
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

# Define tools as plain Python functions with docstrings
def search_web(query: str) -> str:
    """Search the web and return a summary of results."""
    # your search implementation
    ...

def fetch_page(url: str) -> str:
    """Fetch the content of a web page."""
    # your fetch implementation
    ...

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    return str(eval(expression))  # use a safe evaluator in production

class ResearchTask(dspy.Signature):
    """Research a topic and provide a well-sourced answer."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Detailed answer with sources cited")

agent = dspy.ReAct(ResearchTask, tools=[search_web, fetch_page, calculate])

result = agent(question="What was the GDP of Germany in 2023?")
print(result.answer)

# For agents, BootstrapFewShot is usually enough
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=your_metric, max_bootstrapped_demos=2)
optimized = optimizer.compile(agent, trainset=trainset)
```

### Template 4: Classify-then-generate pipeline

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class ClassifyIntent(dspy.Signature):
    """Classify the user intent from the support message."""
    message: str = dspy.InputField()
    intent: str = dspy.OutputField(
        desc="One of: question, complaint, cancellation_request, praise"
    )

class DraftReply(dspy.Signature):
    """Draft a professional support reply given the message and its intent."""
    message: str = dspy.InputField()
    intent: str = dspy.InputField()
    reply: str = dspy.OutputField(desc="Professional, empathetic reply under 150 words")

class SupportResponder(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(ClassifyIntent)
        self.draft = dspy.ChainOfThought(DraftReply)

    def forward(self, message: str) -> dspy.Prediction:
        intent = self.classify(message=message).intent
        reply = self.draft(message=message, intent=intent).reply
        return dspy.Prediction(intent=intent, reply=reply)

responder = SupportResponder()
result = responder(message="I have been charged twice for my subscription.")
print(result.intent)
print(result.reply)
```

### Template 5: RAG pipeline (retrieve, reason, generate)

```python
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

# Configure your retriever (colbert, faiss, weaviate, etc.)
retriever = dspy.Retrieve(k=5)

class IdentifyRelevantFacts(dspy.Signature):
    """From the retrieved passages, identify the facts most relevant to the question."""
    question: str = dspy.InputField()
    passages: list[str] = dspy.InputField(desc="Retrieved context passages")
    relevant_facts: str = dspy.OutputField(
        desc="Bullet-point list of facts directly relevant to the question"
    )

class SynthesizeAnswer(dspy.Signature):
    """Synthesize a final answer from the identified relevant facts."""
    question: str = dspy.InputField()
    relevant_facts: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Clear, well-supported answer")
    citations: str = dspy.OutputField(desc="Which facts were used")

class RAGPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = retriever
        self.identify = dspy.ChainOfThought(IdentifyRelevantFacts)
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        passages = self.retrieve(question).passages
        facts = self.identify(question=question, passages=passages).relevant_facts
        result = self.synthesize(question=question, relevant_facts=facts)
        return dspy.Prediction(
            answer=result.answer,
            citations=result.citations,
            passages=passages,
        )

rag = RAGPipeline()
result = rag(question="What are the refund policy terms?")
print(result.answer)
```

---

## Optimizer pairing details

### BootstrapFewShot
- **Best for:** Getting a quick baseline on any architecture. Works well when you have 20-100 labeled examples.
- **How it works:** Runs the program on training examples, collects traces where the metric passes, and uses those traces as few-shot demonstrations.
- **When to move on:** If accuracy plateaus after 4-8 demos, switch to MIPROv2.

### MIPROv2
- **Best for:** Single modules and pipelines where you have 50+ examples and care about maximum quality.
- **How it works:** Bayesian optimization over both the instruction and the few-shot demonstrations. Tries many prompt variants and picks the best.
- **Cost:** More expensive to run than BootstrapFewShot (many optimization calls). Run it once, save the optimized program.

### BootstrapFinetune
- **Best for:** High-traffic production pipelines where inference cost matters. Generates training data from successful traces and fine-tunes the LM weights.
- **Requires:** A fine-tunable model (GPT-4o fine-tune, local model, etc.).

### BetterTogether
- **Best for:** Combining prompt optimization and fine-tuning. Alternates between MIPROv2 (prompt) and BootstrapFinetune (weights) to get the best of both.
- **Use when:** You need both high quality and low inference cost.

---

## Cost estimation rules of thumb

| Architecture | Relative cost per request | Example at $0.01/1k tokens |
|---|---|---|
| Single Predict (short) | 1x | ~$0.0001 |
| Single ChainOfThought | 1.5-2x | ~$0.0002 |
| Two-stage pipeline | 2-3x | ~$0.0003 |
| Three-stage RAG pipeline | 3-5x | ~$0.0005 |
| ReAct (3 tool calls avg) | 4-6x | ~$0.0006 |
| ReAct (10 tool calls) | 10-15x | ~$0.0015 |
| MultiChainComparison (N=3) | 3-4x | ~$0.0004 |
| BestOfN (N=5) | 5x | ~$0.0005 |

These are rough multipliers. Actual cost depends on token counts, model choice, and retrieval corpus size. Always measure on real inputs before scaling.
