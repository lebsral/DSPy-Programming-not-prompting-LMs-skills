# dspy-langfuse Examples

## Example 1: Traced classification pipeline with scoring

A support ticket classifier that auto-traces every DSPy call and pushes evaluation scores back to Langfuse for tracking quality over time.

```python
import dspy
from langfuse import get_client, observe, propagate_attributes
from openinference.instrumentation.dspy import DSPyInstrumentor

# --- Setup ---
DSPyInstrumentor().instrument()
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.
langfuse = get_client()

# --- DSPy module ---
class TicketClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(
            "ticket_text -> category: str, priority: str, summary: str"
        )

    def forward(self, ticket_text):
        return self.classify(ticket_text=ticket_text)

classifier = TicketClassifier()

# --- Traced endpoint ---
@observe()
def classify_ticket(ticket_text: str, user_id: str):
    with propagate_attributes(
        user_id=user_id,
        tags=["classification", "support"],
        metadata={"source": "zendesk"},
    ):
        result = classifier(ticket_text=ticket_text)
        return result

# --- Run and score ---
result = classify_ticket(
    ticket_text="My payment failed twice and I was charged both times. Need refund ASAP.",
    user_id="customer_789",
)
print(f"Category: {result.category}, Priority: {result.priority}")
print(f"Summary: {result.summary}")

# Push a quality score to Langfuse
trace_id = langfuse.get_current_trace_id()
if trace_id:
    langfuse.score(
        trace_id=trace_id,
        name="classification_correct",
        value=True,
        data_type="BOOLEAN",
    )

langfuse.flush()
```

What this demonstrates:

- **Auto-instrumentation** captures all DSPy internals (ChainOfThought prompt, response, tokens)
- **`@observe()` + `propagate_attributes()`** adds user context and tags to the auto-captured trace
- **Scoring** attaches a boolean quality score to the trace for later analysis
- **`langfuse.flush()`** ensures traces are sent before the script exits

## Example 2: Experiment tracking with DSPy optimization

Compare two prompt strategies by running DSPy evaluation and logging results as Langfuse experiments.

```python
import dspy
from dspy.evaluate import Evaluate
from langfuse import get_client, observe, propagate_attributes
from openinference.instrumentation.dspy import DSPyInstrumentor

# --- Setup ---
DSPyInstrumentor().instrument()
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.
langfuse = get_client()

# --- Two program variants ---
baseline = dspy.Predict("question -> answer")
cot = dspy.ChainOfThought("question -> answer")

# --- Dataset ---
devset = [
    dspy.Example(question="What causes rain?", answer="Water evaporation and condensation").with_inputs("question"),
    dspy.Example(question="Why is the sky blue?", answer="Rayleigh scattering of sunlight").with_inputs("question"),
    # ... more examples
]

def exact_match(example, pred, trace=None):
    return pred.answer.strip().lower() == example.answer.strip().lower()

# --- Evaluate both variants with traced calls ---
evaluator = Evaluate(devset=devset, metric=exact_match, num_threads=2)

@observe(name="experiment-baseline")
def run_baseline():
    with propagate_attributes(
        tags=["experiment", "baseline"],
        metadata={"variant": "predict", "model": "gpt-4o-mini"},
    ):
        return evaluator(baseline)

@observe(name="experiment-cot")
def run_cot():
    with propagate_attributes(
        tags=["experiment", "cot"],
        metadata={"variant": "chain-of-thought", "model": "gpt-4o-mini"},
    ):
        return evaluator(cot)

baseline_score = run_baseline()
cot_score = run_cot()

print(f"Baseline: {baseline_score}, CoT: {cot_score}")

langfuse.flush()
# Compare runs in Langfuse dashboard by filtering on tags: "baseline" vs "cot"
```

What this demonstrates:

- **Experiment comparison** by tagging two DSPy evaluation runs differently
- **All individual LM calls traced** so you can drill into why one variant outperforms
- **Metadata** captures which model and variant was used for each experiment
- **Dashboard filtering** by tags lets you compare score distributions side by side

## Example 3: Multi-turn agent with session tracking

A ReAct agent where every turn in the conversation is grouped into a Langfuse session.

```python
import dspy
from langfuse import get_client, observe, propagate_attributes
from openinference.instrumentation.dspy import DSPyInstrumentor

# --- Setup ---
DSPyInstrumentor().instrument()
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.
langfuse = get_client()

# --- Tools ---
def lookup_order(order_id: str) -> str:
    """Look up order status by ID."""
    # Simulated database lookup
    orders = {"ORD-123": "shipped", "ORD-456": "processing", "ORD-789": "delivered"}
    return orders.get(order_id, "not found")

def check_refund_policy(product_type: str) -> str:
    """Check refund eligibility for a product type."""
    policies = {"electronics": "30 days", "clothing": "60 days", "food": "no refunds"}
    return policies.get(product_type, "standard 30 days")

# --- Agent ---
agent = dspy.ReAct(
    "question -> answer",
    tools=[lookup_order, check_refund_policy],
)

# --- Session-aware handler ---
@observe()
def handle_turn(question: str, session_id: str, user_id: str):
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        tags=["support-agent"],
    ):
        return agent(question=question)

# --- Simulate a multi-turn conversation ---
session = "session_support_001"
user = "customer_42"

r1 = handle_turn("What is the status of order ORD-123?", session, user)
print(f"Turn 1: {r1.answer}")

r2 = handle_turn("Can I get a refund on the electronics I ordered?", session, user)
print(f"Turn 2: {r2.answer}")

r3 = handle_turn("What about the clothing items in ORD-456?", session, user)
print(f"Turn 3: {r3.answer}")

langfuse.flush()
# All 3 turns appear under session_support_001 in the Langfuse dashboard
# Each turn shows the ReAct loop: reasoning -> tool call -> observation -> answer
```

What this demonstrates:

- **Session grouping** links multiple turns under one `session_id` in the dashboard
- **ReAct tool calls traced** showing the full reasoning -> action -> observation loop
- **User tracking** associates all turns with a specific customer
- **Production pattern** for support chatbots where you need to review full conversations
