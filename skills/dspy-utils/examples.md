# Examples: DSPy Utilities

## Example 1: configure_cache patterns for development vs production

Control DSPy caching to match your workflow -- aggressive caching during optimization, disabled during data generation.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# ---- Development: disable disk cache to always hit the API ----
# Use this when testing prompt changes or debugging latency
dspy.configure_cache(enable_disk_cache=False)

classifier = dspy.Predict("ticket_text -> urgency: Literal['low', 'medium', 'high', 'critical']")
result = classifier(ticket_text="Production database is down")

# ---- Data generation: disable all caching for diverse outputs ----
# Identical prompts need different responses, so cache would return the same result
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

generate = dspy.Predict("topic -> example_sentence")
sentences = [generate(topic="weather").example_sentence for _ in range(5)]
# All 5 calls hit the API -- caching would collapse them to one response

# ---- Optimization: use default caching (both enabled) ----
# Never disable cache during optimizer runs -- they make many redundant calls
dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)  # default

optimizer = dspy.BootstrapFewShot(metric=lambda ex, pred, trace=None: True, max_bootstrapped_demos=3)
optimized = optimizer.compile(classifier, trainset=[...])

# ---- Security: harden against pickle injection in shared environments ----
dspy.configure_cache(restrict_pickle=True)
```

### What to notice

- `enable_disk_cache=False` disables only the file-based cache; in-memory cache still speeds up repeated calls within the same process.
- `enable_disk_cache=False, enable_memory_cache=False` is the full off switch -- use for data generation or latency benchmarking.
- Never disable both during optimizer runs -- the optimizer makes hundreds of repeated calls and relies on cache to avoid redundant API charges.
- Per-LM cache control (`dspy.LM("model", cache=False)`) is cleaner when you want one LM cached and another not.

## Example 2: Debug workflow with inspect_history

Walk through a typical debugging session when a DSPy program produces unexpected output. Use `inspect_history` to see exactly what the LM received and returned.

```python
import dspy
from typing import Literal

# Set up
lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# A classification program that isn't working right
class TicketClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(
            "ticket_text -> urgency: Literal['low', 'medium', 'high', 'critical']"
        )

    def forward(self, ticket_text):
        return self.classify(ticket_text=ticket_text)

classifier = TicketClassifier()

# --- Step 1: Run on a failing input ---
result = classifier(ticket_text="Production database is down, all customers affected")
print(f"Got: {result.urgency}")
# Expected: "critical", got: "high" -- why?

# --- Step 2: Inspect what was sent to the LM ---
dspy.inspect_history(n=1)
# This prints:
#   - The full system prompt DSPy generated
#   - The user message with the ticket text
#   - The raw LM response
#   - Which adapter formatted the request

# --- Step 3: Look at the prompt ---
# You might see that the signature description is too vague.
# The LM doesn't know that "all customers affected" implies critical urgency.

# --- Step 4: Improve the signature with a better docstring ---
class ClassifyTicket(dspy.Signature):
    """Classify support ticket urgency. Critical means production outages
    affecting multiple customers. High means significant issues affecting
    individual users."""
    ticket_text: str = dspy.InputField()
    urgency: Literal["low", "medium", "high", "critical"] = dspy.OutputField()

class BetterClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(ClassifyTicket)

    def forward(self, ticket_text):
        return self.classify(ticket_text=ticket_text)

better = BetterClassifier()
result = better(ticket_text="Production database is down, all customers affected")
print(f"Got: {result.urgency}")

# --- Step 5: Inspect again to verify the improved prompt ---
dspy.inspect_history(n=1)
# Now the prompt includes the detailed docstring, and the LM returns "critical"

# --- Bonus: Print the module tree to verify structure ---
print(better)
# BetterClassifier(
#   classify = Predict(ClassifyTicket)
# )
```

### What to notice

- `dspy.inspect_history(n=1)` is your first move when a program behaves unexpectedly. It shows the exact prompt the LM saw.
- The most common fixes after inspecting: improve the signature docstring, add `desc=` to `InputField`/`OutputField`, or add few-shot examples via optimization.
- Use `print(module)` to verify the module structure -- sometimes a sub-module is wired incorrectly.
- Set `temperature=0.0` during debugging for deterministic, reproducible outputs.

## Example 3: Save/load optimized programs for production deployment

Optimize a program once, save the learned state, and load it in production without re-running optimization.

```python
import dspy
from dspy.evaluate import Evaluate

# ==============================
# Part A: Optimize and save (run once, offline)
# ==============================

lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Define the program
class SupportResponder(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("ticket -> category, urgency")
        self.respond = dspy.ChainOfThought("ticket, category, urgency -> response")

    def forward(self, ticket):
        classification = self.classify(ticket=ticket)
        return self.respond(
            ticket=ticket,
            category=classification.category,
            urgency=classification.urgency,
        )

# Training data
trainset = [
    dspy.Example(
        ticket="Can't log in to my account",
        category="account",
        urgency="medium",
        response="I can help you regain access. Please try resetting your password...",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Production API returning 500 errors",
        category="technical",
        urgency="critical",
        response="I'm escalating this immediately. Our on-call team is investigating...",
    ).with_inputs("ticket"),
    # ... more examples
]

# Define a metric
def quality_metric(example, prediction, trace=None):
    return (
        prediction.category == example.category
        and prediction.urgency == example.urgency
    )

# Optimize
optimizer = dspy.BootstrapFewShot(metric=quality_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(SupportResponder(), trainset=trainset)

# Evaluate before saving
evaluator = Evaluate(devset=trainset[:10], metric=quality_metric, num_threads=4)
score = evaluator(optimized)
print(f"Score: {score}")

# Save the optimized state
optimized.save("support_responder_v1.json")
print("Saved optimized program to support_responder_v1.json")

# ==============================
# Part B: Load in production (run on every request)
# ==============================

import dspy

# Must configure LM before loading
lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Create a fresh instance and load saved state
program = SupportResponder()
program.load("support_responder_v1.json")

# The program now has the optimized few-shot demos and instructions
result = program(ticket="I was charged twice on my last invoice")
print(f"Category: {result.category}")
print(f"Urgency: {result.urgency}")
print(f"Response: {result.response}")

# ==============================
# Part C: Version management
# ==============================

# Save new versions as you iterate
# optimized_v2.save("support_responder_v2.json")

# Compare versions
# program_v1 = SupportResponder()
# program_v1.load("support_responder_v1.json")
# score_v1 = evaluator(program_v1)
#
# program_v2 = SupportResponder()
# program_v2.load("support_responder_v2.json")
# score_v2 = evaluator(program_v2)
#
# print(f"v1: {score_v1}, v2: {score_v2}")
```

### What to notice

- The class definition (`SupportResponder`) must exist in both the optimization script and the production code. `load()` restores learned state (demos, instructions) into the module's `Predict` sub-modules, but the Python logic in `forward()` comes from your code.
- Always call `dspy.configure(lm=lm)` before calling `load()`. The saved state does not include the LM configuration.
- You can switch models between optimization and production. For example, optimize with `gpt-4o` for better demos, then serve with `gpt-4o-mini` for lower costs. The few-shot demos still help the cheaper model.
- Version your saved files (e.g., `_v1.json`, `_v2.json`) and compare scores with `Evaluate` before deploying a new version.
