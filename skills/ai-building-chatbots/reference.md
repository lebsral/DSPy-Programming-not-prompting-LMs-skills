> Condensed from [dspy.ai/api/](https://dspy.ai/api/) and [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/). Verify against upstream for latest.

# DSPy & LangGraph API Reference for Chatbots

## DSPy Modules

| Module | Purpose | When to use |
|--------|---------|-------------|
| `dspy.Predict` | Direct LM call, no reasoning step | Intent classification, greetings, summarization |
| `dspy.ChainOfThought` | Adds reasoning before output | Response generation, complaint handling, doc-grounded answers |
| `dspy.Refine` | Runs module N times with reward, returns best | Guardrails — conciseness, tone, no-break-character constraints |

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.ChainOfThought(signature, rationale_field=None, **config)
```

Adds a `reasoning` field automatically before the output. Do not add `reasoning` to your signature — DSPy injects it. Use `dspy.Predict` instead when the call is straightforward (intent classification, greetings) and reasoning adds cost without improving accuracy.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Module to run and refine |
| `N` | `int` | required | Max attempts |
| `reward_fn` | `Callable[[args, pred], float]` | required | Returns 0.0–1.0; higher is better |
| `threshold` | `float \| None` | `None` | Stop early when score >= threshold |

The reward function receives `(args, pred)` — `args` is the dict of module inputs, `pred` is the prediction. Return `0.0` for hard failures (breaks character, prohibited content) and fractional penalties for soft constraints (verbosity, condescending phrasing).

## dspy.History

[API docs](https://dspy.ai/api/primitives/History/)

Typed conversation history for multi-turn chatbots. Works with any LM — DSPy formats it as prior conversation turns.

```python
history = dspy.History(messages=[
    {"role": "user", "content": "What is your return policy?"},
    {"role": "assistant", "content": "30 days, no questions asked."},
])
```

History objects are **immutable** — create a new instance each turn to append prior turns.

Use `dspy.History` as a typed input field in your signature:

```python
class Chat(dspy.Signature):
    history: dspy.History = dspy.InputField(desc="Prior conversation turns")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

**`dspy.History` vs. formatted string**: Use `dspy.History` when you want DSPy to handle turn encoding automatically. Use a formatted `str` (via `format_history(messages[-10:])`) when you need explicit truncation control.

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, auto='light', prompt_model=None, task_model=None,
             max_bootstrapped_demos=4, max_labeled_demos=4,
             num_candidates=None, num_threads=None, seed=9, verbose=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function for a conversation turn |
| `auto` | `'light' \| 'medium' \| 'heavy' \| None` | `'light'` | Optimization intensity |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `4` | Max labeled demos from trainset |

Key method: `.compile(module, trainset=...)` — returns optimized module.

Every `dspy.Example` in the chatbot trainset must call `.with_inputs("conversation_history", "user_message", "context")` — without it the optimizer treats all fields as outputs and optimization silently produces garbage.

## LangGraph Conversation State

```python
from typing import TypedDict, Annotated
import operator

class ConversationState(TypedDict):
    messages: Annotated[list[dict], operator.add]  # appends on update
    current_intent: str
    context: str
    escalate: bool
    resolved: bool
    turn_count: int
```

The `operator.add` reducer causes LangGraph to append rather than replace `messages` on each node update.

## LangGraph Checkpointers

| Checkpointer | Import | Use when |
|---|---|---|
| `MemorySaver` | `langgraph.checkpoint.memory` | Dev / in-process only; lost on restart |
| `PostgresSaver` | `langgraph.checkpoint.postgres` | Production; persists across restarts |

```python
from langgraph.checkpoint.memory import MemorySaver

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user-abc-123"}}

result = app.invoke(initial_state, config=config)   # turn 1
result = app.invoke({"messages": [next_msg]}, config=config)  # turn 2 — state preserved
```

## LangGraph interrupt_before

```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_refund", "cancel_account"],
)

result = app.invoke(input_state, config)  # runs until interrupt node
result = app.invoke(None, config)         # resume after human approval
```

## Quick-Reference Config

```bash
pip install -U dspy langgraph
# dspy >= 2.5 (DSPy 3.2.1 current stable), langgraph >= 0.2
```

```python
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```
