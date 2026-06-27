> Condensed from [dspy.ai/api/modules/](https://dspy.ai/api/modules/) and [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview). Verify against upstream for latest.

# Multi-Agent Coordination — API Reference

## Packages

```bash
pip install -U dspy langgraph langchain-community
```

| Package | Role |
|---------|------|
| `dspy` 3.2.1 | Agent reasoning, signatures, optimization |
| `langgraph` 0.2.x | Graph orchestration, routing, state |
| `langchain-community` | Pre-built tools (search, etc.) — optional |

```python
import dspy
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

## dspy.Module — Agent base class

[API docs](https://dspy.ai/api/modules/Module/)

Subclass `dspy.Module` for every agent. Declare internal predictors as instance attributes so optimizers can discover and tune them.

```python
class MyAgent(dspy.Module):
    def __init__(self):
        self.reason = dspy.ChainOfThought(MySignature)

    def forward(self, **inputs):
        return self.reason(**inputs)
```

| Method | Description |
|--------|-------------|
| `set_lm(lm)` | Override the LM for this module and all its predictors |
| `save(path)` / `load(path)` | Persist and restore optimized demos and instructions |
| `batch(examples, num_threads, max_errors, timeout)` | Parallel processing across a list |

## dspy.ReAct — Tool-using agent

[API docs](https://dspy.ai/api/modules/ReAct/)

```python
dspy.ReAct(signature, tools, max_iters=20)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Input/output contract |
| `tools` | `list[Callable \| dspy.Tool]` | required | Functions the agent can call |
| `max_iters` | `int` | `20` | Max Thought-Action-Observation cycles |

Returns `dspy.Prediction` with your output fields plus `.trajectory`. A `finish` tool is injected automatically.

## dspy.Tool.from_langchain

Wraps a LangChain tool for use in `dspy.ReAct`:

```python
from langchain_community.tools import DuckDuckGoSearchRun
search_tool = dspy.Tool.from_langchain(DuckDuckGoSearchRun())
agent = dspy.ReAct("question -> findings", tools=[search_tool], max_iters=5)
```

## dspy.Refine — Quality gate on agent output

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold=0.8)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Agent module to wrap |
| `N` | `int` | required | Max retries with feedback |
| `reward_fn` | `Callable[[args, pred], float]` | required | Scores output; 1.0 = pass |
| `threshold` | `float` | `0.8` | Stop retrying when reward meets threshold |

## LangGraph StateGraph — Orchestration

State is a `TypedDict`. Use `Annotated[list[T], operator.add]` as the type for fields that should be merged (appended) across partial node returns — the standard pattern for `messages` and `results`.

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(TeamState)
graph.add_node("supervisor", supervisor_fn)
graph.add_node("agent_a", agent_a_fn)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_fn, {"agent_a": "agent_a", "done": END})
graph.add_edge("agent_a", "supervisor")
app = graph.compile()
```

## LangGraph Send — Parallel fan-out

```python
from langgraph.constants import Send

def split_task(state) -> list:
    return [Send("worker", {"task": state["task"], "subtask": st}) for st in state["subtasks"]]

graph.add_conditional_edges(START, split_task)  # spawns one "worker" node per subtask
```

## Human-in-the-loop

```python
from langgraph.checkpoint.memory import MemorySaver

app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["execute_action"])
config = {"configurable": {"thread_id": "task-001"}}
app.invoke(initial_state, config)  # pauses before "execute_action"
app.invoke(None, config)           # resume after human review
```

## Per-agent LM assignment

```python
supervisor_module.set_lm(dspy.LM("openai/gpt-4o"))       # stronger for routing
researcher.set_lm(dspy.LM("openai/gpt-4o-mini"))          # cheaper for workers
```

## Optimization

Optimize each agent independently (per-agent metric), then tune the full team (team metric). Wrap the LangGraph app in a `dspy.Module` to make it optimizable.

```python
# Per-agent
optimized_agent = dspy.MIPROv2(metric=agent_metric, auto="light").compile(
    agent, trainset=agent_trainset
)

# Team-level — wrap LangGraph app as a Module, then optimize
optimized_team = dspy.MIPROv2(metric=team_metric, auto="medium").compile(
    TeamModule(), trainset=team_trainset
)
```

`auto="light"` for individual agents; `auto="medium"` for team runs where each evaluation invokes the full graph.
