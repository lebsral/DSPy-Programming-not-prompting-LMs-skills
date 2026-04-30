---
name: dspy-rlm
description: Recursive Language Model (dspy.RLM) that explores large contexts via a sandboxed Python REPL -- the LM writes code, queries sub-LMs, and iterates until it produces a final answer. Use when your input is too large for the context window, the model needs to explore data iteratively, you need recursive self-refinement with code execution, or you have research-style tasks requiring programmatic investigation. Also: recursive language model, iterative exploration with LLM, model explores data in REPL, agent that keeps digging until it finds the answer, REPL-based reasoning, explore then answer pattern, deep research agent, when one pass is not enough.
---

# Iterative Self-Refinement with dspy.RLM

Guide the user through using DSPy's RLM (Recursive Language Model) module. RLM lets the LM explore data programmatically in a sandboxed Python REPL, writing code to examine inputs, querying sub-LMs for semantic analysis, and iterating until it produces a final answer.

> **Experimental.** RLM is marked as experimental in DSPy. The API may change in future releases.

## Step 1: Gather context

Before building with RLM, clarify:

1. **What data will the LM explore?** Large text corpus, log files, structured data dumps, multi-document collections?
2. **What kind of answer do you need?** Free-text summary, structured extraction (counts, lists), or a specific value?
3. **Does the task need external tools?** If the LM needs to call APIs or databases during exploration, you can pass custom tool functions.
4. **What LM are you using?** RLM works best with strong reasoning models (GPT-4o, Claude Sonnet) as the main LM; cheaper models can handle sub-queries.

## What is RLM

`dspy.RLM` implements the Recursive Language Models approach ([Zhang, Kraska, Khattab 2025](https://arxiv.org/abs/2512.24601)). Instead of feeding the full input context into the LM's prompt, RLM:

1. **Shows metadata only** -- the LM receives type, length, and a preview of each input, not the full content.
2. **Lets the LM write code** -- the LM generates Python in a sandboxed REPL to search, filter, aggregate, or transform the data.
3. **Executes in a sandbox** -- code runs in a WASM-based Python interpreter (Pyodide via Deno) for safety.
4. **Supports sub-LM queries** -- the LM can call `llm_query(prompt)` to do semantic analysis on slices of the data.
5. **Iterates** -- the LM loops through code-execute-observe cycles until it calls `SUBMIT(output)` with a final answer.

This makes RLM ideal for tasks where the input is too large for the context window, or where the LM needs to programmatically explore the data to find the answer.

## When to use RLM

| Scenario | Why RLM helps |
|----------|---------------|
| Very large input contexts (100K+ chars) | LM sees metadata, explores programmatically instead of stuffing the context |
| Data exploration tasks | LM writes code to search, filter, aggregate |
| Tasks requiring code + reasoning | Built-in REPL combines computation with LM reasoning |
| Multi-step analysis over structured data | LM can iterate, inspect intermediate results, refine approach |

When RLM is **not** the right fit:
- Simple input-output tasks -- use `dspy.Predict` or `dspy.ChainOfThought`
- Tasks that need external tool use (APIs, databases) -- use `dspy.ReAct`
- Quick classification or extraction -- overhead of the REPL loop is unnecessary

## Prerequisites

RLM's default sandbox requires Deno for the Pyodide WASM interpreter:

```bash
curl -fsSL https://deno.land/install.sh | sh
```

## Basic usage

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

rlm = dspy.RLM("context, query -> answer")
result = rlm(
    context="...very long document or dataset...",
    query="What is the total revenue for Q3?",
)
print(result.answer)
```

The LM will:
1. See a preview of `context` (type, length, first/last chars).
2. Write Python code to search and parse the content.
3. Optionally call `llm_query()` for semantic questions about slices.
4. Call `SUBMIT(answer)` when ready.

## Constructor parameters

```python
dspy.RLM(
    signature,              # str | Signature -- required, defines inputs/outputs
    max_iterations=20,      # max REPL interaction loops
    max_llm_calls=50,       # max sub-LM query calls per execution
    max_output_chars=10_000,# max chars from REPL output per step
    verbose=False,          # enable detailed execution logging
    tools=None,             # list[Callable] -- custom tool functions
    sub_lm=None,            # dspy.LM -- separate (cheaper) LM for sub-queries
    interpreter=None,       # custom CodeInterpreter (defaults to PythonInterpreter)
)
```

## Built-in tools available inside the REPL

When the LM writes code in the sandbox, these functions are available:

| Function | Purpose |
|----------|---------|
| `llm_query(prompt)` | Query the sub-LM with a prompt (up to ~500K chars) |
| `llm_query_batched(prompts)` | Concurrent multi-prompt queries |
| `print()` | Display REPL output (required to see results) |
| `SUBMIT(output)` | End execution and return the final answer |

## Using a cheaper sub-LM

Route expensive reasoning to a strong model while using a cheap model for sub-queries:

```python
main_lm = dspy.LM("openai/gpt-4o")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
cheap_lm = dspy.LM("openai/gpt-4o-mini")  # or any cheaper model

dspy.configure(lm=main_lm)

rlm = dspy.RLM("data, query -> summary", sub_lm=cheap_lm)
result = rlm(data=large_dataset, query="Summarize the key trends")
```

## Typed outputs

RLM supports DSPy's typed output fields, just like other modules:

```python
rlm = dspy.RLM("logs -> error_count: int, critical_errors: list[str]")
result = rlm(logs=server_logs)
print(result.error_count)        # int
print(result.critical_errors)    # list[str]
```

## Custom tools

Pass additional Python functions that the LM can call inside the sandbox:

```python
def fetch_metadata(doc_id: str) -> str:
    """Look up metadata for a document by ID."""
    return database.get_metadata(doc_id)

rlm = dspy.RLM("documents, query -> answer", tools=[fetch_metadata])
result = rlm(documents=docs, query="Which document has the latest revision?")
```

## Inspecting the trajectory

After execution, inspect the code-execute-observe steps the LM took:

```python
result = rlm(context=data, query="Find the outlier values")

for step in result.trajectory:
    print(f"Code:\n{step['code']}")
    print(f"Output:\n{step['output']}\n")
```

This is useful for debugging, understanding the LM's exploration strategy, and building trust in the result.

## Async execution

```python
async def process():
    result = await rlm.aforward(context=data, query="Summarize findings")
    return result.answer
```

## How RLM differs from other refinement approaches

| Approach | Mechanism | Best for |
|----------|-----------|----------|
| **RLM** | LM writes code in a REPL to explore data, calls sub-LMs, iterates until `SUBMIT()` | Large contexts, data exploration, programmatic analysis |
| **Refine** (`dspy.Refine`) | Retry with feedback from a reward function until score threshold is met | Improving a single output with a known quality metric |
| **Best-of-N** | Generate N candidates, pick the best by a metric | When you want diversity of attempts and can score them |
| **ChainOfThought** | Single-pass step-by-step reasoning | Standard tasks that fit in context |
| **Assertions** (`dspy.Assert`/`dspy.Suggest`) | Constraint-based retry with error messages | Enforcing hard/soft rules on outputs |

Key difference: RLM gives the LM a **code execution environment** to actively explore the input, rather than just re-prompting with feedback. The LM decides its own exploration strategy.

## Thread safety

RLM instances with custom interpreters are not thread-safe. For concurrent usage, create separate instances or use the default `PythonInterpreter`.

## Gotchas

- **Forgetting Deno installation.** RLM's default PythonInterpreter uses a Pyodide WASM sandbox that requires Deno. If Deno is not installed, you get a cryptic subprocess error. Always check `deno --version` before running RLM code.
- **Using RLM for tasks that fit in context.** Claude defaults to RLM when asked about "iterative refinement" even for short inputs. RLM adds significant overhead (multiple REPL loops, sub-LM calls). For inputs under ~50K chars, use `dspy.ChainOfThought` or `dspy.Predict` instead.
- **Forgetting `print()` in REPL code.** The LM must `print()` values to see REPL output -- assignments alone produce no visible result. If the LM's exploration seems stuck, check the trajectory for code that computes but never prints.
- **Setting `max_iterations` too low for complex tasks.** Claude tends to set `max_iterations=5` for brevity. RLM defaults to 20 for a reason -- complex data exploration often needs 10-15 iterations. Only lower it for simple lookups.
- **Not using `sub_lm` for cost control.** Every `llm_query()` call inside the REPL uses the main LM by default. For large-context tasks with many sub-queries, this gets expensive fast. Always set `sub_lm` to a cheaper model for semantic analysis calls.

## Additional resources

- [dspy.RLM API docs](https://dspy.ai/api/modules/RLM)
- For API details, see [reference.md](reference.md)
- For worked examples, see [examples.md](examples.md)

## Cross-references

> Install any skill: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill <name>`

- **Refine** for reward-function-based retry loops -- see `/dspy-refine`
- **Best-of-N** for generating and scoring multiple candidates -- see `/dspy-best-of-n`
- **Improving accuracy** with optimizers and evaluation -- see `/ai-improving-accuracy`
- **Building pipelines** with multi-step module composition -- see `/ai-building-pipelines`
- **Install `/ai-do` if you do not have it** — it routes any AI problem to the right skill and is the fastest way to work: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do`
