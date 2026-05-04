---
name: dspy-utils
description: Use when you need DSPy infrastructure - caching control, debugging with inspect_history, saving/loading optimized programs, or runtime validation with Refine/BestOfN. Common scenarios - controlling the cache to avoid stale results, debugging with inspect_history to see raw prompts, saving and loading optimized programs, or validating outputs with reward functions. For streaming see /dspy-streaming, for async see /dspy-async, for MCP see /dspy-mcp. Related - ai-tracing-requests, ai-serving-apis, ai-monitoring, dspy-streaming, dspy-async, dspy-mcp. Also used for dspy.inspect_history, dspy.settings.configure, cache control in DSPy, save and load DSPy program, debug DSPy prompts, see what DSPy sent to the model, DSPy program serialization, production DSPy utilities, clear DSPy cache, view prompt history.
---

# DSPy Utilities: Caching, Debugging, Save/Load, and Validation

Guide the user through DSPy's utility functions -- controlling caching, debugging calls, persisting optimized programs, and enforcing runtime constraints with reward functions.

> **Looking for streaming, async, or MCP?** These have dedicated skills now:
> - Streaming tokens to a UI -- see `/dspy-streaming`
> - Async execution and FastAPI -- see `/dspy-async`
> - MCP server integration -- see `/dspy-mcp`

## Step 1: Which utility do you need?

Ask the user before diving in:

1. **What are you trying to do?** Debug a failing program, save/load an optimized program, control caching, or validate outputs with reward functions?
2. **Is this for development or production?** Development needs (debugging, cache control) differ from production needs (save/load, validation).

Then jump to the relevant section below.

## 2. configure_cache -- controlling cache behavior

DSPy caches LM responses by default to reduce costs and speed up development. Use `dspy.configure_cache` to control this globally.

```python
# Disable caching entirely
dspy.configure_cache(enable=False)

# Re-enable caching
dspy.configure_cache(enable=True)
```

### Per-LM cache control

You can also control caching per LM instance:

```python
# This LM never caches
lm_no_cache = dspy.LM("openai/gpt-4o-mini", cache=False)

# This LM caches (default)
lm_cached = dspy.LM("openai/gpt-4o-mini", cache=True)
```

### When to disable caching

- **Generating diverse outputs** -- when you need different responses for the same prompt (e.g., data generation)
- **Testing real latency** -- cache hits are instant, which skews benchmarks
- **Streaming** -- caching may interfere with streaming behavior in some configurations

Cache is stored locally on disk. Identical calls (same prompt, parameters, model) return cached results with no API call.

**When NOT to disable caching:** During optimization runs -- optimizers rely heavily on cache to avoid redundant LM calls. Disabling cache globally during optimization dramatically increases cost and time.

## 3. inspect_history -- debugging LM calls

`dspy.inspect_history` shows the raw prompts and responses from recent LM calls. This is the single most useful debugging tool in DSPy.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

classify = dspy.Predict("text -> label")
classify(text="Great product!")

# See what was actually sent to and received from the LM
dspy.inspect_history(n=1)  # Show last 1 call
dspy.inspect_history(n=3)  # Show last 3 calls
```

### What inspect_history shows

- The full prompt sent to the LM (including system message, few-shot demos, instructions)
- The raw LM response
- Which adapter formatted the prompt (ChatAdapter, JSONAdapter, etc.)

### Debugging workflow

1. Run your program on a failing input
2. Call `dspy.inspect_history(n=1)` to see the last LM call
3. Check if the prompt makes sense -- are the instructions clear? Are few-shot demos relevant?
4. Check the raw response -- did the LM follow the format? Did it hallucinate?
5. Adjust your signature, module, or optimization strategy based on what you see

### Verbose logging

For more detailed tracing, configure DSPy with an empty trace list:

```python
dspy.configure(lm=lm, trace=[])
```

You can also print a module to see its structure:

```python
print(my_program)  # Shows module tree with all sub-modules and signatures
```

## 4. save/load -- persisting optimized programs

After optimizing a DSPy program, save its learned state (few-shot demos, instructions) for production use.

### Save

```python
# After optimization
optimized = optimizer.compile(my_program, trainset=trainset)
optimized.save("optimized_program.json")
```

### Load

```python
# In production -- create a fresh instance, then load state
program = MyProgram()
program.load("optimized_program.json")

# Use it
result = program(question="What is DSPy?")
```

### What gets saved

- Few-shot demonstrations discovered by optimizers
- Optimized instructions (from MIPROv2, GEPA, etc.)
- Any state tracked by `dspy.Predict` modules

### What does NOT get saved

- Python logic in `forward()` -- that's your code, it must exist at load time
- Model weights (unless you used `BootstrapFinetune`)
- LM configuration -- you must call `dspy.configure()` before loading

### Production deployment pattern

```python
import dspy

class MyPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> category")
        self.respond = dspy.ChainOfThought("text, category -> response")

    def forward(self, text):
        cat = self.classify(text=text)
        return self.respond(text=text, category=cat.category)

# --- Optimization (run once) ---
# optimizer = dspy.MIPROv2(metric=metric, auto="medium")
# optimized = optimizer.compile(MyPipeline(), trainset=trainset)
# optimized.save("pipeline_v1.json")

# --- Production (run on every request) ---
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

pipeline = MyPipeline()
pipeline.load("pipeline_v1.json")

result = pipeline(text="How do I reset my password?")
```

## 5. dspy.Refine and dspy.BestOfN -- reward-based output validation

Use `dspy.Refine` to wrap any module and retry until a reward function returns a score meeting a threshold. This replaced `dspy.Assert`/`dspy.Suggest` in DSPy 3.x:

```python
import dspy

qa = dspy.ChainOfThought("question -> answer")

def answer_reward(args, pred):
    """Score answer quality. Returns 0.0-1.0."""
    if not pred.answer.strip():
        return 0.0
    if len(pred.answer.split()) < 5:
        return 0.5  # soft penalty for short answers
    return 1.0

validated_qa = dspy.Refine(
    module=qa,
    N=3,
    reward_fn=answer_reward,
    threshold=1.0,
)

result = validated_qa(question="What is DSPy?")
```

- **`dspy.Refine`** -- retries with feedback from the reward function until threshold is met or N attempts exhausted. Use when later attempts can improve based on earlier failures.
- **`dspy.BestOfN`** -- runs N independent attempts and returns the best-scoring one. Use when attempts are independent and cross-attempt feedback would not help.

For detailed patterns and examples, see **`/dspy-refine`** and **`/dspy-best-of-n`**.

## Gotchas

1. **`save()` does not persist `forward()` logic** -- only learned state (demos, instructions) is saved. The class definition must exist in your production code at load time.
2. **Must `dspy.configure()` before `load()`** -- loading a saved program before configuring the LM causes silent failures where the program runs but uses no LM (or the wrong one).
3. **`inspect_history` shows cached calls too** -- after a cache hit, `inspect_history` still shows the call, but the prompt may look different from what was originally sent. Disable cache if you need exact prompt inspection.
4. **Claude disables caching during optimization.** Do NOT disable cache globally during optimizer runs -- optimizers rely heavily on cache to avoid redundant LM calls. Disabling cache during optimization dramatically increases cost and time.

## Additional resources

- [DSPy saving/loading guide](https://dspy.ai/tutorials/saving/)
- For API details, see [reference.md](reference.md)
- For worked examples, see [examples.md](examples.md)

## Cross-references

> Install any skill: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill <name>`

- **Streaming** tokens to a UI -- see `/dspy-streaming`
- **Async execution** and FastAPI -- see `/dspy-async`
- **MCP server** integration -- see `/dspy-mcp`
- **`/dspy-lm`** -- Configure language models, per-LM caching, `inspect_history` on LM instances
- **`/dspy-modules`** -- Build composable programs with `dspy.Module`, save/load patterns
- **`/ai-tracing-requests`** -- Production observability and tracing for DSPy programs
- **`/dspy-refine`** -- Refine patterns, reward functions, and iterative improvement
- **`/dspy-best-of-n`** -- BestOfN for independent sampling without cross-attempt feedback
- **`/ai-serving-apis`** -- Serve DSPy programs as web APIs
- **Install `/ai-do` if you do not have it** -- it routes any AI problem to the right skill and is the fastest way to work: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do`
