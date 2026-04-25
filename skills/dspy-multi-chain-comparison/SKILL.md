---
name: dspy-multi-chain-comparison
description: "Use when you want higher accuracy by generating multiple reasoning chains and selecting the best answer — trading speed for quality on critical outputs. Common scenarios: high-stakes decisions where you want multiple reasoning paths compared, classification tasks where one chain of thought isn't reliable enough, improving accuracy by generating several answers and selecting the best-reasoned one, or tasks where different reasoning approaches yield different answers. Related: ai-reasoning, ai-improving-accuracy, dspy-chain-of-thought. Also: \"dspy.MultiChainComparison\", \"compare multiple reasoning chains\", \"select best reasoning path\", \"multi-path reasoning\", \"vote across chain-of-thought outputs\", \"more reliable than single CoT\", \"deliberation for hard problems\", \"when one reasoning chain isn't enough\", \"robust reasoning through comparison\", \"ensemble reasoning\", \"trade speed for accuracy on critical tasks\"."
---

# Get Better Answers by Comparing Multiple Reasoning Chains

Guide the user through using `dspy.MultiChainComparison` to improve answer quality. Instead of relying on a single chain of thought, this module generates several independent reasoning chains and then selects the best final answer by comparing them.

## What is MultiChainComparison

`dspy.MultiChainComparison` is a DSPy module that:

1. **Generates multiple chains of thought** -- each one reasons through the problem independently
2. **Compares the candidates** -- a final comparison step evaluates all chains and picks the best answer
3. **Returns a single answer** -- the output looks the same as any other DSPy module

Think of it as getting multiple opinions from different experts, then having a judge pick the most convincing one. The diversity of reasoning paths surfaces better answers than any single chain alone.

## When to use MultiChainComparison

Use it when:

- **Quality matters more than speed** -- you can afford extra LM calls for a better answer
- **Tasks have genuine ambiguity** -- multiple valid approaches exist and you want the best one
- **Single CoT is unreliable** -- the model sometimes reasons poorly and you want redundancy
- **High-stakes decisions** -- recommendations, diagnoses, critical analysis where being wrong is costly

Do NOT use it when:

- **Latency is critical** -- it makes multiple LM calls (one per chain + one comparison), so it is slower than a single `ChainOfThought`
- **The task is straightforward** -- simple classification, extraction, or lookup does not benefit from multiple chains
- **Cost is a hard constraint** -- each chain is a separate LM call, so costs scale linearly with the number of chains
- **You need deterministic output** -- the comparison step adds variability

## Basic usage

`MultiChainComparison` works with any signature, just like `ChainOfThought`:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Inline signature -- same as you'd use with ChainOfThought
recommend = dspy.MultiChainComparison("problem -> recommendation")
result = recommend(problem="We need to migrate from MongoDB to PostgreSQL for our 50GB dataset with complex joins")
print(result.recommendation)
```

With a class-based signature:

```python
import dspy

class TechRecommendation(dspy.Signature):
    """Recommend the best technical approach for this problem."""
    problem: str = dspy.InputField(desc="Technical problem or decision to make")
    constraints: str = dspy.InputField(desc="Budget, timeline, or technical constraints")
    recommendation: str = dspy.OutputField(desc="The recommended approach with justification")

recommend = dspy.MultiChainComparison(TechRecommendation)
result = recommend(
    problem="Our API response times are over 2 seconds under load",
    constraints="Small team, no budget for new infrastructure",
)
print(result.recommendation)
```

## How it works internally

When you call a `MultiChainComparison` module, DSPy does the following:

1. **Runs N independent `ChainOfThought` calls** -- each produces its own `reasoning` and output fields
2. **Formats all completions** -- collects the reasoning and answers from each chain
3. **Runs a comparison step** -- a final LM call sees all candidate chains and selects the best answer

The comparison step is the key differentiator. Rather than picking randomly or voting, the model actively evaluates the quality of each reasoning chain before choosing.

```
Input --> CoT Chain 1 --> reasoning_1 + answer_1 --|
      --> CoT Chain 2 --> reasoning_2 + answer_2 --|--> Comparison --> best answer
      --> CoT Chain 3 --> reasoning_3 + answer_3 --|
```

## Configuring the number of chains

By default, `MultiChainComparison` generates 3 chains. You can adjust this with the `M` parameter:

```python
# Fewer chains -- faster, cheaper, but less diversity
quick = dspy.MultiChainComparison("question -> answer", M=2)

# Default -- good balance
default = dspy.MultiChainComparison("question -> answer", M=3)

# More chains -- better quality ceiling, but slower and more expensive
thorough = dspy.MultiChainComparison("question -> answer", M=5)
```

Guidelines for choosing M:

| M value | LM calls | Best for |
|---------|----------|----------|
| 2 | 3 (2 chains + 1 comparison) | Slight quality boost over single CoT |
| 3 | 4 (3 chains + 1 comparison) | Good default, balances quality and cost |
| 5 | 6 (5 chains + 1 comparison) | High-stakes tasks where accuracy is critical |
| 7+ | 8+ | Diminishing returns for most tasks |

## Using MultiChainComparison in a module

Wrap it in a `dspy.Module` to combine with other steps:

```python
import dspy
from typing import Literal

class RiskAssessment(dspy.Signature):
    """Assess the risk level of this proposed change."""
    change_description: str = dspy.InputField(desc="What is being changed")
    system_context: str = dspy.InputField(desc="The system being modified")
    risk_level: Literal["low", "medium", "high", "critical"] = dspy.OutputField()
    risk_factors: str = dspy.OutputField(desc="Key risks identified")
    mitigation: str = dspy.OutputField(desc="Recommended mitigation steps")

class ChangeReviewer(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("change_description -> change_type: str")
        self.assess = dspy.MultiChainComparison(RiskAssessment, M=3)

    def forward(self, change_description, system_context):
        change_type = self.classify(change_description=change_description).change_type
        result = self.assess(
            change_description=f"[{change_type}] {change_description}",
            system_context=system_context,
        )
        return dspy.Prediction(
            change_type=change_type,
            risk_level=result.risk_level,
            risk_factors=result.risk_factors,
            mitigation=result.mitigation,
        )

# Usage
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

reviewer = ChangeReviewer()
result = reviewer(
    change_description="Drop the users_v1 table and migrate all queries to users_v2",
    system_context="Production e-commerce platform with 10k daily active users",
)
print(f"Risk: {result.risk_level}")
print(f"Factors: {result.risk_factors}")
print(f"Mitigation: {result.mitigation}")
```

## Cost and latency tradeoffs

MultiChainComparison trades speed and cost for quality. Here is a rough comparison with M=3:

| Aspect | ChainOfThought | MultiChainComparison (M=3) |
|--------|---------------|---------------------------|
| LM calls | 1 | 4 (3 chains + 1 comparison) |
| Latency | 1x | ~3-4x (chains can run in parallel internally) |
| Cost | 1x | ~4x |
| Quality | Good | Better on ambiguous/complex tasks |

Strategies to manage cost:

- **Use a cheaper model for chains, expensive model for comparison** -- the comparison step benefits most from a strong model:

```python
cheap_lm = dspy.LM("openai/gpt-4o-mini")
expensive_lm = dspy.LM("openai/gpt-4o")

dspy.configure(lm=cheap_lm)  # default for chains

pipeline = ChangeReviewer()
# The comparison predict inside MultiChainComparison can be set separately
# by accessing the internal predict module
```

- **Use MultiChainComparison selectively** -- route only hard tasks through it:

```python
class AdaptiveReasoner(dspy.Module):
    def __init__(self):
        self.classify_difficulty = dspy.Predict("question -> difficulty: str")
        self.fast = dspy.ChainOfThought("question -> answer")
        self.thorough = dspy.MultiChainComparison("question -> answer", M=3)

    def forward(self, question):
        difficulty = self.classify_difficulty(question=question).difficulty.lower()
        if "hard" in difficulty or "complex" in difficulty:
            return self.thorough(question=question)
        return self.fast(question=question)
```

## Optimizing MultiChainComparison

MultiChainComparison modules are optimizable like any other DSPy module. Optimizers tune the prompts for both the chain generation and comparison steps:

```python
def quality_metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

program = dspy.MultiChainComparison("question -> answer", M=3)

optimizer = dspy.BootstrapFewShot(metric=quality_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(program, trainset=trainset)

# Save for production
optimized.save("optimized_mcc.json")
```

For best results with MIPROv2:

```python
optimizer = dspy.MIPROv2(metric=quality_metric, auto="medium")
optimized = optimizer.compile(program, trainset=trainset)
```

## When NOT to use MultiChainComparison

Pick a simpler alternative when:

| Situation | Use instead |
|-----------|-------------|
| Simple classification or extraction | `dspy.Predict` |
| Needs reasoning but latency matters | `dspy.ChainOfThought` |
| Math or computation tasks | `dspy.ProgramOfThought` |
| Need tool use or API calls | `dspy.ReAct` |
| Want retries with self-correction | `dspy.Assert` + `ChainOfThought` |

MultiChainComparison is most valuable when the problem genuinely benefits from diverse perspectives -- not when there is a single clearly correct approach.

## Cross-references

- **ChainOfThought** is the single-chain version and the right default -- see `/dspy-chain-of-thought`
- **Reasoning strategies** including when to pick MultiChainComparison vs other approaches -- see `/ai-reasoning`
- **Improving accuracy** with evaluation and optimization -- see `/ai-improving-accuracy`
- For worked examples, see [examples.md](examples.md)
