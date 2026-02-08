---
name: ai-generating-data
description: Generate synthetic training data when you don't have enough real examples. Use when you're starting from scratch with no data, need a proof of concept fast, have too few examples for optimization, can't use real customer data for privacy or compliance, need to fill gaps in edge cases, have unbalanced categories, added new categories, or changed your schema. Covers DSPy synthetic data generation, quality filtering, and bootstrapping from zero.
---

# Generate Synthetic Training Data

Guide the user through generating high-quality synthetic training data with DSPy. This solves the "I don't have data" problem that blocks every other AI workflow.

## When you need synthetic data

- **Cold start**: You're building a new feature and have zero labeled examples
- **Not enough for optimization**: You have 10-30 examples but optimizers need 200+
- **Privacy/compliance**: You can't use real customer data for training
- **Edge cases**: Your AI works on common inputs but fails on rare ones
- **Unbalanced categories**: Some categories have 500 examples, others have 10
- **New categories**: You added a category and have no examples for it
- **Schema changed**: Your input/output format changed, old data doesn't fit
- **Proof of concept**: PM wants a demo by Friday, no time to collect real data

## The core idea

Define a generator signature whose outputs match your task's input/output fields. Use an LM to produce examples. Filter for quality. Use for optimization.

Research shows this works surprisingly well:
- Optimized generator prompts match models trained on 100K+ human labels using only 10 gold labels (arXiv 2406.11706)
- DSPy-optimized Chain-of-Thought generation outperforms hand-written static templates (arXiv 2508.13930)

The key insight: the prompt used to *generate* data is a critical hyperparameter — optimizing it matters more than generating more data.

## Step 1: Define what an example looks like

Your generator's outputs should match your task's inputs and expected outputs.

```python
import dspy

# Your task — what the AI will do in production
class ClassifyTicket(dspy.Signature):
    """Classify a support ticket into a category."""
    ticket_text: str = dspy.InputField()
    category: str = dspy.OutputField()

# Generator — produces examples for your task
class GenerateTicketExample(dspy.Signature):
    """Generate a realistic support ticket with its correct category."""
    category: str = dspy.InputField(desc="the target category to generate an example for")
    ticket_text: str = dspy.OutputField(desc="a realistic support ticket for this category")
```

The generator's output fields become inputs to your task. Think of it as: "given what I want the answer to be, generate a realistic input."

### Multi-field tasks

If your task has multiple inputs or outputs, mirror all of them:

```python
# Task: extract structured data from text
class ExtractContact(dspy.Signature):
    """Extract contact info from a message."""
    message: str = dspy.InputField()
    name: str = dspy.OutputField()
    email: str = dspy.OutputField()
    phone: str = dspy.OutputField()

# Generator: produce realistic messages with known contact info
class GenerateContactExample(dspy.Signature):
    """Generate a realistic message that contains contact information."""
    name: str = dspy.InputField(desc="the person's name to embed in the message")
    email: str = dspy.InputField(desc="the email address to embed in the message")
    phone: str = dspy.InputField(desc="the phone number to embed in the message")
    message: str = dspy.OutputField(desc="a realistic message containing this contact info")
```

## Step 2: Write seed examples

Start with 5-10 hand-written examples. These anchor the generator's understanding of what "realistic" means for your domain.

```python
seeds = [
    dspy.Example(
        ticket_text="I was charged twice for my subscription this month. Order #4521.",
        category="billing"
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="The app crashes when I try to upload a profile photo on Android.",
        category="bug"
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="How do I export my data to CSV? I can't find the option anywhere.",
        category="how-to"
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="I'd love to see dark mode added. The white background hurts my eyes.",
        category="feature-request"
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="My account got locked after too many login attempts. Please help.",
        category="account"
    ).with_inputs("ticket_text"),
]
```

Even 5 seeds dramatically improve generation quality over zero.

## Step 3: Generate in batches

Two patterns depending on your LM provider:

### Pattern A: `n=N` batch generation

When your provider supports the `n` parameter (OpenAI does), this generates multiple completions in one call — faster and often more diverse:

```python
generator = dspy.Predict(GenerateTicketExample, n=20)
response = generator(category="billing")
examples = [
    dspy.Example(ticket_text=t, category="billing").with_inputs("ticket_text")
    for t in response.completions.ticket_text
]
```

### Pattern B: Loop generation

Works with any provider. More control over each example:

```python
examples = []
categories = ["billing", "bug", "how-to", "feature-request", "account"]

for category in categories:
    generator = dspy.Predict(GenerateTicketExample)
    for i in range(40):
        result = generator(category=category)
        examples.append(
            dspy.Example(ticket_text=result.ticket_text, category=category)
            .with_inputs("ticket_text")
        )

print(f"Generated {len(examples)} examples")
```

The `n` parameter isn't supported by all providers — use the loop pattern as a reliable fallback.

### Generation strategies

Pick the strategy that fits your gap:

**Category-driven** — generate N per category (fixes imbalance):

```python
for category in categories:
    for i in range(50):
        result = generator(category=category)
        examples.append(dspy.Example(ticket_text=result.ticket_text, category=category).with_inputs("ticket_text"))
```

**Seed-and-vary** — pass a seed example with a variation instruction:

```python
class GenerateVariation(dspy.Signature):
    """Generate a variation of this support ticket with a different tone and phrasing."""
    original_ticket: str = dspy.InputField(desc="the original ticket to vary")
    variation_type: str = dspy.InputField(desc="how to vary it: tone, length, complexity, or language")
    ticket_text: str = dspy.OutputField(desc="a new ticket with the same meaning but different style")

vary = dspy.Predict(GenerateVariation)
for seed in seeds:
    for variation in ["angry tone", "very brief", "verbose and detailed", "non-native English"]:
        result = vary(original_ticket=seed.ticket_text, variation_type=variation)
        examples.append(dspy.Example(ticket_text=result.ticket_text, category=seed.category).with_inputs("ticket_text"))
```

**Scenario-driven** — specify edge case scenarios:

```python
class GenerateScenarioTicket(dspy.Signature):
    """Generate a support ticket matching a specific scenario."""
    category: str = dspy.InputField()
    scenario: str = dspy.InputField(desc="the specific scenario to generate")
    ticket_text: str = dspy.OutputField()

gen = dspy.Predict(GenerateScenarioTicket)
scenarios = [
    ("billing", "customer charged in wrong currency"),
    ("billing", "refund for a cancelled subscription"),
    ("bug", "issue only happens on slow network connections"),
    ("bug", "multi-step reproduction involving two features"),
    ("how-to", "customer is non-technical and confused by jargon"),
]
for category, scenario in scenarios:
    result = gen(category=category, scenario=scenario)
    examples.append(dspy.Example(ticket_text=result.ticket_text, category=category).with_inputs("ticket_text"))
```

**Difficulty-driven** — generate easy, medium, hard examples separately:

```python
class GenerateByDifficulty(dspy.Signature):
    """Generate a support ticket at a specific difficulty level for classification."""
    category: str = dspy.InputField()
    difficulty: str = dspy.InputField(desc="easy (clear-cut), medium (some ambiguity), or hard (could be multiple categories)")
    ticket_text: str = dspy.OutputField()

gen = dspy.Predict(GenerateByDifficulty)
for category in categories:
    for difficulty in ["easy", "medium", "hard"]:
        for i in range(15):
            result = gen(category=category, difficulty=difficulty)
            examples.append(dspy.Example(ticket_text=result.ticket_text, category=category).with_inputs("ticket_text"))
```

**Diversity trick** — add a random `sindex` field to push the LM toward varied outputs:

```python
import random

class GenerateDiverse(dspy.Signature):
    """Generate a unique and realistic support ticket."""
    category: str = dspy.InputField()
    sindex: str = dspy.InputField(desc="a unique seed index for diversity")
    ticket_text: str = dspy.OutputField()

gen = dspy.Predict(GenerateDiverse)
for category in categories:
    for i in range(50):
        result = gen(category=category, sindex=str(random.randint(0, 1_000_000)))
        examples.append(dspy.Example(ticket_text=result.ticket_text, category=category).with_inputs("ticket_text"))
```

The random `sindex` prevents the LM from falling into repetitive patterns.

## Step 4: Filter for quality

Generated data always contains some bad examples. Filter aggressively — aim to generate 2-3x what you need and keep ~50%.

### Simple: metric-based filtering

Run each generated example through your task program and check with your metric:

```python
program = dspy.ChainOfThought(ClassifyTicket)
filtered = []

for ex in examples:
    pred = program(**ex.inputs())
    if metric(ex, pred):
        filtered.append(ex)

print(f"Kept {len(filtered)}/{len(examples)} ({100*len(filtered)//len(examples)}%)")
```

This works when your program is already decent — it filters out examples that are confusing or mislabeled.

### Robust: LM-based assessment

Use a separate assessment step to check realism and correctness:

```python
class AssessExample(dspy.Signature):
    """Is this a realistic and correctly labeled example?"""
    ticket_text: str = dspy.InputField()
    category: str = dspy.InputField()
    is_realistic: bool = dspy.OutputField(desc="true if this looks like a real support ticket")
    is_correctly_labeled: bool = dspy.OutputField(desc="true if the category matches the ticket")

assessor = dspy.Predict(AssessExample)
filtered = []

for ex in examples:
    result = assessor(ticket_text=ex.ticket_text, category=ex.category)
    if result.is_realistic and result.is_correctly_labeled:
        filtered.append(ex)

print(f"Kept {len(filtered)}/{len(examples)} ({100*len(filtered)//len(examples)}%)")
```

### Quality gates with `dspy.Suggest`

For tighter integration, build quality checks into the generator itself. When a `Suggest` constraint fails, DSPy retries the generation:

```python
class QualityGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateTicketExample)
        self.assess = dspy.Predict(AssessExample)

    def forward(self, category):
        result = self.generate(category=category)
        assessment = self.assess(ticket_text=result.ticket_text, category=category)
        dspy.Suggest(assessment.is_realistic, "Generated ticket should be realistic")
        dspy.Suggest(assessment.is_correctly_labeled, "Category label should be correct")
        return result

generator = QualityGenerator()
# DSPy retries generation when Suggest constraints fail
```

### Check for duplicates

Remove near-duplicates to keep your dataset diverse:

```python
seen = set()
unique = []
for ex in filtered:
    # Normalize and check
    key = ex.ticket_text.strip().lower()
    if key not in seen:
        seen.add(key)
        unique.append(ex)

print(f"Removed {len(filtered) - len(unique)} near-duplicates")
filtered = unique
```

## Step 5: Optimize the generator itself (advanced)

Research (arXiv 2406.11706) shows that optimizing the prompt used to generate data dramatically improves downstream quality. This is meta-optimization: optimizing the generator so it produces better training data.

```python
class DataGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateTicketExample)

    def forward(self, category):
        return self.generate(category=category)

# Define a metric that measures generated data quality
def generator_metric(example, prediction, trace=None):
    # Check if a downstream classifier gets the right answer on this generated example
    classifier = dspy.Predict(ClassifyTicket)
    task_example = dspy.Example(ticket_text=prediction.ticket_text, category=example.category).with_inputs("ticket_text")
    task_pred = classifier(**task_example.inputs())
    return task_pred.category.lower() == example.category.lower()

# Optimize the generator's prompts
optimizer = dspy.BootstrapFewShot(metric=generator_metric)
optimized_generator = optimizer.compile(DataGenerator(), trainset=seeds)

# Now generate with the optimized generator
better_examples = []
for category in categories:
    for i in range(50):
        result = optimized_generator(category=category)
        better_examples.append(
            dspy.Example(ticket_text=result.ticket_text, category=category).with_inputs("ticket_text")
        )
```

This closes the loop: better generator prompts produce better data, which produces better task programs.

## Step 6: Use generated data for optimization

Full pipeline: generate, filter, split, optimize, evaluate.

```python
import random
from dspy.evaluate import Evaluate

# Shuffle and split
random.shuffle(filtered)
split = int(len(filtered) * 0.8)
trainset = filtered[:split]
devset = filtered[split:]

print(f"Train: {len(trainset)}, Dev: {len(devset)}")

# Configure your task LM (can be cheaper than the generator LM)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Build and optimize your task program
program = dspy.ChainOfThought(ClassifyTicket)

optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized = optimizer.compile(program, trainset=trainset)

# Evaluate
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
score = evaluator(optimized)
print(f"Score on synthetic dev set: {score:.1f}%")

# Save
optimized.save("optimized_program.json")
```

If you have even a small number of real examples, use them as the dev set instead — real data gives more trustworthy evaluation.

## Common scenarios

**Cold start** — zero real data. Write 5-10 seeds. Generate 200+ synthetic examples across all categories. Filter and optimize. See [examples.md](examples.md) for a full walkthrough.

**Edge case gaps** — your AI works at 85% but fails on specific scenarios. Run error analysis, identify the failure patterns, then use scenario-driven generation targeting those gaps. Re-optimize with the augmented dataset.

**Privacy/compliance** — can't use real customer data. Generate synthetic examples with realistic patterns but no PII. Validate with domain-specific assessments. The `dspy.Suggest` quality gate pattern ensures generated data meets your standards.

**New categories** — added a category with no examples. Use category-driven generation to produce 50+ examples for the new category, then retrain.

**Rebalancing** — some categories have 500 examples, others have 10. Generate more for underrepresented categories until all are roughly balanced.

**Schema changed** — your input/output format changed. Generate new examples matching the new schema rather than manually converting old data.

## Tips and pitfalls

- **Always validate generated data** — LMs produce plausible but wrong labels. Filter aggressively.
- **Mix synthetic with real data when available** — even 20 real examples mixed in improve quality significantly.
- **Use a stronger model to generate, cheaper model for your task** — e.g., generate with GPT-4o, run your task on GPT-4o-mini.
- **Generate more than you need** — aim for 2-3x your target, keep ~50% after filtering.
- **Check for duplicates** — LMs tend to repeat themselves, especially without the diversity trick.
- **Iterate** — generate, optimize, evaluate, identify gaps, generate more for gaps.
- **Don't trust synthetic eval scores blindly** — if possible, validate final quality on real data.
- **The `n` parameter for batch generation isn't supported by all providers** — use the loop pattern as a reliable fallback.

## Additional resources

- For end-to-end worked examples (cold start, edge cases, privacy), see [examples.md](examples.md)
- Use `/ai-improving-accuracy` to measure and improve your optimized program
- Use `/ai-fine-tuning` once you have enough generated data for weight optimization
- Use `/ai-kickoff` to scaffold a project, then fill data with this skill
