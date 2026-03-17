# dspy-bootstrap-finetune -- Worked Examples

## Example 1: Fine-tuning a small model from a large teacher

Distill a GPT-4o teacher into a GPT-4o-mini student for a sentiment classification task. The teacher generates high-quality reasoning traces, and the student learns to replicate them at 1/30th the inference cost.

```python
import dspy
from dspy.evaluate import Evaluate

# --- Define the task ---

class SentimentClassify(dspy.Signature):
    """Classify the sentiment of a product review."""
    review: str = dspy.InputField(desc="Product review text")
    sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")


def metric(example, prediction, trace=None):
    return prediction.sentiment.strip().lower() == example.sentiment.strip().lower()


# --- Prepare data ---
# In practice, load from a file or database. Need 500+ examples.

import json

with open("reviews_labeled.json") as f:
    raw = json.load(f)

examples = [
    dspy.Example(review=r["review"], sentiment=r["sentiment"]).with_inputs("review")
    for r in raw
]

# Split: 80% train, 10% dev, 10% test
n = len(examples)
trainset = examples[: int(n * 0.8)]
devset = examples[int(n * 0.8) : int(n * 0.9)]
testset = examples[int(n * 0.9) :]

print(f"Train: {len(trainset)}, Dev: {len(devset)}, Test: {len(testset)}")

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)


# --- Step 1: Build and optimize the teacher ---

teacher_lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=teacher_lm)

teacher = dspy.ChainOfThought(SentimentClassify)

# Optimize the teacher's prompts first for higher-quality traces
prompt_optimizer = dspy.MIPROv2(metric=metric, auto="medium")
teacher_optimized = prompt_optimizer.compile(teacher, trainset=trainset)

teacher_score = evaluator(teacher_optimized)
print(f"Teacher (GPT-4o, prompt-optimized): {teacher_score:.1f}%")


# --- Step 2: Measure the untuned student baseline ---

student_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=student_lm)

student_baseline = dspy.ChainOfThought(SentimentClassify)
baseline_score = evaluator(student_baseline)
print(f"Student baseline (GPT-4o-mini, no tuning): {baseline_score:.1f}%")


# --- Step 3: Fine-tune the student on the teacher's traces ---

student = dspy.ChainOfThought(SentimentClassify)

ft_optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=24)
student_finetuned = ft_optimizer.compile(
    student,
    trainset=trainset,
    teacher=teacher_optimized,
)

finetuned_score = evaluator(student_finetuned)
print(f"Student fine-tuned (GPT-4o-mini, distilled): {finetuned_score:.1f}%")


# --- Step 4: Compare all three on the held-out test set ---

test_evaluator = Evaluate(
    devset=testset, metric=metric, num_threads=4, display_progress=True
)

print("\n--- Test set results ---")
print(f"Teacher (GPT-4o):                  {test_evaluator(teacher_optimized):.1f}%")
print(f"Student baseline (GPT-4o-mini):    {test_evaluator(student_baseline):.1f}%")
print(f"Student fine-tuned (GPT-4o-mini):  {test_evaluator(student_finetuned):.1f}%")


# --- Step 5: Save for production ---

student_finetuned.save("sentiment_finetuned.json")
```

Key points:
- The teacher is prompt-optimized with MIPROv2 before distillation -- better teacher traces lead to a better student
- The student baseline (untuned GPT-4o-mini) gives you a floor to measure improvement against
- Always evaluate on a held-out test set, not the dev set you used during optimization
- The fine-tuned student typically retains 90-95% of teacher quality at a fraction of the cost
- The saved program file stores the fine-tuned model ID, so loading it later automatically uses the right model


## Example 2: Production cost reduction via fine-tuning

Take a working production system that uses an expensive model and reduce costs by fine-tuning a cheaper model to replace it. This example shows the full workflow from measuring the current system to deploying the fine-tuned replacement.

```python
import dspy
from dspy.evaluate import Evaluate

# --- The existing production system ---
# Assume this is already running in production with GPT-4o

class ExtractOrderInfo(dspy.Signature):
    """Extract structured order information from a customer message."""
    message: str = dspy.InputField(desc="Customer support message")
    order_id: str = dspy.OutputField(desc="Order ID mentioned, or 'none'")
    issue_type: str = dspy.OutputField(desc="return, shipping, damage, billing, other")
    urgency: str = dspy.OutputField(desc="low, medium, high")


expensive_lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=expensive_lm)

production_program = dspy.ChainOfThought(ExtractOrderInfo)


# --- Metric: all three fields must match ---

def metric(example, prediction, trace=None):
    order_match = prediction.order_id.strip().lower() == example.order_id.strip().lower()
    issue_match = prediction.issue_type.strip().lower() == example.issue_type.strip().lower()
    urgency_match = prediction.urgency.strip().lower() == example.urgency.strip().lower()
    return order_match and issue_match and urgency_match


# --- Collect labeled data from production logs ---
# In practice, export from your logging system. You need 500+ labeled examples.

import json

with open("support_messages_labeled.json") as f:
    raw = json.load(f)

examples = [
    dspy.Example(
        message=r["message"],
        order_id=r["order_id"],
        issue_type=r["issue_type"],
        urgency=r["urgency"],
    ).with_inputs("message")
    for r in raw
]

n = len(examples)
trainset = examples[: int(n * 0.8)]
devset = examples[int(n * 0.8) : int(n * 0.9)]
testset = examples[int(n * 0.9) :]

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
test_evaluator = Evaluate(devset=testset, metric=metric, num_threads=4, display_progress=True)


# --- Step 1: Measure current production quality ---

production_score = evaluator(production_program)
print(f"Current production (GPT-4o): {production_score:.1f}%")


# --- Step 2: Check how much quality we lose with the cheap model ---

cheap_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=cheap_lm)

cheap_baseline = dspy.ChainOfThought(ExtractOrderInfo)
cheap_score = evaluator(cheap_baseline)
print(f"Cheap baseline (GPT-4o-mini, no tuning): {cheap_score:.1f}%")


# --- Step 3: Fine-tune the cheap model using production system as teacher ---

# Switch back to expensive LM for generating teacher traces
dspy.configure(lm=expensive_lm)

# Optionally prompt-optimize the teacher for even better traces
prompt_optimizer = dspy.MIPROv2(metric=metric, auto="light")
teacher = prompt_optimizer.compile(production_program, trainset=trainset)

teacher_score = evaluator(teacher)
print(f"Teacher (GPT-4o, prompt-optimized): {teacher_score:.1f}%")

# Now fine-tune the cheap model
dspy.configure(lm=cheap_lm)

student = dspy.ChainOfThought(ExtractOrderInfo)
ft_optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=24)
student_finetuned = ft_optimizer.compile(
    student,
    trainset=trainset,
    teacher=teacher,
)

finetuned_score = evaluator(student_finetuned)
print(f"Fine-tuned (GPT-4o-mini, distilled): {finetuned_score:.1f}%")


# --- Step 4: Final evaluation on held-out test set ---

print("\n--- Test set results (held-out) ---")
prod_test = test_evaluator(production_program)
ft_test = test_evaluator(student_finetuned)
print(f"Production (GPT-4o):               {prod_test:.1f}%")
print(f"Fine-tuned (GPT-4o-mini):          {ft_test:.1f}%")
print(f"Quality retained:                  {ft_test / max(prod_test, 0.01) * 100:.0f}%")


# --- Step 5: Cost comparison ---

# Approximate costs per 1M input tokens (as of 2024)
# GPT-4o:      $2.50 input / $10.00 output
# GPT-4o-mini: $0.15 input / $0.60 output  (fine-tuned: ~$0.23 / $0.90)

print("\n--- Cost comparison (per 1M messages, ~200 tokens each) ---")
gpt4o_cost = (200 * 2.50 / 1_000_000) + (100 * 10.00 / 1_000_000)  # input + output
mini_ft_cost = (200 * 0.23 / 1_000_000) + (100 * 0.90 / 1_000_000)
savings = (1 - mini_ft_cost / gpt4o_cost) * 100

print(f"GPT-4o per message:           ${gpt4o_cost * 1_000_000:.2f} per 1M messages")
print(f"GPT-4o-mini (fine-tuned):     ${mini_ft_cost * 1_000_000:.2f} per 1M messages")
print(f"Cost reduction:               {savings:.0f}%")


# --- Step 6: Save and deploy ---

student_finetuned.save("order_extraction_finetuned.json")

# To load in production:
# from my_module import OrderExtractionProgram
# program = OrderExtractionProgram()
# program.load("order_extraction_finetuned.json")
```

Key points:
- Start by measuring your current production system's quality -- this is the bar the fine-tuned model needs to clear
- Always check the cheap model's untuned baseline first. If it's already close to the expensive model, you might not need fine-tuning at all (just prompt optimization with `/ai-improving-accuracy`)
- The teacher can be the existing production program or a prompt-optimized version of it. Better teacher traces produce a better student.
- The strict metric (all three fields must match) ensures only high-quality traces become training data
- Run the cost comparison before deploying to confirm the savings justify the fine-tuning effort
- If the fine-tuned model retains less than 90% of production quality, consider using `dspy.BetterTogether` (see `/ai-fine-tuning`) or adding more training data
