# Worked Examples: Generating Synthetic Data

## Example 1: Cold start — ticket classifier with zero real data

You're building a support ticket classifier for a new product. No real tickets exist yet. The PM wants a working prototype by end of week.

### Define the task and generator

```python
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

# What the AI does in production
class ClassifyTicket(dspy.Signature):
    """Classify a support ticket into a category."""
    ticket_text: str = dspy.InputField()
    category: str = dspy.OutputField(desc="one of: billing, bug, how-to, feature-request, account")

# What generates training examples
class GenerateTicketExample(dspy.Signature):
    """Generate a realistic support ticket for the given category. Make it sound like a real customer wrote it — varied tone, length, and detail level."""
    category: str = dspy.InputField(desc="the target category")
    sindex: str = dspy.InputField(desc="unique seed for diversity")
    ticket_text: str = dspy.OutputField(desc="a realistic support ticket")
```

### Write seed examples (5 is enough)

```python
seeds = [
    dspy.Example(ticket_text="I was charged twice for my subscription this month. Order #4521.", category="billing").with_inputs("ticket_text"),
    dspy.Example(ticket_text="The app crashes when I try to upload a profile photo on Android.", category="bug").with_inputs("ticket_text"),
    dspy.Example(ticket_text="How do I export my data to CSV? Can't find the option.", category="how-to").with_inputs("ticket_text"),
    dspy.Example(ticket_text="Would love to see dark mode. The white background hurts my eyes at night.", category="feature-request").with_inputs("ticket_text"),
    dspy.Example(ticket_text="My account got locked after too many login attempts. Need help ASAP.", category="account").with_inputs("ticket_text"),
]
```

### Generate 200 examples (40 per category)

```python
import random

categories = ["billing", "bug", "how-to", "feature-request", "account"]
generator = dspy.Predict(GenerateTicketExample)
generated = []

for category in categories:
    for i in range(40):
        result = generator(category=category, sindex=str(random.randint(0, 1_000_000)))
        generated.append(
            dspy.Example(ticket_text=result.ticket_text, category=category).with_inputs("ticket_text")
        )

print(f"Generated {len(generated)} examples")
# Generated 200 examples
```

### Filter for quality

```python
class AssessExample(dspy.Signature):
    """Assess whether a generated support ticket is realistic and correctly labeled."""
    ticket_text: str = dspy.InputField()
    category: str = dspy.InputField()
    is_realistic: bool = dspy.OutputField(desc="true if this reads like a real customer ticket")
    is_correctly_labeled: bool = dspy.OutputField(desc="true if the category is correct for this ticket")

assessor = dspy.Predict(AssessExample)
filtered = []

for ex in generated:
    result = assessor(ticket_text=ex.ticket_text, category=ex.category)
    if result.is_realistic and result.is_correctly_labeled:
        filtered.append(ex)

# Deduplicate
seen = set()
unique = []
for ex in filtered:
    key = ex.ticket_text.strip().lower()
    if key not in seen:
        seen.add(key)
        unique.append(ex)

filtered = unique
print(f"Kept {len(filtered)}/200 after filtering")
# Kept 156/200 after filtering
```

### Optimize and evaluate

```python
from dspy.evaluate import Evaluate

# Split
random.shuffle(filtered)
split = int(len(filtered) * 0.8)
trainset = filtered[:split]
devset = filtered[split:]

# Switch to cheaper model for the task
task_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=task_lm)

def metric(example, prediction, trace=None):
    return prediction.category.strip().lower() == example.category.strip().lower()

# Baseline (no optimization)
program = dspy.ChainOfThought(ClassifyTicket)
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
baseline = evaluator(program)
print(f"Baseline: {baseline:.1f}%")
# Baseline: 72.0%

# Optimize
optimizer = dspy.MIPROv2(metric=metric, auto="light")
optimized = optimizer.compile(program, trainset=trainset)

optimized_score = evaluator(optimized)
print(f"Optimized: {optimized_score:.1f}%")
# Optimized: 89.0%

optimized.save("ticket_classifier.json")
```

From zero data to 89% accuracy, without a single real ticket.

## Example 2: Filling edge case gaps

Your ticket classifier runs at 85% on real data, but error analysis shows it fails on:
- Angry customers with profanity and caps
- Multi-issue tickets (billing AND bug in one message)
- Non-English or mixed-language tickets

### Identify the gaps

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=real_devset, metric=metric, num_threads=4, display_table=20)
score = evaluator(optimized_program)
# Look at the failures in the display table to identify patterns
```

### Generate targeted examples for each gap

```python
class GenerateScenarioTicket(dspy.Signature):
    """Generate a support ticket matching a specific edge case scenario. Make it realistic — these should be the tricky cases that are hard to classify correctly."""
    category: str = dspy.InputField(desc="the correct category for this ticket")
    scenario: str = dspy.InputField(desc="the edge case scenario to generate")
    ticket_text: str = dspy.OutputField(desc="a realistic ticket matching this scenario")

gen = dspy.Predict(GenerateScenarioTicket)

# Define edge case scenarios
edge_cases = [
    # Angry customers
    ("billing", "furious customer using caps and strong language about being overcharged"),
    ("bug", "frustrated user who has reported this bug three times already"),
    ("account", "angry customer locked out before an important deadline"),
    # Multi-issue tickets
    ("billing", "customer reports both a billing error AND a bug in the same ticket"),
    ("bug", "user asks how to do something AND reports a bug they found while trying"),
    ("account", "customer has account access issues AND wants a feature added"),
    # Non-English / mixed language
    ("billing", "ticket written mostly in Spanish with some English technical terms"),
    ("how-to", "ticket in broken English from a non-native speaker"),
    ("feature-request", "ticket mixing French and English"),
    ("bug", "ticket written in informal/slang English that's hard to parse"),
]

edge_examples = []
for category, scenario in edge_cases:
    for i in range(20):
        result = gen(category=category, scenario=scenario)
        edge_examples.append(
            dspy.Example(ticket_text=result.ticket_text, category=category).with_inputs("ticket_text")
        )

print(f"Generated {len(edge_examples)} edge case examples")
# Generated 200 edge case examples
```

### Filter and merge with existing data

```python
# Filter the edge case examples
assessor = dspy.Predict(AssessExample)
filtered_edges = []

for ex in edge_examples:
    result = assessor(ticket_text=ex.ticket_text, category=ex.category)
    if result.is_realistic and result.is_correctly_labeled:
        filtered_edges.append(ex)

print(f"Kept {len(filtered_edges)} edge case examples after filtering")

# Merge with existing training data
augmented_trainset = existing_trainset + filtered_edges
random.shuffle(augmented_trainset)
```

### Re-optimize and compare

```python
program = dspy.ChainOfThought(ClassifyTicket)
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
re_optimized = optimizer.compile(program, trainset=augmented_trainset)

# Evaluate on real data
score = evaluator(re_optimized)
print(f"Before edge cases: 85.0%")
print(f"After edge cases:  {score:.1f}%")
# After edge cases: 91.3%
```

Targeted synthetic data for specific failure modes is more effective than generating more uniform data.

## Example 3: Privacy-safe dataset for medical triage

You're building a medical triage system that categorizes patient complaints. Compliance says you can't use real patient data for AI training. Everything must be synthetic.

### Define task and generator with domain-specific detail

```python
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

class TriageComplaint(dspy.Signature):
    """Triage a patient complaint by urgency level."""
    complaint: str = dspy.InputField(desc="patient's description of their symptoms")
    urgency: str = dspy.OutputField(desc="one of: emergency, urgent, standard, routine")

class GenerateComplaint(dspy.Signature):
    """Generate a realistic patient complaint for a medical triage system. The complaint should sound like how a real patient would describe their symptoms — using everyday language, not medical terminology. Include realistic but entirely fictional details. Never use real patient data."""
    urgency: str = dspy.InputField(desc="target urgency level: emergency, urgent, standard, or routine")
    scenario: str = dspy.InputField(desc="the medical scenario to generate")
    complaint: str = dspy.OutputField(desc="a realistic patient complaint in the patient's own words")
```

### Define domain-specific scenarios

```python
scenarios = {
    "emergency": [
        "chest pain with shortness of breath",
        "severe allergic reaction with swelling",
        "sudden loss of consciousness",
        "heavy uncontrolled bleeding",
        "signs of stroke — slurred speech, face drooping",
        "difficulty breathing in a child",
    ],
    "urgent": [
        "high fever lasting more than 3 days",
        "deep cut that may need stitches",
        "severe abdominal pain",
        "possible broken bone after a fall",
        "worsening infection with spreading redness",
        "persistent vomiting preventing hydration",
    ],
    "standard": [
        "mild ear infection symptoms",
        "persistent cough for two weeks",
        "minor skin rash that isn't improving",
        "recurring headaches",
        "mild back pain after lifting",
        "urinary tract infection symptoms",
    ],
    "routine": [
        "annual checkup scheduling",
        "prescription refill request",
        "vaccination appointment",
        "follow-up after normal test results",
        "minor seasonal allergy management",
        "request for medical records",
    ],
}
```

### Generate with quality gates

Use `dspy.Suggest` to enforce quality during generation:

```python
class AssessMedicalExample(dspy.Signature):
    """Assess a generated patient complaint for quality."""
    complaint: str = dspy.InputField()
    urgency: str = dspy.InputField()
    is_medically_plausible: bool = dspy.OutputField(desc="symptoms match a real medical scenario")
    urgency_is_correct: bool = dspy.OutputField(desc="urgency level is appropriate for these symptoms")
    contains_no_pii: bool = dspy.OutputField(desc="no real names, dates of birth, SSNs, or identifiable info")
    uses_patient_language: bool = dspy.OutputField(desc="written like a patient, not a doctor")

class SafeMedicalGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateComplaint)
        self.assess = dspy.Predict(AssessMedicalExample)

    def forward(self, urgency, scenario):
        result = self.generate(urgency=urgency, scenario=scenario)
        assessment = self.assess(complaint=result.complaint, urgency=urgency)
        dspy.Suggest(assessment.is_medically_plausible, "Complaint should describe a real medical scenario")
        dspy.Suggest(assessment.urgency_is_correct, "Urgency level should match the symptoms described")
        dspy.Suggest(assessment.contains_no_pii, "Complaint must not contain any personally identifiable information")
        dspy.Suggest(assessment.uses_patient_language, "Complaint should sound like a patient, not a medical professional")
        return result

generator = SafeMedicalGenerator()
```

### Generate and collect

```python
import random

generated = []
for urgency, scenario_list in scenarios.items():
    for scenario in scenario_list:
        for i in range(15):
            try:
                result = generator(urgency=urgency, scenario=scenario)
                generated.append(
                    dspy.Example(complaint=result.complaint, urgency=urgency).with_inputs("complaint")
                )
            except Exception:
                # Suggest retries exhausted — skip this one
                pass

print(f"Generated {len(generated)} examples")
# Generated 342 examples (some lost to quality gate failures)
```

### Post-generation privacy audit

Even with quality gates, do a final pass:

```python
class PrivacyAudit(dspy.Signature):
    """Check if text contains any personally identifiable information (PII)."""
    text: str = dspy.InputField()
    contains_pii: bool = dspy.OutputField(desc="true if text contains names, DOBs, SSNs, addresses, phone numbers, or other PII")
    pii_found: str = dspy.OutputField(desc="description of PII found, or 'none'")

auditor = dspy.Predict(PrivacyAudit)
safe = []

for ex in generated:
    result = auditor(text=ex.complaint)
    if not result.contains_pii:
        safe.append(ex)
    else:
        print(f"Removed (PII: {result.pii_found}): {ex.complaint[:60]}...")

print(f"After privacy audit: {len(safe)}/{len(generated)} examples kept")
```

### Optimize and deploy

```python
from dspy.evaluate import Evaluate

random.shuffle(safe)
split = int(len(safe) * 0.8)
trainset = safe[:split]
devset = safe[split:]

# Use cheaper model for the task
task_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=task_lm)

def metric(example, prediction, trace=None):
    return prediction.urgency.strip().lower() == example.urgency.strip().lower()

program = dspy.ChainOfThought(TriageComplaint)
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized = optimizer.compile(program, trainset=trainset)

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
score = evaluator(optimized)
print(f"Triage accuracy: {score:.1f}%")

optimized.save("triage_program.json")
```

The result: a medical triage system trained entirely on synthetic data, with no real patient information in the training pipeline. When real data becomes available (with proper consent), mix it in as the dev set for more trustworthy evaluation.
