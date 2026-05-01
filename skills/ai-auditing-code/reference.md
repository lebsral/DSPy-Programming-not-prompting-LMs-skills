# ai-auditing-code: 7-Category Checklist

Full checklist items and before/after code examples for each audit category.

---

## Category 1: Signature Design

- [ ] Field names are descriptive — not `input`, `output`, `result`, `text`, `data`
- [ ] Output fields use specific types: `str`, `int`, `float`, `bool`, `list[str]`, or Pydantic model
- [ ] Complex structured outputs use a `pydantic.BaseModel`, not a plain `str` that gets parsed later
- [ ] Ambiguous fields have a `desc=` argument explaining what the field should contain
- [ ] Input fields that are not part of the LM reasoning are not placed in the signature

**Before — generic field names:**
```python
class Classify(dspy.Signature):
    input: str = dspy.InputField()
    output: str = dspy.OutputField()
```

**After — descriptive field names with desc:**
```python
class ClassifyTicket(dspy.Signature):
    """Classify a customer support ticket into a category."""
    ticket_text: str = dspy.InputField(desc="raw text of the customer support ticket")
    category: str = dspy.OutputField(desc="one of: billing, technical, account, general")
```

**Before — plain string output that gets parsed:**
```python
class ExtractEntities(dspy.Signature):
    document: str = dspy.InputField()
    entities: str = dspy.OutputField(desc="comma-separated list of entities")
```

**After — typed Pydantic output:**
```python
from pydantic import BaseModel

class EntityList(BaseModel):
    names: list[str]
    organizations: list[str]

class ExtractEntities(dspy.Signature):
    document: str = dspy.InputField()
    entities: EntityList = dspy.OutputField()
```

---

## Category 2: Module Composition

- [ ] All sub-modules are assigned in `__init__` so DSPy can track them
- [ ] `forward()` passes outputs of one sub-module as typed fields to the next, not as strings
- [ ] No `str(prediction)` or `prediction.completions[0]` hacks to extract values
- [ ] No string concatenation to build inputs for sub-modules
- [ ] Module returns a `dspy.Prediction` or the fields directly, not a plain dict

**Before — sub-module created inside forward(), not tracked:**
```python
class MyPipeline(dspy.Module):
    def forward(self, question):
        answer = dspy.ChainOfThought("question -> answer")(question=question)
        return answer
```

**After — sub-module registered in __init__:**
```python
class MyPipeline(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.answer(question=question)
```

**Before — string concatenation between modules:**
```python
def forward(self, docs):
    summaries = [self.summarize(doc=d).summary for d in docs]
    combined = "\n".join(summaries)
    return self.conclude(summaries=combined)
```

**After — pass structured data:**
```python
def forward(self, docs):
    summaries = [self.summarize(doc=d).summary for d in docs]
    return self.conclude(summaries=summaries)  # pass list, let DSPy handle it
```

---

## Category 3: Data Pipeline

- [ ] `dspy.Example(...).with_inputs(...)` is called on every example
- [ ] The field names in examples match the signature's input/output field names exactly
- [ ] A train/dev split exists; evaluation is not done on the training set
- [ ] Data loading uses a fixed random seed for reproducibility
- [ ] There are at least 20 examples in the training set (50+ recommended for MIPROv2)

**Before — missing with_inputs():**
```python
examples = [
    dspy.Example(question="What is 2+2?", answer="4"),
    dspy.Example(question="Capital of France?", answer="Paris"),
]
```

**After — with_inputs() called:**
```python
examples = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question"),
]
```

**Before — no train/dev split:**
```python
examples = load_all_examples()
compiled = optimizer.compile(program, trainset=examples)
evaluate(program, devset=examples)  # evaluating on training data
```

**After — proper split:**
```python
import random
random.seed(42)
all_examples = load_all_examples()
random.shuffle(all_examples)
split = int(0.8 * len(all_examples))
trainset = all_examples[:split]
devset = all_examples[split:]
compiled = optimizer.compile(program, trainset=trainset)
evaluate(program, devset=devset)
```

---

## Category 4: Metric Design

- [ ] Metric function signature is `metric(example, prediction, trace=None)`
- [ ] Metric returns `float` in range 0.0–1.0 or `bool`
- [ ] Metric handles `None` and empty string values without crashing
- [ ] When `trace is not None` (optimization mode), metric can apply stricter criteria
- [ ] Metric logic can be tested with a simple unit test, independent of the module

**Before — metric ignores trace parameter, always returns True:**
```python
def my_metric(example, prediction):
    return True
```

**After — proper metric with trace parameter and real logic:**
```python
def my_metric(example, prediction, trace=None):
    if not prediction.answer:
        return False
    expected = example.answer.strip().lower()
    actual = prediction.answer.strip().lower()
    match = expected in actual or actual in expected
    if trace is not None:
        # stricter during optimization: require exact match
        return expected == actual
    return match
```

**Before — metric crashes on edge cases:**
```python
def accuracy_metric(example, prediction, trace=None):
    return prediction.label.lower() == example.label.lower()
```

**After — handles None and missing fields:**
```python
def accuracy_metric(example, prediction, trace=None):
    pred_label = getattr(prediction, "label", None)
    true_label = getattr(example, "label", None)
    if pred_label is None or true_label is None:
        return False
    return pred_label.strip().lower() == true_label.strip().lower()
```

---

## Category 5: Optimizer Usage

- [ ] Optimizer is matched to dataset size: `BootstrapFewShot` for small datasets (under 50 examples), `MIPROv2` for larger ones
- [ ] `trainset` passed to `compile()` contains `dspy.Example` objects with `with_inputs()` called
- [ ] Metric is passed to the optimizer, not hardcoded or skipped
- [ ] Optimized program is saved with `program.save("path/program.json")`
- [ ] The uncompiled baseline is evaluated before compilation for comparison

**Before — MIPROv2 with 10 examples:**
```python
optimizer = dspy.MIPROv2(metric=my_metric, auto="medium")
compiled = optimizer.compile(program, trainset=small_set)  # only 10 examples
```

**After — BootstrapFewShot for small datasets:**
```python
optimizer = dspy.BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=4)
compiled = optimizer.compile(program, trainset=small_set)
```

**Before — optimized program not saved:**
```python
compiled = optimizer.compile(program, trainset=trainset)
# program is lost when the script exits
```

**After — save the optimized program:**
```python
compiled = optimizer.compile(program, trainset=trainset)
compiled.save("optimized_program.json")
# later: program.load("optimized_program.json")
```

---

## Category 6: Production Readiness

- [ ] LM calls are wrapped in try/except with a meaningful fallback or error message
- [ ] Timeouts are configured on the LM client
- [ ] There is a retry strategy or fallback model for LM failures
- [ ] Token usage and estimated costs have been calculated for expected traffic
- [ ] Logging captures inputs/outputs for debugging production issues

**Before — no error handling:**
```python
def classify(ticket_text):
    result = program(ticket_text=ticket_text)
    return result.category
```

**After — error handling with fallback:**
```python
def classify(ticket_text):
    try:
        result = program(ticket_text=ticket_text)
        return result.category
    except Exception as e:
        logger.error(f"Classification failed: {e}", extra={"ticket_text": ticket_text})
        return "general"  # safe fallback category
```

**Before — no timeout set:**
```python
lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)
```

**After — timeout configured:**
```python
lm = dspy.LM("openai/gpt-4o", timeout=30)
dspy.configure(lm=lm)
```

---

## Category 7: Anti-patterns

- [ ] No f-string prompt construction that bypasses the signature system
- [ ] No direct `lm("some prompt")` calls — all LM calls go through a module
- [ ] No hardcoded prompt strings sitting alongside DSPy signatures in the same file
- [ ] No mixing of raw `openai.chat.completions.create()` calls with DSPy modules in the same pipeline
- [ ] No manual JSON parsing of LM outputs that should be typed fields

**Before — f-string prompt instead of signature:**
```python
prompt = f"Classify this ticket: {ticket_text}. Categories: billing, technical, account."
result = lm(prompt)
category = result[0]["content"]
```

**After — DSPy signature:**
```python
class ClassifyTicket(dspy.Signature):
    """Classify a customer support ticket."""
    ticket_text: str = dspy.InputField()
    category: str = dspy.OutputField(desc="one of: billing, technical, account, general")

classifier = dspy.Predict(ClassifyTicket)
result = classifier(ticket_text=ticket_text)
category = result.category
```

**Before — direct lm() call:**
```python
lm = dspy.LM("openai/gpt-4o")
response = lm(f"Summarize: {doc}")
```

**After — module with signature:**
```python
class Summarize(dspy.Signature):
    document: str = dspy.InputField()
    summary: str = dspy.OutputField()

summarizer = dspy.Predict(Summarize)
result = summarizer(document=doc)
```

**Before — manual JSON parsing:**
```python
class ExtractData(dspy.Signature):
    text: str = dspy.InputField()
    result: str = dspy.OutputField(desc='JSON with keys "name" and "age"')

prediction = extractor(text=text)
data = json.loads(prediction.result)  # fragile, breaks if LM adds prose
```

**After — typed Pydantic field:**
```python
from pydantic import BaseModel

class PersonData(BaseModel):
    name: str
    age: int

class ExtractData(dspy.Signature):
    text: str = dspy.InputField()
    result: PersonData = dspy.OutputField()

prediction = extractor(text=text)
data = prediction.result  # already a PersonData instance
```
