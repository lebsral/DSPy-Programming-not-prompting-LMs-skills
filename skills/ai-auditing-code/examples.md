# ai-auditing-code: Worked Examples

Three worked audits showing the full process: code being reviewed, findings report, and fixes applied.

---

## Example 1: Ticket Classifier

### Code being audited

```python
# classifier.py
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

class Classify(dspy.Signature):
    input: str = dspy.InputField()
    output: str = dspy.OutputField()

class TicketClassifier(dspy.Module):
    def forward(self, text):
        result = dspy.Predict(Classify)(input=text)
        return result.output

examples = [
    dspy.Example(input="my card was charged twice", output="billing"),
    dspy.Example(input="app keeps crashing", output="technical"),
    dspy.Example(input="reset my password", output="account"),
]

def my_metric(example, prediction):
    return prediction.output == example.output

optimizer = dspy.MIPROv2(metric=my_metric, auto="medium")
compiled = optimizer.compile(TicketClassifier(), trainset=examples)
```

### Findings report

```
## DSPy Code Audit: TicketClassifier

### Summary
- 5 findings: 3 critical, 1 warning, 1 info
- Overall assessment: Needs fixes before production

### Critical Findings

1. **Data Pipeline — missing with_inputs()**
   - File: classifier.py:14-16
   - Problem: Examples do not call with_inputs(), so DSPy cannot distinguish input
     fields from output fields. The optimizer will treat all fields as inputs.
   - Fix: Add .with_inputs("input") to each example.
   - Code:
     Before: dspy.Example(input="my card was charged twice", output="billing")
     After:  dspy.Example(input="my card was charged twice", output="billing").with_inputs("input")

2. **Metric Design — metric ignores trace parameter**
   - File: classifier.py:18
   - Problem: Metric does not accept the trace parameter. This causes a TypeError
     during MIPROv2 optimization, which passes trace as a keyword argument.
   - Fix: Add trace=None to the metric signature.
   - Code:
     Before: def my_metric(example, prediction):
     After:  def my_metric(example, prediction, trace=None):

3. **Optimizer Usage — MIPROv2 with 3 examples**
   - File: classifier.py:21
   - Problem: MIPROv2 requires at least 50 examples to work effectively. With 3
     examples the optimizer cannot bootstrap meaningful demonstrations and will
     likely produce worse results than the baseline.
   - Fix: Use BootstrapFewShot for small datasets.
   - Code:
     Before: optimizer = dspy.MIPROv2(metric=my_metric, auto="medium")
     After:  optimizer = dspy.BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=2)

### Warnings

1. **Signature Design — generic field names**
   - File: classifier.py:7-8
   - Problem: Fields named "input" and "output" give the LM no context about the
     task. This reduces accuracy because the LM cannot infer expected content.
   - Fix: Rename to ticket_text and category with descriptive desc= arguments.

### Info

1. **Module Composition — Predict created inside forward()**
   - File: classifier.py:12
   - Problem: dspy.Predict(Classify) is constructed on every call to forward().
     This works but prevents DSPy from tracking the module for optimization.
   - Fix: Move to __init__: self.classify = dspy.Predict(Classify)

### Recommended Next Steps
1. Add with_inputs() to all examples
2. Fix metric signature to accept trace parameter
3. Switch to BootstrapFewShot or add more training examples
4. Rename signature fields to be descriptive
5. Run /ai-improving-accuracy to measure baseline quality
```

### Fixes applied

```python
# classifier.py — after fixes
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

class ClassifyTicket(dspy.Signature):
    """Classify a customer support ticket into a category."""
    ticket_text: str = dspy.InputField(desc="raw text of the customer support ticket")
    category: str = dspy.OutputField(desc="one of: billing, technical, account, general")

class TicketClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(ClassifyTicket)

    def forward(self, ticket_text):
        return self.classify(ticket_text=ticket_text)

examples = [
    dspy.Example(ticket_text="my card was charged twice", category="billing").with_inputs("ticket_text"),
    dspy.Example(ticket_text="app keeps crashing", category="technical").with_inputs("ticket_text"),
    dspy.Example(ticket_text="reset my password", category="account").with_inputs("ticket_text"),
]

def my_metric(example, prediction, trace=None):
    pred = getattr(prediction, "category", None)
    true = getattr(example, "category", None)
    if pred is None or true is None:
        return False
    return pred.strip().lower() == true.strip().lower()

optimizer = dspy.BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=2)
compiled = optimizer.compile(TicketClassifier(), trainset=examples)
compiled.save("ticket_classifier.json")
```

---

## Example 2: RAG Pipeline

### Code being audited

```python
# rag.py
import dspy

lm = dspy.LM("openai/gpt-4o")
rm = dspy.ColBERTv2(url="http://my-colbert-server/")
dspy.configure(lm=lm, rm=rm)

class GenerateAnswer(dspy.Signature):
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class RAGPipeline(dspy.Module):
    def forward(self, question):
        docs = dspy.Retrieve(k=3)(question)
        context = ""
        for doc in docs.passages:
            context = context + doc + " "
        result = dspy.ChainOfThought(GenerateAnswer)(
            context=context,
            question=question
        )
        return result

# training
trainset = load_training_data()  # returns 200 examples
optimizer = dspy.MIPROv2(metric=answer_metric, auto="medium")
compiled = optimizer.compile(RAGPipeline(), trainset=trainset)
print("done")  # optimized program is not saved
```

### Findings report

```
## DSPy Code Audit: RAGPipeline

### Summary
- 4 findings: 2 critical, 1 warning, 1 info
- Overall assessment: Needs fixes before production

### Critical Findings

1. **Optimizer Usage — optimized program not saved**
   - File: rag.py:25
   - Problem: The compiled program exists only in memory. When the script exits,
     all optimization work is lost. The program must be saved to disk to be
     reused.
   - Fix: Call compiled.save() after compilation.
   - Code:
     Before: print("done")
     After:  compiled.save("rag_pipeline.json")

2. **Module Composition — sub-modules created inside forward()**
   - File: rag.py:14-15
   - Problem: dspy.Retrieve and dspy.ChainOfThought are constructed on every
     forward() call. DSPy cannot track these for optimization — the optimizer
     cannot tune their prompts or demonstrations.
   - Fix: Move both to __init__.

### Warnings

1. **Module Composition — string concatenation of retrieved docs**
   - File: rag.py:16-17
   - Problem: Concatenating passages with string addition and trailing spaces is
     fragile. If passages contain special characters or are very long, the
     context field may be malformed. Use str.join() instead.
   - Fix: context = "\n\n".join(docs.passages)

### Info

1. **Production Readiness — no error handling around LM calls**
   - File: rag.py:18-21
   - Problem: If the retrieval server is down or the LM times out, the pipeline
     raises an unhandled exception. A fallback response keeps the application
     running.

### Recommended Next Steps
1. Save the compiled program after optimization
2. Move Retrieve and ChainOfThought to __init__
3. Replace string concatenation with str.join()
4. Add try/except around the forward() body
```

### Fixes applied

```python
# rag.py — after fixes
import dspy

lm = dspy.LM("openai/gpt-4o", timeout=30)
rm = dspy.ColBERTv2(url="http://my-colbert-server/")
dspy.configure(lm=lm, rm=rm)

class GenerateAnswer(dspy.Signature):
    """Answer a question using the provided context passages."""
    context: str = dspy.InputField(desc="retrieved passages relevant to the question")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class RAGPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        try:
            docs = self.retrieve(question)
            context = "\n\n".join(docs.passages)
            return self.generate(context=context, question=question)
        except Exception as e:
            import logging
            logging.error(f"RAG pipeline failed: {e}")
            return dspy.Prediction(answer="I was unable to retrieve an answer at this time.")

# training
trainset = load_training_data()
optimizer = dspy.MIPROv2(metric=answer_metric, auto="medium")
compiled = optimizer.compile(RAGPipeline(), trainset=trainset)
compiled.save("rag_pipeline.json")
```

---

## Example 3: Content Generator

### Code being audited

```python
# generator.py
import openai
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

SYSTEM_PROMPT = """You are a helpful content writer. Write in a friendly tone.
Always include a call to action. Use bullet points where appropriate."""

class DraftContent(dspy.Signature):
    topic: str = dspy.InputField()
    draft: str = dspy.OutputField()

class ContentGenerator(dspy.Module):
    def __init__(self):
        self.draft = dspy.Predict(DraftContent)

    def forward(self, topic, audience):
        # use raw OpenAI for the outline because DSPy is slow
        client = openai.OpenAI()
        outline_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Create an outline for: {topic}, audience: {audience}"}
            ]
        )
        outline = outline_response.choices[0].message.content

        # inject outline into DSPy call
        prompt = f"Topic: {topic}\nOutline: {outline}\nWrite the full article now."
        result = lm(prompt)
        return result[0]["content"]
```

### Findings report

```
## DSPy Code Audit: ContentGenerator

### Summary
- 5 findings: 3 critical, 1 warning, 1 info
- Overall assessment: Needs significant rework

### Critical Findings

1. **Anti-patterns — raw OpenAI API call inside DSPy module**
   - File: generator.py:18-25
   - Problem: The outline step uses openai.chat.completions.create() directly.
     This call is invisible to DSPy — it cannot be optimized, traced, cached, or
     swapped to a different model. The module cannot be compiled.
   - Fix: Replace with a DSPy signature and Predict module.

2. **Anti-patterns — direct lm() call with f-string prompt**
   - File: generator.py:27-28
   - Problem: Calling lm(prompt) with a manually constructed f-string bypasses
     the entire DSPy signature system. The output is raw LM text that has to be
     unpacked manually. This call cannot be optimized.
   - Fix: Define a signature for the drafting step and use a Predict or
     ChainOfThought module.

3. **Anti-patterns — hardcoded SYSTEM_PROMPT alongside DSPy code**
   - File: generator.py:6-9
   - Problem: SYSTEM_PROMPT is a hardcoded string that gets passed to the raw
     OpenAI call. This creates two separate prompt systems in the same file.
     DSPy manages prompts through signatures and optimizers — hardcoded strings
     will not be updated when the program is compiled.
   - Fix: Remove SYSTEM_PROMPT. Encode style instructions in the signature
     docstring or as field desc= values, which the optimizer can tune.

### Warnings

1. **Signature Design — audience field missing from signature**
   - File: generator.py:12-13
   - Problem: The forward() method accepts an audience parameter but the
     signature has no audience field. The audience value is injected via the
     f-string workaround rather than being a proper signature input. The LM
     never sees audience in a structured way.
   - Fix: Add audience as an InputField to the signature.

### Info

1. **Production Readiness — no error handling**
   - File: generator.py:18-29
   - Problem: Both the raw OpenAI call and the lm() call can raise exceptions.
     No error handling is present.

### Recommended Next Steps
1. Replace raw OpenAI call with a DSPy Outline signature
2. Replace lm() call with a DSPy Draft signature
3. Remove SYSTEM_PROMPT; encode intent in signature docstrings
4. Add audience to the signature
5. Add error handling
```

### Fixes applied

```python
# generator.py — after fixes
import dspy

lm = dspy.LM("openai/gpt-4o", timeout=30)
dspy.configure(lm=lm)

class CreateOutline(dspy.Signature):
    """Create a structured outline for a content piece in a friendly, engaging tone."""
    topic: str = dspy.InputField()
    audience: str = dspy.InputField(desc="description of the target audience for this content")
    outline: str = dspy.OutputField(desc="structured outline with main sections and key points")

class DraftContent(dspy.Signature):
    """Write a full article from an outline. Include a call to action at the end."""
    topic: str = dspy.InputField()
    audience: str = dspy.InputField(desc="description of the target audience")
    outline: str = dspy.InputField(desc="structured outline to follow")
    article: str = dspy.OutputField(desc="complete article ready for publication")

class ContentGenerator(dspy.Module):
    def __init__(self):
        self.outline = dspy.ChainOfThought(CreateOutline)
        self.draft = dspy.ChainOfThought(DraftContent)

    def forward(self, topic, audience):
        try:
            outline_result = self.outline(topic=topic, audience=audience)
            draft_result = self.draft(
                topic=topic,
                audience=audience,
                outline=outline_result.outline
            )
            return draft_result
        except Exception as e:
            import logging
            logging.error(f"Content generation failed for topic={topic!r}: {e}")
            raise
```
