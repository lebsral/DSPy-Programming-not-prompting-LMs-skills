# dspy-modules -- Worked Examples

## Example 1: Simple QA module with pre/post processing

A question-answering module that normalizes input, generates an answer, and formats the output.

```python
import dspy


class CleanQA(dspy.Module):
    """QA module with input normalization and output formatting."""

    def __init__(self):
        self.answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str):
        # Pre-processing: normalize the question
        cleaned = question.strip().rstrip("?").strip() + "?"
        cleaned = cleaned[0].upper() + cleaned[1:]

        # Generate answer
        result = self.answer(question=cleaned)

        # Post-processing: ensure answer is complete sentences
        answer_text = result.answer.strip()
        if answer_text and not answer_text.endswith((".", "!", "?")):
            answer_text += "."

        return dspy.Prediction(
            answer=answer_text,
            reasoning=result.reasoning,
        )


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

qa = CleanQA()
result = qa(question="  what is dspy  ")
print(result.answer)
print(result.reasoning)
```

Key points:
- `forward()` is just Python -- do whatever pre/post processing you need
- `dspy.Prediction(...)` lets you return a clean result with named fields
- The `ChainOfThought` sub-module is declared in `__init__` so optimizers can find it


## Example 2: RAG pipeline module

A retrieval-augmented generation module that searches for context, then generates a grounded answer with source citations.

```python
import dspy
from typing import Literal


class AnswerWithSources(dspy.Signature):
    """Answer the question using only the provided context. Cite your sources."""
    context: list[str] = dspy.InputField(desc="Retrieved passages")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Answer grounded in the context")
    confidence: Literal["high", "medium", "low"] = dspy.OutputField(
        desc="How confident the answer is based on available context"
    )


class RAGPipeline(dspy.Module):
    """Retrieve relevant passages, then generate a grounded answer."""

    def __init__(self, k=3):
        self.retrieve = dspy.Retrieve(k=k)
        self.generate = dspy.ChainOfThought(AnswerWithSources)

    def forward(self, question: str):
        # Stage 1: Retrieve relevant passages
        retrieval_result = self.retrieve(question)
        passages = retrieval_result.passages

        # Guard: if no passages found, say so
        if not passages:
            return dspy.Prediction(
                answer="No relevant information found.",
                confidence="low",
                passages=[],
            )

        # Stage 2: Generate grounded answer
        result = self.generate(context=passages, question=question)

        # Soft constraint: answer should reference the context
        dspy.Suggest(
            result.confidence != "low",
            "If confidence is low, try to find a partial answer from the context",
        )

        return dspy.Prediction(
            answer=result.answer,
            confidence=result.confidence,
            passages=passages,
        )


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Note: dspy.Retrieve requires a retrieval model to be configured.
# See DSPy docs for ColBERTv2 or custom retriever setup.
rag = RAGPipeline(k=5)
result = rag(question="How does DSPy optimize prompts?")
print(result.answer)
print(f"Confidence: {result.confidence}")
print(f"Sources: {len(result.passages)} passages retrieved")
```

Key points:
- Each stage has a clear signature with typed fields
- `forward()` handles edge cases (no passages) with plain Python
- `dspy.Suggest` adds a soft quality constraint without hard-failing
- The module returns a `Prediction` that bundles the answer, confidence, and source passages


## Example 3: Multi-stage analysis -- classify, route, generate

A module that classifies incoming text, routes it to a specialized handler, and generates a tailored response. This pattern is common in support systems, content moderation, and document processing.

```python
import dspy
from typing import Literal


# --- Signatures ---

class ClassifyIntent(dspy.Signature):
    """Classify the user's message into a category."""
    message: str = dspy.InputField(desc="The user's message")
    category: Literal["question", "complaint", "feedback", "request"] = dspy.OutputField()


class AnswerQuestion(dspy.Signature):
    """Answer a factual question helpfully and concisely."""
    message: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="A direct, helpful answer")


class HandleComplaint(dspy.Signature):
    """Respond to a complaint with empathy and a resolution plan."""
    message: str = dspy.InputField()
    response: str = dspy.OutputField(desc="Empathetic response with next steps")
    escalate: bool = dspy.OutputField(desc="Whether this needs human review")


class HandleFeedback(dspy.Signature):
    """Acknowledge feedback and summarize the key points."""
    message: str = dspy.InputField()
    response: str = dspy.OutputField(desc="Acknowledgment and summary")


class HandleRequest(dspy.Signature):
    """Process a request and explain what will happen next."""
    message: str = dspy.InputField()
    response: str = dspy.OutputField(desc="Confirmation and next steps")


# --- Module ---

class SmartRouter(dspy.Module):
    """Classify a message, then route to a specialized handler."""

    def __init__(self):
        self.classify = dspy.Predict(ClassifyIntent)
        self.handlers = {
            "question": dspy.ChainOfThought(AnswerQuestion),
            "complaint": dspy.ChainOfThought(HandleComplaint),
            "feedback": dspy.Predict(HandleFeedback),
            "request": dspy.Predict(HandleRequest),
        }

    def forward(self, message: str):
        # Stage 1: Classify
        classification = self.classify(message=message)
        category = classification.category

        # Stage 2: Route to the right handler
        handler = self.handlers.get(category, self.handlers["question"])
        result = handler(message=message)

        # Stage 3: Build unified response
        response_text = result.answer if hasattr(result, "answer") else result.response
        escalate = result.escalate if hasattr(result, "escalate") else False

        return dspy.Prediction(
            category=category,
            response=response_text,
            escalate=escalate,
        )


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

router = SmartRouter()

# Print module structure to verify
print(router)

# Test with different message types
messages = [
    "What are your business hours?",
    "I've been waiting 3 weeks for my order and nobody responds!",
    "Love the new dashboard design, much easier to navigate.",
    "Can you change my subscription to the annual plan?",
]

for msg in messages:
    result = router(message=msg)
    print(f"\n[{result.category.upper()}] {msg}")
    print(f"Response: {result.response}")
    if result.escalate:
        print("** ESCALATE TO HUMAN **")


# --- Optimization ---

def router_metric(example, prediction, trace=None):
    """Score based on correct category and response quality."""
    category_correct = prediction.category == example.category
    # Check response is non-empty and reasonable length
    has_response = len(prediction.response.strip()) > 20
    return category_correct + 0.5 * has_response

# trainset = [dspy.Example(message=m, category=c).with_inputs("message") for m, c in data]
# optimizer = dspy.BootstrapFewShot(metric=router_metric, max_bootstrapped_demos=4)
# optimized_router = optimizer.compile(router, trainset=trainset)
# optimized_router.save("smart_router.json")
```

Key points:
- **Classify then route** is one of the most useful patterns -- cheap classification directs traffic to specialized handlers
- Handlers stored in a dict make the module easy to extend (add a new category = add a new handler)
- `dspy.Predict` (no reasoning) is used for simple tasks; `dspy.ChainOfThought` for ones that benefit from step-by-step thinking
- The unified `dspy.Prediction` return normalizes different handler output shapes
- When optimized, DSPy tunes the classifier and every handler together to maximize the end-to-end metric
