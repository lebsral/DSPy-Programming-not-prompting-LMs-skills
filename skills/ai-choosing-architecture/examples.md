# ai-choosing-architecture: Worked Examples

---

## Example 1: Support ticket classifier

### The task
A SaaS company receives ~2,000 support tickets per day. They want to automatically assign each ticket to one of four queues: billing, technical, account, other. A human agent reviews the assignment before acting on it.

### 3 questions answered

1. **What goes in and what comes out?** In: plain-text ticket body (50-500 words). Out: a single label from a fixed set of four values.
2. **Does the AI need external tools?** No. The category is determinable from the ticket text alone.
3. **How complex is the reasoning?** Simple. The mapping from text to category is direct. A human can usually classify a ticket in under five seconds.

### Decision tree walkthrough

```
Does it need tools? → No
How complex is the reasoning? → Simple (direct mapping)
→ Predict
```

ChainOfThought is tempting here but wrong. The reasoning trace adds tokens and cost but does not improve label accuracy — the model may even talk itself into the wrong category by over-analyzing edge cases. Predict is the right choice.

### Recommendation

**Module:** `dspy.Predict`
**Why:** Direct label assignment from a fixed taxonomy. Reasoning adds noise, not signal. Predict is faster and cheaper.

**Skeleton:**

```python
import dspy

class ClassifyTicket(dspy.Signature):
    """Classify a support ticket into exactly one support queue."""
    ticket_text: str = dspy.InputField()
    queue: str = dspy.OutputField(
        desc="One of: billing, technical, account, other"
    )

classifier = dspy.Predict(ClassifyTicket)

# Optimize with BootstrapFewShot on 100-200 labeled tickets
from dspy.teleprompt import BootstrapFewShot

def exact_match(example, pred, trace=None):
    return example.queue == pred.queue

optimizer = BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=6)
optimized_classifier = optimizer.compile(classifier, trainset=labeled_tickets)
```

**Optimizer path:**
1. BootstrapFewShot — quick baseline, should reach 85-90% accuracy
2. MIPROv2 if accuracy needs to improve beyond that

**Alternative considered:** ChainOfThought — rejected because the task is a fixed-label classification and reasoning does not help; it adds cost and can introduce label drift on edge cases.

---

## Example 2: Customer Q&A over product docs

### The task
A B2B software company wants a chatbot that answers customer questions using their product documentation (300 markdown files, ~2M tokens total). Answers must be grounded in the docs, not hallucinated.

### 3 questions answered

1. **What goes in and what comes out?** In: a natural-language question. Out: a plain-text answer with source references.
2. **Does the AI need external tools?** Yes — retrieval from the doc corpus. The docs are too large to fit in context.
3. **How complex is the reasoning?** Moderate. The model needs to identify the relevant parts of retrieved passages and synthesize a coherent answer.

### Decision tree walkthrough

```
Does it need tools? → Yes (retrieval)
Does it need to write and run code? → No
→ ReAct ... but wait
```

ReAct is the first branch for tool use, but retrieval-augmented generation is a well-known pattern where the tool call sequence is fixed and known upfront: always retrieve, then answer. A pipeline is preferable to ReAct here because:
- The steps are predetermined (no dynamic tool selection needed)
- Pipelines are cheaper and easier to optimize than ReAct
- The retrieval step can be a standard DSPy Retrieve module

**Revised decision:** Use a RAG pipeline (Retrieve + ChainOfThought).

### Recommendation

**Module:** RAG pipeline — `dspy.Retrieve` + `dspy.ChainOfThought`
**Why:** The retrieval step is always the same (fetch top-k passages). ChainOfThought synthesizes the answer with visible reasoning, making hallucination easier to detect.

**Skeleton:**

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

retriever = dspy.Retrieve(k=5)

class AnswerFromDocs(dspy.Signature):
    """Answer the customer question using only the provided documentation passages."""
    question: str = dspy.InputField()
    passages: list[str] = dspy.InputField(desc="Relevant documentation excerpts")
    answer: str = dspy.OutputField(desc="Clear answer grounded in the passages")
    source_hint: str = dspy.OutputField(desc="Which passage(s) supported this answer")

class DocQA(dspy.Module):
    def __init__(self):
        self.retrieve = retriever
        self.answer = dspy.ChainOfThought(AnswerFromDocs)

    def forward(self, question: str) -> dspy.Prediction:
        passages = self.retrieve(question).passages
        result = self.answer(question=question, passages=passages)
        return dspy.Prediction(
            answer=result.answer,
            source_hint=result.source_hint,
        )

qa = DocQA()
result = qa(question="How do I configure SSO with Okta?")
print(result.answer)
```

**Optimizer path:**
1. BootstrapFewShot on 50-100 QA pairs
2. MIPROv2 if answer quality needs to improve (optimize both the reasoning and answer stages end-to-end)

**Alternative considered:** Single ChainOfThought with full docs in context — rejected because the corpus is too large. ReAct — rejected because the retrieval pattern is fixed and a pipeline is simpler and cheaper.

---

## Example 3: AI research assistant

### The task
A market research firm wants an AI assistant that can research any company on demand: find recent news, pull financial data, and summarize competitive positioning. The set of data sources to consult varies by question.

### 3 questions answered

1. **What goes in and what comes out?** In: a research question (e.g., "What is Stripe's competitive position in payments?"). Out: a structured research summary.
2. **Does the AI need external tools?** Yes — web search, financial data APIs, news feeds. The specific tools needed depend on the question.
3. **How complex is the reasoning?** High. The agent must decide which sources to consult, in what order, and how to synthesize conflicting information.

### Decision tree walkthrough

```
Does it need tools? → Yes
Does it need to write and run code? → No
→ ReAct
```

Unlike the doc QA example, here the tool selection is dynamic. A question about a private company needs different tools than a question about a public company. ReAct's ability to choose tools at runtime is the right fit.

### Recommendation

**Module:** `dspy.ReAct`
**Why:** The task genuinely requires dynamic tool selection. The agent decides at runtime which sources to consult based on what it learns from each tool call.

**Skeleton:**

```python
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

def search_news(query: str) -> str:
    """Search recent news articles. Returns titles and snippets."""
    ...

def get_financial_data(ticker: str) -> str:
    """Get key financial metrics for a public company by stock ticker."""
    ...

def search_web(query: str) -> str:
    """General web search for any topic."""
    ...

class ResearchCompany(dspy.Signature):
    """Research a company and produce a structured competitive summary."""
    question: str = dspy.InputField()
    summary: str = dspy.OutputField(
        desc="Structured summary covering: overview, recent news, competitive position"
    )

agent = dspy.ReAct(
    ResearchCompany,
    tools=[search_news, get_financial_data, search_web],
    max_iters=8,
)

result = agent(question="What is Stripe competitive position in payments as of 2024?")
print(result.summary)
```

**Optimizer path:**
1. BootstrapFewShot with 10-20 research examples — agents need fewer demos than classifiers
2. Stay with BootstrapFewShot; MIPROv2 can be unstable for multi-turn agent traces

**Alternative considered:** Fixed RAG pipeline — rejected because the sources vary by question type. Pipeline with hardcoded stages — rejected because the number and order of API calls is not known upfront.

---

## Example 4: Essay grading system

### The task
An online learning platform grades student essays on a 1-5 rubric covering: thesis clarity, argument strength, evidence quality, and writing mechanics. Grades must be consistent — the same essay should always get the same score.

### 3 questions answered

1. **What goes in and what comes out?** In: a student essay (200-1,000 words). Out: four integer scores (1-5) plus brief justifications.
2. **Does the AI need external tools?** No. Grading is based solely on the essay text.
3. **How complex is the reasoning?** High. Rubric application requires careful analysis of multiple dimensions. Consistency across runs is critical.

### Decision tree walkthrough

```
Does it need tools? → No
How complex is the reasoning? → Very complex (multi-dimensional, must be consistent)
→ MultiChainComparison or ChainOfThought + BestOfN
```

MultiChainComparison generates N reasoning chains and compares them to pick the most consistent answer — good for subjective tasks where consistency matters. BestOfN with a scoring function is an alternative if a reliable scorer exists.

For essay grading, `ChainOfThought` with `BestOfN` (where the scorer checks rubric adherence and internal consistency) is often more practical than MCC because you can define the scoring criteria explicitly.

### Recommendation

**Module:** `dspy.ChainOfThought` + `dspy.BestOfN`
**Why:** ChainOfThought applies the rubric with visible reasoning. BestOfN samples multiple grading attempts and selects the most internally consistent one, addressing the consistency requirement.

**Skeleton:**

```python
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

RUBRIC = """
Score each dimension 1-5:
- thesis_clarity: Is the main argument clearly stated?
- argument_strength: Is the argument logically sound?
- evidence_quality: Is evidence relevant and well-cited?
- writing_mechanics: Is grammar, spelling, and structure correct?
"""

class GradeEssay(dspy.Signature):
    """Grade a student essay using the provided rubric. Be consistent and fair."""
    essay: str = dspy.InputField()
    rubric: str = dspy.InputField()
    thesis_clarity: int = dspy.OutputField(desc="Score 1-5")
    argument_strength: int = dspy.OutputField(desc="Score 1-5")
    evidence_quality: int = dspy.OutputField(desc="Score 1-5")
    writing_mechanics: int = dspy.OutputField(desc="Score 1-5")
    justification: str = dspy.OutputField(desc="2-3 sentences explaining the scores")

grader_module = dspy.ChainOfThought(GradeEssay)

def consistency_score(prediction) -> float:
    """Score a prediction by checking that all scores are in range and the
    justification references at least two rubric dimensions."""
    scores = [
        prediction.thesis_clarity,
        prediction.argument_strength,
        prediction.evidence_quality,
        prediction.writing_mechanics,
    ]
    if not all(1 <= s <= 5 for s in scores):
        return 0.0
    rubric_terms = ["thesis", "argument", "evidence", "mechanics", "writing"]
    mentions = sum(1 for t in rubric_terms if t in prediction.justification.lower())
    return min(1.0, mentions / 2)

grader = dspy.BestOfN(
    module=grader_module,
    N=3,
    reward_fn=consistency_score,
)

result = grader(essay=student_essay, rubric=RUBRIC)
print(result.thesis_clarity, result.argument_strength)
print(result.justification)
```

**Optimizer path:**
1. BootstrapFewShot on 50 human-graded essays
2. MIPROv2 if inter-rater agreement with human graders is below target

**Alternative considered:** MultiChainComparison — considered but BestOfN with an explicit scorer is preferred here because the scoring criteria are well-defined and inspectable. Single ChainOfThought — rejected because consistency across runs was measured to be insufficient without sampling.
