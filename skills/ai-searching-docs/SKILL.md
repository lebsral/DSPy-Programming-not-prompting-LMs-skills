---
name: ai-searching-docs
description: Build AI that searches your documents and answers questions. Use when building a knowledge base, help center Q&A, chatting with documents, answering questions from a database, search-and-answer over internal docs, customer support bot, or FAQ system. Powered by DSPy RAG (retrieval-augmented generation).
---

# Build AI-Powered Document Search

Guide the user through building an AI that searches documents and answers questions accurately. Uses DSPy's RAG (retrieval-augmented generation) pattern — retrieve relevant passages, then generate an answer grounded in them.

## Step 1: Understand the setup

Ask the user:
1. **What documents are you searching?** (PDFs, web pages, database, help articles, etc.)
2. **What kind of questions will users ask?** (factual lookups, how-to questions, multi-step research?)
3. **Do you have a search backend already?** (Elasticsearch, Pinecone, ChromaDB, pgvector, etc.)
4. **Do questions need info from multiple documents?** (simple lookup vs. combining info)

## Step 2: Build the search-and-answer pipeline

### Basic: search then answer

```python
import dspy

class AnswerFromDocs(dspy.Signature):
    """Answer the question based on the given context."""
    context: list[str] = dspy.InputField(desc="Relevant passages from the knowledge base")
    question: str = dspy.InputField(desc="User's question")
    answer: str = dspy.OutputField(desc="Answer grounded in the context")

class DocSearch(dspy.Module):
    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.answer = dspy.ChainOfThought(AnswerFromDocs)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)
```

### Configure the search backend

DSPy supports multiple search backends. Set up via `dspy.configure`:

```python
# ColBERTv2 (hosted)
colbert = dspy.ColBERTv2(url="http://your-server:port/endpoint")
dspy.configure(lm=lm, rm=colbert)

# Or wrap your own search (Elasticsearch, Pinecone, pgvector, etc.)
class MySearchBackend(dspy.Retrieve):
    def forward(self, query, k=None):
        k = k or self.k
        # Your search logic here
        results = your_search_function(query, top_k=k)
        return dspy.Prediction(passages=[r["text"] for r in results])
```

## Step 3: Multi-document search (for complex questions)

When questions need info from multiple places:

```python
class GenerateSearchQuery(dspy.Signature):
    """Generate a search query to find missing information."""
    context: list[str] = dspy.InputField(desc="Information gathered so far")
    question: str = dspy.InputField(desc="The question to answer")
    query: str = dspy.OutputField(desc="Search query to find missing information")

class MultiStepSearch(dspy.Module):
    def __init__(self, num_passages=3, num_searches=2):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(num_searches)]
        self.answer = dspy.ChainOfThought(AnswerFromDocs)

    def forward(self, question):
        context = []

        for hop in self.generate_query:
            query = hop(context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        return self.answer(context=context, question=question)

def deduplicate(passages):
    seen = set()
    result = []
    for p in passages:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result
```

## Step 4: Test the quality

```python
def search_metric(example, prediction, trace=None):
    # Exact match (simple)
    return prediction.answer == example.answer

# Or use an AI judge for open-ended answers
class JudgeAnswer(dspy.Signature):
    """Is the predicted answer correct given the expected answer?"""
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    is_correct: bool = dspy.OutputField()

def judge_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeAnswer)
    result = judge(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=prediction.answer,
    )
    return result.is_correct
```

## Step 5: Improve accuracy

```python
optimizer = dspy.BootstrapFewShot(metric=search_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(DocSearch(), trainset=trainset)
```

## Key patterns

- **Always use `ChainOfThought`** for the answer step — reasoning helps ground answers in the documents
- **Include context in the signature** so the AI knows to use the retrieved passages
- **Multi-step search for complex questions** — if one search isn't enough, chain search queries
- **Use `dspy.Assert`** to ensure answers actually cite the documents
- **Separate search from answer generation** — optimize each independently

## Additional resources

- For worked examples, see [examples.md](examples.md)
- Next: `/ai-improving-accuracy` to measure and improve your AI
