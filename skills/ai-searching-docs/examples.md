# Document Search Examples

## Basic Search with ColBERTv2

```python
import dspy

# Setup
lm = dspy.LM("openai/gpt-4o-mini")
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm, rm=colbert)

# Signature
class AnswerFromDocs(dspy.Signature):
    """Answer the question based on the given context."""
    context: list[str] = dspy.InputField(desc="Retrieved passages")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Short factual answer")

# Module
class DocSearch(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(AnswerFromDocs)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Use
search = DocSearch()
result = search(question="What castle did David Gregory inherit?")
print(result.answer)

# Training data (from HotPotQA)
from datasets import load_dataset
dataset = load_dataset("hotpotqa", "fullwiki", split="train[:200]")
trainset = [
    dspy.Example(question=x["question"], answer=x["answer"]).with_inputs("question")
    for x in dataset
]

# Optimize
def metric(example, pred, trace=None):
    return pred.answer.lower() == example.answer.lower()

optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(search, trainset=trainset[:50])
```

## Multi-Step Search

For questions that need info from multiple documents:

```python
class GenerateSearchQuery(dspy.Signature):
    """Generate a search query to find information needed to answer the question."""
    context: list[str] = dspy.InputField(desc="Passages gathered so far")
    question: str = dspy.InputField()
    query: str = dspy.OutputField(desc="Focused search query for missing info")

class MultiStepSearch(dspy.Module):
    def __init__(self, num_steps=2):
        self.retrieve = dspy.Retrieve(k=3)
        self.steps = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(num_steps)]
        self.generate = dspy.ChainOfThought(AnswerFromDocs)

    def forward(self, question):
        context = []
        for step in self.steps:
            query = step(context=context, question=question).query
            new_passages = self.retrieve(query).passages
            context = list(dict.fromkeys(context + new_passages))  # deduplicate

        return self.generate(context=context, question=question)

# Works well for questions like:
# "Who was born first, the director of Jaws or the director of Titanic?"
# Requires: find director of Jaws -> find birth year -> find director of Titanic -> compare
search = MultiStepSearch(num_steps=2)
result = search(question="Who was born first, the director of Jaws or the director of Titanic?")
```

## Search with Citations

```python
class AnswerWithCitations(dspy.Signature):
    """Answer the question and cite which passages support the answer."""
    context: list[str] = dspy.InputField(desc="Retrieved passages, numbered [1], [2], etc.")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Answer to the question")
    citations: list[int] = dspy.OutputField(desc="Indices of passages that support the answer")

class CitedSearch(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought(AnswerWithCitations)

    def forward(self, question):
        passages = self.retrieve(question).passages
        numbered = [f"[{i+1}] {p}" for i, p in enumerate(passages)]
        return self.generate(context=numbered, question=question)

search = CitedSearch()
result = search(question="What is photosynthesis?")
print(f"Answer: {result.answer}")
print(f"Supported by passages: {result.citations}")
```

## Search with Custom Backend (ChromaDB)

```python
# Example: wrapping a ChromaDB collection
import chromadb

class ChromaSearch(dspy.Retrieve):
    def __init__(self, collection_name, k=3):
        super().__init__(k=k)
        client = chromadb.Client()
        self.collection = client.get_collection(collection_name)

    def forward(self, query, k=None):
        k = k or self.k
        results = self.collection.query(query_texts=[query], n_results=k)
        passages = results["documents"][0]
        return dspy.Prediction(passages=passages)

# Use it
dspy.configure(lm=lm, rm=ChromaSearch("my_docs"))
search = DocSearch()
```
