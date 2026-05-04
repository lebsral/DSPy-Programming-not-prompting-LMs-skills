# Citations Examples

## Example 1: RAG with verifiable citations

A retrieval-augmented QA system that returns cited answers:

```python
import dspy
from dspy.experimental import Citations, Document

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class CitedRAG(dspy.Module):
    def __init__(self, retriever):
        self.retriever = retriever
        self.qa = dspy.ChainOfThought(CitedQA)

    def forward(self, question):
        # Retrieve relevant documents
        passages = self.retriever(question).passages

        # Format as context with source IDs
        context = [f"[{i}] {p}" for i, p in enumerate(passages)]

        # Generate answer with citations
        result = self.qa(context=context, question=question)
        return result


class CitedQA(dspy.Signature):
    """Answer using only the provided context. Cite each claim."""
    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()


# Usage
retriever = dspy.ColBERTv2(url="http://localhost:8893/api/search")
rag = CitedRAG(retriever=retriever)
result = rag(question="What optimizers does DSPy support?")

print(result.answer)
for c in result.citations:
    print(f"  Source [{c.document_index}]: {c.cited_text[:60]}...")
```

## Example 2: Legal document citation

Extracting claims with precise source attribution for compliance:

```python
import dspy
from dspy.experimental import Citations, Document

lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929")
dspy.configure(
    lm=lm,
    adapter=dspy.ChatAdapter(adapt_to_native_lm_feature=["citations"]),
)


class LegalCitedAnswer(dspy.Signature):
    """Answer the legal question citing specific clauses from the contract.
    Every factual claim must have a citation."""
    contract_sections: list[str] = dspy.InputField(desc="numbered contract sections")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()


qa = dspy.ChainOfThought(LegalCitedAnswer)

contract_sections = [
    "Section 1.1: The term of this agreement is 24 months from the effective date.",
    "Section 2.3: Either party may terminate with 90 days written notice.",
    "Section 4.1: The monthly fee is $5,000, payable within 30 days of invoice.",
]

result = qa(
    contract_sections=contract_sections,
    question="What is the termination notice period?",
)

print(f"Answer: {result.answer}")
print(f"\nCitations:")
for c in result.citations:
    print(f"  Section [{c.document_index}]: \"{c.cited_text}\"")
    # Verify citation accuracy
    assert c.cited_text in contract_sections[c.document_index]
```

## Example 3: Multi-source research with citation validation

Research across multiple documents with validation:

```python
import dspy
from dspy.experimental import Citations, Document

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class ResearchAnswer(dspy.Signature):
    """Synthesize information from multiple sources. Cite each claim."""
    sources: list[str] = dspy.InputField(desc="labeled source documents")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="synthesis with citations")
    citations: Citations = dspy.OutputField()


def validate_citations(result, sources):
    """Check that all citations reference real text in the sources."""
    valid = 0
    invalid = 0
    for c in result.citations:
        if c.document_index < len(sources):
            if c.cited_text in sources[c.document_index]:
                valid += 1
            else:
                invalid += 1
                print(f"  INVALID: '{c.cited_text[:40]}...' not in source {c.document_index}")
        else:
            invalid += 1
            print(f"  INVALID: document_index {c.document_index} out of range")
    print(f"Citations: {valid} valid, {invalid} invalid")
    return invalid == 0


qa = dspy.ChainOfThought(ResearchAnswer)

sources = [
    "[Wikipedia] Python was created by Guido van Rossum and first released in 1991.",
    "[Official docs] Python 3.12 introduced per-interpreter GIL and improved error messages.",
    "[Blog] Python is the most popular language for machine learning and data science.",
]

result = qa(sources=sources, question="When was Python created and what is it used for?")
print(f"Answer: {result.answer}\n")
validate_citations(result, sources)
```
