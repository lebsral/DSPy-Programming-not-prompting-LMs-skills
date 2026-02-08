# Stopping Hallucinations — Examples

## Citation-Enforced Customer Support Bot

A support bot that answers questions from your help docs and must cite every claim.

```python
import dspy
import re

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class SupportAnswer(dspy.Signature):
    """Answer the customer question using only the provided help articles.
    Cite every claim with [1], [2], etc. matching the article number."""
    articles: list[str] = dspy.InputField(desc="Numbered help articles")
    question: str = dspy.InputField(desc="Customer question")
    answer: str = dspy.OutputField(desc="Answer with inline citations")

class CitedSupportBot(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought(SupportAnswer)

    def forward(self, articles, question):
        result = self.answer(articles=articles, question=question)

        # Require citations
        citations = re.findall(r"\[(\d+)\]", result.answer)
        dspy.Assert(
            len(citations) >= 1,
            "You must cite at least one help article using [1], [2], etc."
        )

        # Verify cited articles exist
        valid = set(range(1, len(articles) + 1))
        invalid = set(int(c) for c in citations) - valid
        dspy.Assert(
            len(invalid) == 0,
            f"Invalid citations: {invalid}. Only articles [1] through [{len(articles)}] exist."
        )

        return result

# Usage
bot = CitedSupportBot()
articles = [
    "Refunds are available within 30 days of purchase. Contact support@example.com.",
    "To cancel your subscription, go to Settings > Billing > Cancel Plan.",
    "Enterprise plans include priority support with 4-hour response time.",
]
result = bot(articles=articles, question="How do I get a refund?")
print(result.answer)
# "You can get a refund within 30 days of purchase [1]. Contact support@example.com to start the process [1]."
```

## Faithfulness-Checked Medical FAQ

A medical FAQ that verifies every answer is grounded in approved content. Uses a second LM call as a faithfulness judge.

```python
class MedicalAnswer(dspy.Signature):
    """Answer the health question using only the approved medical content."""
    approved_content: list[str] = dspy.InputField(desc="Approved medical information")
    question: str = dspy.InputField(desc="Patient question")
    answer: str = dspy.OutputField(desc="Answer grounded in approved content")

class VerifyMedical(dspy.Signature):
    """Check if the answer is fully supported by the approved medical content.
    Be strict: flag anything not explicitly stated in the content."""
    approved_content: list[str] = dspy.InputField()
    answer: str = dspy.InputField()
    is_supported: bool = dspy.OutputField(desc="Is every claim in the answer supported?")
    unsupported_claims: list[str] = dspy.OutputField(desc="Claims not in the approved content")

class SafeMedicalFAQ(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought(MedicalAnswer)
        self.verify = dspy.Predict(VerifyMedical)

    def forward(self, approved_content, question):
        result = self.answer(approved_content=approved_content, question=question)

        # Strict faithfulness check
        check = self.verify(approved_content=approved_content, answer=result.answer)
        dspy.Assert(
            check.is_supported,
            f"Answer contains claims not in approved content: {check.unsupported_claims}. "
            "Rewrite using ONLY information from the approved medical content."
        )

        return result

# Usage
faq = SafeMedicalFAQ()
approved = [
    "Ibuprofen is an over-the-counter NSAID. Standard adult dose is 200-400mg every 4-6 hours.",
    "Do not exceed 1200mg per day without medical supervision.",
    "Common side effects include stomach upset and dizziness.",
]
result = faq(approved_content=approved, question="What's the right dose of ibuprofen?")
```

## Grounded Q&A with Retrieval

End-to-end example: retrieve docs, generate answer, verify faithfulness. Combines `/ai-searching-docs` with hallucination prevention.

```python
class GroundedAnswer(dspy.Signature):
    """Answer the question using only the retrieved documents. Cite sources."""
    documents: list[str] = dspy.InputField(desc="Retrieved source documents")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Grounded answer with [1], [2] citations")

class CheckGrounding(dspy.Signature):
    """Verify the answer only contains information from the documents."""
    documents: list[str] = dspy.InputField()
    answer: str = dspy.InputField()
    is_grounded: bool = dspy.OutputField()
    fabricated_info: list[str] = dspy.OutputField(desc="Information not in the documents")

class GroundedQA(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.answer = dspy.ChainOfThought(GroundedAnswer)
        self.check = dspy.Predict(CheckGrounding)

    def forward(self, question):
        # Retrieve relevant docs
        docs = self.retrieve(question).passages

        # Generate grounded answer
        result = self.answer(documents=docs, question=question)

        # Verify grounding
        check = self.check(documents=docs, answer=result.answer)
        dspy.Assert(
            check.is_grounded,
            f"Answer includes fabricated information: {check.fabricated_info}. "
            "Rewrite using only facts from the retrieved documents."
        )

        return dspy.Prediction(
            answer=result.answer,
            sources=docs,
        )

# Requires a retriever to be configured:
# import dspy
# from dspy.retrieve.chromadb_rm import ChromadbRM
# retriever = ChromadbRM(collection_name="my_docs", persist_directory="./chroma")
# dspy.configure(lm=lm, rm=retriever)
#
# qa = GroundedQA()
# result = qa(question="What is our refund policy?")
```

## Cross-Checked Financial Report

For high-stakes outputs, generate the answer twice and compare. If two independent generations disagree, flag it.

```python
class FinancialSummary(dspy.Signature):
    """Summarize the financial data accurately."""
    data: str = dspy.InputField(desc="Raw financial data")
    question: str = dspy.InputField(desc="What to summarize")
    summary: str = dspy.OutputField(desc="Accurate financial summary")

class CheckAgreement(dspy.Signature):
    """Do these two financial summaries agree on all numbers and claims?"""
    summary_a: str = dspy.InputField()
    summary_b: str = dspy.InputField()
    agree: bool = dspy.OutputField(desc="Do they agree on all facts and figures?")
    discrepancies: list[str] = dspy.OutputField(desc="Specific disagreements")

class CrossCheckedFinance(dspy.Module):
    def __init__(self):
        self.gen_a = dspy.ChainOfThought(FinancialSummary)
        self.gen_b = dspy.ChainOfThought(FinancialSummary)
        self.compare = dspy.Predict(CheckAgreement)

    def forward(self, data, question):
        a = self.gen_a(data=data, question=question)
        b = self.gen_b(data=data, question=question)

        check = self.compare(summary_a=a.summary, summary_b=b.summary)
        dspy.Assert(
            check.agree,
            f"Two independent summaries disagree: {check.discrepancies}. "
            "Regenerate with careful attention to the source data."
        )

        return a

# Usage
reporter = CrossCheckedFinance()
result = reporter(
    data="Q1 Revenue: $2.3M, Q2 Revenue: $2.8M, Q1 Expenses: $1.9M, Q2 Expenses: $2.1M",
    question="Summarize revenue growth and profitability trend",
)
```

## Confidence-Gated Legal Q&A

Route low-confidence answers to human lawyers instead of showing them to users.

```python
class LegalAnswer(dspy.Signature):
    """Answer the legal question based on the provided statutes and precedents."""
    sources: list[str] = dspy.InputField(desc="Legal sources")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="0.0 to 1.0")
    confidence_reason: str = dspy.OutputField(desc="Why this confidence level")

class GatedLegalQA(dspy.Module):
    def __init__(self, threshold=0.8):
        self.respond = dspy.ChainOfThought(LegalAnswer)
        self.threshold = threshold

    def forward(self, sources, question):
        result = self.respond(sources=sources, question=question)

        if result.confidence < self.threshold:
            return dspy.Prediction(
                answer=None,
                needs_lawyer=True,
                confidence=result.confidence,
                reason=result.confidence_reason,
                draft=result.answer,  # lawyer can review the draft
            )

        return dspy.Prediction(
            answer=result.answer,
            needs_lawyer=False,
            confidence=result.confidence,
        )

# Usage
qa = GatedLegalQA(threshold=0.8)
result = qa(
    sources=["Section 230 provides immunity for platforms..."],
    question="Are we liable for user-generated content?",
)
if result.needs_lawyer:
    print(f"Routing to human lawyer (confidence: {result.confidence})")
    print(f"Reason: {result.reason}")
    print(f"Draft for review: {result.draft}")
else:
    print(result.answer)
```
