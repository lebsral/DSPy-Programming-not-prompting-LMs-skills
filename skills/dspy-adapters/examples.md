# dspy-adapters -- Worked Examples

## Example 1: Using JSONAdapter for reliable structured output

Extract structured product reviews into a Pydantic model with `JSONAdapter` for reliable parsing. This pattern is useful when you need guaranteed valid JSON from the LM -- for example, writing results to a database or returning them from an API.

```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal, Optional


class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "mixed"]
    key_topics: list[str] = Field(description="Main topics mentioned in the review")
    purchase_intent: bool = Field(description="Whether the reviewer would buy again")
    summary: str = Field(description="One-sentence summary of the review")


class AnalyzeReview(dspy.Signature):
    """Analyze a product review and extract structured insights."""
    review_text: str = dspy.InputField(desc="Raw product review text")
    analysis: ReviewAnalysis = dspy.OutputField()


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
adapter = dspy.JSONAdapter()
dspy.configure(lm=lm, adapter=adapter)

analyze = dspy.Predict(AnalyzeReview)

# Single review
result = analyze(
    review_text="I've been using this keyboard for 3 months now. The mechanical switches "
    "feel great and the build quality is solid. Battery life is disappointing though -- "
    "I have to charge it every 4 days. Would still recommend it for the price."
)

review = result.analysis
print(review.sentiment)        # mixed
print(review.key_topics)       # ["mechanical switches", "build quality", "battery life", "price"]
print(review.purchase_intent)  # True
print(review.summary)          # "Good keyboard with great switches and build quality but disappointing battery life."

# Batch processing -- JSONAdapter gives consistent structure across all items
reviews = [
    "Absolute garbage. Broke after one week. Returning immediately.",
    "Best purchase I've made this year. Works exactly as advertised.",
    "It's fine. Does what it says. Nothing special but no complaints either.",
]

for text in reviews:
    r = analyze(review_text=text)
    a = r.analysis
    print(f"[{a.sentiment}] intent={a.purchase_intent} -- {a.summary}")
```

Key points:
- `JSONAdapter` instructs the LM to respond with a JSON object matching the Pydantic schema
- The adapter uses `json_repair` under the hood, so minor formatting issues in the LM response are fixed automatically
- Nested Pydantic models, lists, and Literal types all work reliably with JSON output
- If you are hitting parse errors with the default `ChatAdapter`, switching to `JSONAdapter` is often the fix


## Example 2: TwoStepAdapter for complex extraction with a reasoning model

Use a reasoning model (o3-mini) for a hard analytical task, then extract structured output with a cheap model. This pattern works well when the thinking is the hard part -- the extraction is easy once the reasoning is done.

```python
import dspy
from pydantic import BaseModel
from typing import Literal


class ContractRisk(BaseModel):
    clause: str
    risk_level: Literal["low", "medium", "high", "critical"]
    explanation: str
    recommended_action: str


class ContractAnalysis(BaseModel):
    overall_risk: Literal["low", "medium", "high", "critical"]
    risks: list[ContractRisk]
    missing_clauses: list[str]
    recommendation: str


class AnalyzeContract(dspy.Signature):
    """Analyze a contract for legal risks, missing protections, and problematic clauses.
    Be thorough -- identify every potential issue."""
    contract_text: str = dspy.InputField(desc="Full text of the contract")
    party_name: str = dspy.InputField(desc="Name of the party we represent")
    analysis: ContractAnalysis = dspy.OutputField()


# --- Usage ---

# Reasoning model for deep analysis
reasoning_lm = dspy.LM("openai/o3-mini", max_tokens=16000, temperature=1.0)  # or another reasoning model

# Cheap model just for extracting structure from the reasoning output
extraction_lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-haiku-3-5-20241022", etc.

adapter = dspy.TwoStepAdapter(extraction_model=extraction_lm)
dspy.configure(lm=reasoning_lm, adapter=adapter)

analyze = dspy.ChainOfThought(AnalyzeContract)

result = analyze(
    contract_text="""
    SERVICE AGREEMENT between Acme Corp ("Provider") and ClientCo ("Client").

    1. TERM: This agreement is effective for 36 months with automatic renewal.
    2. PAYMENT: Client shall pay $10,000/month, due within 30 days of invoice.
    3. TERMINATION: Provider may terminate with 30 days notice. Client may terminate
       with 90 days notice and payment of remaining contract value.
    4. LIABILITY: Provider's total liability shall not exceed fees paid in the
       prior 3 months.
    5. IP: All work product created during the engagement belongs to Provider.
    """,
    party_name="ClientCo",
)

analysis = result.analysis
print(f"Overall risk: {analysis.overall_risk}")

for risk in analysis.risks:
    print(f"\n[{risk.risk_level.upper()}] {risk.clause}")
    print(f"  Issue: {risk.explanation}")
    print(f"  Action: {risk.recommended_action}")

print(f"\nMissing clauses: {analysis.missing_clauses}")
print(f"Recommendation: {analysis.recommendation}")
```

Key points:
- The reasoning model (o3-mini) gets a natural language prompt with no formatting constraints -- it can think freely
- The extraction model (gpt-4o-mini) reads the reasoning output and extracts the structured `ContractAnalysis` object
- This costs two LM calls per prediction, but the reasoning quality is significantly better than forcing o3-mini to output JSON directly
- Use `ChainOfThought` so the reasoning model has space to work through the problem step by step


## Example 3: Switching adapters per module in a pipeline

Use different adapters for different steps in a pipeline. Here, a summarization step uses the default `ChatAdapter` (freeform text output is fine), while a metadata extraction step uses `JSONAdapter` for reliable structured output.

```python
import dspy
from pydantic import BaseModel
from typing import Literal


class ArticleMetadata(BaseModel):
    category: Literal["tech", "business", "science", "politics", "sports", "other"]
    entities: list[str]
    key_dates: list[str]
    sentiment: Literal["positive", "negative", "neutral"]


class Summarize(dspy.Signature):
    """Write a concise 2-3 sentence summary of the article."""
    article: str = dspy.InputField(desc="Full article text")
    summary: str = dspy.OutputField(desc="2-3 sentence summary")


class ExtractMetadata(dspy.Signature):
    """Extract structured metadata from the article."""
    article: str = dspy.InputField(desc="Full article text")
    metadata: ArticleMetadata = dspy.OutputField()


class ArticleProcessor(dspy.Module):
    """Process articles: summarize and extract metadata."""

    def __init__(self):
        self.summarize = dspy.ChainOfThought(Summarize)
        self.extract = dspy.Predict(ExtractMetadata)

    def forward(self, article):
        summary = self.summarize(article=article)
        metadata = self.extract(article=article)
        return dspy.Prediction(
            summary=summary.summary,
            metadata=metadata.metadata,
        )


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

processor = ArticleProcessor()

# ChatAdapter (default) for summarization -- freeform text is fine
# JSONAdapter for extraction -- need reliable structured output
processor.extract.set_adapter(dspy.JSONAdapter())

article = """
Apple announced its new M4 chip today at a special event in Cupertino. The chip
delivers 2x faster CPU performance and 3x faster GPU performance compared to M3.
CEO Tim Cook called it "the most powerful chip we've ever created for Mac."
The new MacBook Pro models featuring M4 will be available starting November 8
at prices starting from $1,599. Analysts expect strong holiday quarter sales,
with Morgan Stanley raising its price target to $240.
"""

result = processor(article=article)

print("Summary:")
print(result.summary)

print("\nMetadata:")
meta = result.metadata
print(f"  Category: {meta.category}")
print(f"  Entities: {meta.entities}")
print(f"  Key dates: {meta.key_dates}")
print(f"  Sentiment: {meta.sentiment}")
```

Key points:
- `set_adapter()` on a module overrides the global adapter for that module only
- The summarization step uses `ChatAdapter` (the default) because it outputs plain text -- no need for JSON
- The extraction step uses `JSONAdapter` because it outputs a complex Pydantic model and needs reliable parsing
- You can also use `dspy.context(adapter=...)` for temporary overrides instead of `set_adapter()`
- This pattern scales to any pipeline -- use the simplest adapter that works for each step
