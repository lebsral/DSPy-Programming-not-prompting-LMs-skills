# Signature Examples

## Email Classifier

Classify incoming emails into categories using a class-based signature with `Literal` output.

```python
import dspy
from typing import Literal

# Configure any LM provider
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

CATEGORIES = ["sales_inquiry", "support_request", "billing", "spam", "partnership", "other"]

class ClassifyEmail(dspy.Signature):
    """Classify an incoming email by intent for routing to the correct team.
    Focus on the sender's primary intent, not just keywords."""
    subject: str = dspy.InputField(desc="Email subject line")
    body: str = dspy.InputField(desc="Email body text")
    category: Literal[tuple(CATEGORIES)] = dspy.OutputField(desc="Primary intent category")
    priority: Literal["low", "medium", "high"] = dspy.OutputField(
        desc="Urgency: high = revenue impact or angry customer, medium = needs response today, low = can wait"
    )

classifier = dspy.ChainOfThought(ClassifyEmail)

# Test it
result = classifier(
    subject="Urgent: billing discrepancy on last invoice",
    body="Hi, I was charged $500 instead of the agreed $350 on invoice #1042. "
         "This is the third billing error this quarter. Please fix this ASAP or "
         "we will need to reconsider our contract."
)
print(f"Category: {result.category}")    # billing
print(f"Priority: {result.priority}")    # high
print(f"Reasoning: {result.reasoning}")  # ChainOfThought explains its logic

# Batch classify
emails = [
    {"subject": "Partnership opportunity", "body": "We'd love to integrate your API into our platform."},
    {"subject": "Can't log in", "body": "Password reset isn't working, tried 5 times."},
    {"subject": "RE: RE: RE: pricing", "body": "What's the cost for 500 seats?"},
]

for email in emails:
    r = classifier(subject=email["subject"], body=email["body"])
    print(f"  {email['subject'][:40]:40s} -> {r.category:20s} ({r.priority})")
```

**Expected output:**

```
Category: billing
Priority: high
Reasoning: The customer reports a billing error ...
  Partnership opportunity                  -> partnership           (medium)
  Can't log in                             -> support_request       (medium)
  RE: RE: RE: pricing                      -> sales_inquiry         (low)
```

## Invoice Parser

Extract structured invoice data using a Pydantic model as the output type.

```python
import dspy
from pydantic import BaseModel, Field
from typing import Optional

# Configure any LM provider
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class LineItem(BaseModel):
    description: str = Field(description="Item or service description")
    quantity: int = Field(description="Number of units", ge=1)
    unit_price: float = Field(description="Price per unit in dollars", ge=0)
    total: float = Field(description="Line total (quantity * unit_price)", ge=0)

class InvoiceData(BaseModel):
    vendor_name: str
    invoice_number: Optional[str] = None
    invoice_date: str = Field(description="Date in YYYY-MM-DD format")
    due_date: Optional[str] = Field(default=None, description="Due date in YYYY-MM-DD format")
    items: list[LineItem]
    subtotal: float
    tax: Optional[float] = None
    total: float
    currency: str = Field(default="USD", description="ISO 4217 currency code")

class ParseInvoice(dspy.Signature):
    """Extract structured invoice data from raw text. Return None for optional
    fields that are not present in the text -- do not guess."""
    raw_text: str = dspy.InputField(desc="Raw invoice text, possibly messy or OCR output")
    invoice: InvoiceData = dspy.OutputField(desc="Structured invoice data")

parser = dspy.ChainOfThought(ParseInvoice)

# Test with a sample invoice
invoice_text = """
INVOICE #2025-0042
From: Stellar Design Co.
Date: March 3, 2025
Due: April 2, 2025

Description                 Qty    Unit Price    Total
---------------------------------------------------------
Website redesign             1      $3,500.00    $3,500.00
Logo design (revisions x3)  1      $1,200.00    $1,200.00
Stock photography           10        $25.00      $250.00
Hosting setup                1       $150.00      $150.00

                              Subtotal:  $5,100.00
                              Tax (8%):    $408.00
                              TOTAL:     $5,508.00

Payment terms: Net 30
"""

result = parser(raw_text=invoice_text)
inv = result.invoice

print(f"Vendor: {inv.vendor_name}")
print(f"Invoice #: {inv.invoice_number}")
print(f"Date: {inv.invoice_date}")
print(f"Due: {inv.due_date}")
print(f"\nLine items:")
for item in inv.items:
    print(f"  {item.description:30s}  {item.quantity:3d} x ${item.unit_price:>8.2f} = ${item.total:>9.2f}")
print(f"\nSubtotal: ${inv.subtotal:,.2f}")
print(f"Tax:      ${inv.tax:,.2f}")
print(f"Total:    ${inv.total:,.2f}")

# Convert to dict for JSON/API use
print(f"\nAs dict: {inv.model_dump()}")
```

**Expected output:**

```
Vendor: Stellar Design Co.
Invoice #: 2025-0042
Date: 2025-03-03
Due: 2025-04-02

Line items:
  Website redesign                1 x $ 3500.00 = $  3500.00
  Logo design (revisions x3)     1 x $ 1200.00 = $  1200.00
  Stock photography              10 x $   25.00 = $   250.00
  Hosting setup                   1 x $  150.00 = $   150.00

Subtotal: $5,100.00
Tax:      $408.00
Total:    $5,508.00
```

## Multi-Output Content Analysis

Analyze a piece of content across multiple dimensions with different output types.

```python
import dspy
from typing import Literal
from pydantic import BaseModel

# Configure any LM provider
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class AudienceProfile(BaseModel):
    primary_audience: str
    expertise_level: str
    industry: str

class AnalyzeContent(dspy.Signature):
    """Perform a comprehensive analysis of a blog post or article.
    Evaluate readability for a general tech audience."""
    title: str = dspy.InputField(desc="Article title")
    content: str = dspy.InputField(desc="Full article text")

    # String output
    summary: str = dspy.OutputField(desc="2-3 sentence summary of the article")

    # Literal output
    content_type: Literal["tutorial", "opinion", "news", "case_study", "review"] = dspy.OutputField()

    # List output
    topics: list[str] = dspy.OutputField(desc="Key topics covered, max 5")

    # Integer output
    estimated_read_minutes: int = dspy.OutputField(desc="Estimated reading time in minutes")

    # Float output
    readability_score: float = dspy.OutputField(desc="Readability from 0.0 (very technical) to 1.0 (very accessible)")

    # Boolean output
    has_code_examples: bool = dspy.OutputField(desc="Whether the article contains code snippets")

    # Pydantic model output
    audience: AudienceProfile = dspy.OutputField(desc="Target audience profile")

analyzer = dspy.ChainOfThought(AnalyzeContent)

# Test with a sample article
result = analyzer(
    title="Building Real-Time Data Pipelines with Apache Kafka and Python",
    content="""
    In this tutorial, we'll walk through setting up a real-time data pipeline
    using Apache Kafka as the message broker and Python consumers. We'll cover
    topic configuration, producer setup, consumer groups, and error handling.

    First, install the confluent-kafka package:
        pip install confluent-kafka

    Here's a basic producer:
        from confluent_kafka import Producer
        producer = Producer({'bootstrap.servers': 'localhost:9092'})
        producer.produce('my-topic', value='hello world')
        producer.flush()

    For production use, you'll want to handle serialization with Avro or Protobuf,
    implement proper error callbacks, and monitor consumer lag. We've used this
    pattern at scale processing 50,000 events per second with a team of 3 engineers.
    """
)

print(f"Summary: {result.summary}")
print(f"Type: {result.content_type}")
print(f"Topics: {result.topics}")
print(f"Read time: {result.estimated_read_minutes} min")
print(f"Readability: {result.readability_score}")
print(f"Has code: {result.has_code_examples}")
print(f"Audience: {result.audience.primary_audience} ({result.audience.expertise_level})")
print(f"Industry: {result.audience.industry}")
```

**Expected output:**

```
Summary: A hands-on tutorial for building real-time data pipelines using Apache Kafka with Python. Covers producer setup, consumer groups, and production considerations for high-throughput event processing.
Type: tutorial
Topics: ['Apache Kafka', 'Python', 'data pipelines', 'real-time processing', 'event streaming']
Read time: 4
Readability: 0.6
Has code: True
Audience: Backend developers (intermediate)
Industry: software engineering
```
