# Data Parsing Examples

## Entity Extraction from News

```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class Entity(BaseModel):
    name: str = Field(description="The entity name as it appears in text")
    type: Literal["person", "organization", "location", "date", "money"] = Field(
        description="Entity type"
    )

class ParseEntities(dspy.Signature):
    """Extract all named entities from the news article."""
    article: str = dspy.InputField(desc="News article text")
    entities: list[Entity] = dspy.OutputField(desc="All named entities found")

parser = dspy.ChainOfThought(ParseEntities)

result = parser(
    article="Apple CEO Tim Cook announced a $3 billion investment in Austin, Texas on January 15, 2025."
)
for entity in result.entities:
    print(f"  {entity.name} ({entity.type})")
# Apple (organization)
# Tim Cook (person)
# $3 billion (money)
# Austin, Texas (location)
# January 15, 2025 (date)
```

## Invoice Parsing

```python
class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    date: str
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float

class ParseInvoice(dspy.Signature):
    """Parse an invoice and extract all structured fields."""
    invoice_text: str = dspy.InputField(desc="Raw invoice text")
    invoice: Invoice = dspy.OutputField(desc="Structured invoice data")

parser = dspy.ChainOfThought(ParseInvoice)

result = parser(invoice_text="""
INVOICE #2024-001
Vendor: Acme Corp
Date: 2024-03-15

Widget A     x10    $5.00    $50.00
Widget B     x3     $12.50   $37.50

Subtotal: $87.50
Tax (8%): $7.00
Total: $94.50
""")
print(result.invoice)
```

## Resume/CV Parsing

```python
class Experience(BaseModel):
    company: str
    title: str
    duration: str
    description: str

class Education(BaseModel):
    institution: str
    degree: str
    year: str

class ResumeData(BaseModel):
    name: str
    email: str
    phone: str
    skills: list[str]
    experience: list[Experience]
    education: list[Education]

class ParseResume(dspy.Signature):
    """Extract structured data from a resume."""
    resume_text: str = dspy.InputField()
    data: ResumeData = dspy.OutputField()

parser = dspy.ChainOfThought(ParseResume)
```

## Key-Value Extraction from Forms

```python
class ParseFormFields(dspy.Signature):
    """Extract key-value pairs from a form or document."""
    document: str = dspy.InputField(desc="Form or document text")
    fields: dict[str, str] = dspy.OutputField(desc="Extracted field names and values")

parser = dspy.ChainOfThought(ParseFormFields)

result = parser(document="""
Patient Name: Jane Smith
DOB: 04/12/1985
Insurance ID: BC-12345-XY
Reason for Visit: Annual checkup
Allergies: Penicillin, shellfish
""")
print(result.fields)
# {'Patient Name': 'Jane Smith', 'DOB': '04/12/1985', ...}
```

## Relation Extraction

```python
class Relation(BaseModel):
    subject: str
    predicate: str = Field(description="The relationship type, e.g. 'works_at', 'founded', 'located_in'")
    object: str

class ParseRelations(dspy.Signature):
    """Extract semantic relations between entities in the text."""
    text: str = dspy.InputField()
    relations: list[Relation] = dspy.OutputField()

parser = dspy.ChainOfThought(ParseRelations)

result = parser(text="Elon Musk founded SpaceX in Hawthorne, California.")
for r in result.relations:
    print(f"  {r.subject} --{r.predicate}--> {r.object}")
# Elon Musk --founded--> SpaceX
# SpaceX --located_in--> Hawthorne, California
```
