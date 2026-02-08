# Task Decomposition Examples

## Medical Report Extraction (Sequential Pattern)

The pattern that took error rates from 40% to near-zero: identify panels first, then extract results per panel.

```python
import dspy
from pydantic import BaseModel, Field

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Step 1: Identify all panels
class IdentifyPanels(dspy.Signature):
    """Identify all lab test panels in the medical report."""
    report: str = dspy.InputField(desc="Medical lab report text")
    panel_names: list[str] = dspy.OutputField(desc="Names of all test panels found")

# Step 2: Extract results per panel
class LabResult(BaseModel):
    test_name: str
    value: str
    unit: str
    reference_range: str
    flag: str = Field(description="'normal', 'high', or 'low'")

class ExtractPanelResults(dspy.Signature):
    """Extract all test results for a specific panel."""
    report: str = dspy.InputField(desc="Medical lab report")
    panel_name: str = dspy.InputField(desc="The specific panel to extract")
    results: list[LabResult] = dspy.OutputField(desc="Test results for this panel")

class MedicalReportExtractor(dspy.Module):
    def __init__(self):
        self.identify = dspy.ChainOfThought(IdentifyPanels)
        self.extract = dspy.ChainOfThought(ExtractPanelResults)

    def forward(self, report: str):
        panels = self.identify(report=report)
        all_results = {}
        for panel in panels.panel_names:
            result = self.extract(report=report, panel_name=panel)
            all_results[panel] = result.results
        return dspy.Prediction(panels=panels.panel_names, results=all_results)

extractor = MedicalReportExtractor()
result = extractor(report="""
COMPREHENSIVE METABOLIC PANEL
Glucose: 95 mg/dL (70-100) Normal
BUN: 18 mg/dL (7-20) Normal
Creatinine: 1.1 mg/dL (0.7-1.3) Normal
Sodium: 140 mEq/L (136-145) Normal
Potassium: 4.2 mEq/L (3.5-5.0) Normal

COMPLETE BLOOD COUNT
WBC: 7.5 x10^3/uL (4.5-11.0) Normal
RBC: 4.8 x10^6/uL (4.5-5.5) Normal
Hemoglobin: 14.2 g/dL (13.5-17.5) Normal
Hematocrit: 42.1% (38.3-48.6) Normal
Platelets: 250 x10^3/uL (150-400) Normal

LIPID PANEL
Total Cholesterol: 220 mg/dL (<200) High
LDL: 145 mg/dL (<100) High
HDL: 55 mg/dL (>40) Normal
Triglycerides: 160 mg/dL (<150) High
""")

for panel, results in result.results.items():
    print(f"\n{panel}:")
    for r in results:
        print(f"  {r.test_name}: {r.value} {r.unit} [{r.flag}]")
```

### Why decomposition wins here

A single "extract all results" prompt misses items when reports have 3+ panels. The model loses track of which values belong to which tests. By extracting per panel, each call is focused on 3-8 results instead of 15-20.

## Invoice Line-Item Extraction (Identify-then-Process)

```python
class IdentifyLineItems(dspy.Signature):
    """Identify every line item in the invoice. Include all items, even small charges."""
    invoice_text: str = dspy.InputField(desc="Raw invoice text")
    item_descriptions: list[str] = dspy.OutputField(desc="Brief description of each item found")

class LineItemDetail(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class ExtractLineItem(dspy.Signature):
    """Extract exact details for one specific line item."""
    invoice_text: str = dspy.InputField(desc="The full invoice text")
    item_description: str = dspy.InputField(desc="The item to extract details for")
    details: LineItemDetail = dspy.OutputField()

class InvoiceExtractor(dspy.Module):
    def __init__(self):
        self.identify = dspy.ChainOfThought(IdentifyLineItems)
        self.extract_item = dspy.ChainOfThought(ExtractLineItem)

    def forward(self, invoice_text: str):
        items = self.identify(invoice_text=invoice_text)
        line_items = []
        for desc in items.item_descriptions:
            result = self.extract_item(invoice_text=invoice_text, item_description=desc)
            line_items.append(result.details)
        return dspy.Prediction(line_items=line_items)

extractor = InvoiceExtractor()
result = extractor(invoice_text="""
INVOICE #2024-0847
Vendor: Industrial Supply Co.
Date: 2024-11-15

1. Steel bolts M8x30 (box of 100)    x5     $12.50    $62.50
2. Rubber gaskets 2" ID              x20     $3.25     $65.00
3. Hydraulic fluid ISO 46 (5L)        x2     $45.00    $90.00
4. Safety gloves (pair)              x10     $8.99     $89.90
5. Cable ties 300mm (bag of 100)      x3     $4.50     $13.50
6. Shipping & handling                x1     $15.00    $15.00
7. Rush delivery surcharge            x1     $25.00    $25.00

Subtotal: $360.90
Tax (8.5%): $30.68
Total: $391.58
""")

for item in result.line_items:
    print(f"  {item.description}: {item.quantity} x ${item.unit_price} = ${item.total}")

# The identify step catches items 6 and 7 (shipping, surcharge) that single-step often misses
```

## Resume Parsing (Identify Sections, Then Extract)

```python
class IdentifySections(dspy.Signature):
    """Identify all sections in the resume."""
    resume_text: str = dspy.InputField(desc="Raw resume text")
    sections: list[str] = dspy.OutputField(
        desc="Section names found, e.g. 'contact', 'experience', 'education', 'skills'"
    )

class ExperienceEntry(BaseModel):
    company: str
    title: str
    dates: str
    highlights: list[str]

class ExtractExperience(dspy.Signature):
    """Extract work experience entries from the resume."""
    resume_text: str = dspy.InputField()
    entries: list[ExperienceEntry] = dspy.OutputField()

class EducationEntry(BaseModel):
    institution: str
    degree: str
    year: str

class ExtractEducation(dspy.Signature):
    """Extract education entries from the resume."""
    resume_text: str = dspy.InputField()
    entries: list[EducationEntry] = dspy.OutputField()

class ExtractSkills(dspy.Signature):
    """Extract the list of skills from the resume."""
    resume_text: str = dspy.InputField()
    skills: list[str] = dspy.OutputField()

class ExtractContact(dspy.Signature):
    """Extract contact information from the resume."""
    resume_text: str = dspy.InputField()
    name: str = dspy.OutputField()
    email: str = dspy.OutputField()
    phone: str = dspy.OutputField()

class ResumeParser(dspy.Module):
    def __init__(self):
        self.identify = dspy.ChainOfThought(IdentifySections)
        self.extractors = {
            "experience": dspy.ChainOfThought(ExtractExperience),
            "education": dspy.ChainOfThought(ExtractEducation),
            "skills": dspy.ChainOfThought(ExtractSkills),
            "contact": dspy.ChainOfThought(ExtractContact),
        }

    def forward(self, resume_text: str):
        sections = self.identify(resume_text=resume_text)

        results = {}
        for section in sections.sections:
            section_key = section.lower().strip()
            extractor = self.extractors.get(section_key)
            if extractor:
                results[section_key] = extractor(resume_text=resume_text)

        return dspy.Prediction(sections=sections.sections, extracted=results)

parser = ResumeParser()
result = parser(resume_text="""
JANE SMITH
jane.smith@email.com | (555) 123-4567

EXPERIENCE
Senior Engineer, Acme Corp (2021-Present)
- Led migration from monolith to microservices
- Reduced API latency by 40%

Software Engineer, StartupXYZ (2018-2021)
- Built real-time data pipeline processing 1M events/day
- Mentored 3 junior engineers

EDUCATION
B.S. Computer Science, State University, 2018

SKILLS
Python, Go, Kubernetes, PostgreSQL, Redis, AWS, Terraform
""")

print(f"Sections found: {result.sections}")
```

## Comparing Single-Step vs Decomposed

```python
from dspy.evaluate import Evaluate

# Single-step baseline
class ExtractAllItems(dspy.Signature):
    """Extract all line items from the invoice."""
    invoice_text: str = dspy.InputField()
    line_items: list[LineItemDetail] = dspy.OutputField()

single_step = dspy.ChainOfThought(ExtractAllItems)
decomposed = InvoiceExtractor()

# Metric: recall (what fraction of gold items were found)
def recall_metric(example, prediction, trace=None):
    gold = set(item.lower() for item in example.item_names)
    pred = set(item.description.lower() for item in prediction.line_items)
    if not gold:
        return 1.0
    return len(gold & pred) / len(gold)

# Evaluate on simple (1-3 items) and complex (5+ items) invoices
simple_set = [ex for ex in devset if len(ex.item_names) <= 3]
complex_set = [ex for ex in devset if len(ex.item_names) >= 5]

simple_eval = Evaluate(devset=simple_set, metric=recall_metric, num_threads=4)
complex_eval = Evaluate(devset=complex_set, metric=recall_metric, num_threads=4)

print("Simple invoices (1-3 items):")
print(f"  Single-step: {simple_eval(single_step):.1f}%")
print(f"  Decomposed:  {simple_eval(decomposed):.1f}%")

print("Complex invoices (5+ items):")
print(f"  Single-step: {complex_eval(single_step):.1f}%")
print(f"  Decomposed:  {complex_eval(decomposed):.1f}%")

# Typical results:
# Simple: ~95% vs ~97% (small difference)
# Complex: ~70% vs ~95% (decomposition shines)
```
