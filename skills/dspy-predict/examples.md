# dspy-predict -- Worked Examples

## Example 1: Simple text classification with Predict

Classify customer feedback into categories using `Predict` with a `Literal` type constraint. No reasoning needed -- the mapping is direct.

```python
import dspy
from typing import Literal


class ClassifyFeedback(dspy.Signature):
    """Classify customer feedback into a category for the product team."""
    feedback: str = dspy.InputField(desc="Raw customer feedback text")
    category: Literal["bug", "feature_request", "praise", "question", "other"] = dspy.OutputField()
    priority: Literal["low", "medium", "high"] = dspy.OutputField(
        desc="How urgently this needs attention"
    )


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

classify = dspy.Predict(ClassifyFeedback)

# Single prediction
result = classify(feedback="The export button doesn't work on Safari, I get a blank page")
print(result.category)  # bug
print(result.priority)  # high

# Try several inputs
samples = [
    "Would be great if you added dark mode",
    "Your app saved me hours of work this week!",
    "How do I connect my Slack workspace?",
    "App crashes every time I upload a CSV larger than 10MB",
]

for text in samples:
    r = classify(feedback=text)
    print(f"[{r.priority}] {r.category}: {text}")
```

Key points:
- `Literal` constrains the LM to only return values from the listed options
- `Predict` is ideal here because classification is a direct mapping -- no reasoning steps needed
- Two output fields (`category` and `priority`) are generated in a single LM call
- If accuracy is too low, swap `dspy.Predict` for `dspy.ChainOfThought` -- everything else stays the same


## Example 2: Multi-field extraction

Extract structured data from unstructured text using `Predict` with a Pydantic output type. This pattern is common for parsing emails, invoices, resumes, or any semi-structured text.

```python
import dspy
from pydantic import BaseModel, Field
from typing import Optional


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "US"


class PersonInfo(BaseModel):
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    address: Optional[Address] = None


class ExtractPerson(dspy.Signature):
    """Extract structured person information from the text.
    If a field is not mentioned, leave it as null."""
    text: str = dspy.InputField(desc="Unstructured text containing person information")
    person: PersonInfo = dspy.OutputField(desc="Extracted person details")


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

extractor = dspy.Predict(ExtractPerson)

# Full info
result = extractor(
    text="Please send the contract to Maria Garcia, VP of Engineering at Acme Corp. "
         "Her email is maria@acme.com and she's at 123 Main St, Austin, TX 78701."
)
person = result.person
print(person.full_name)       # Maria Garcia
print(person.role)            # VP of Engineering
print(person.company)         # Acme Corp
print(person.email)           # maria@acme.com
print(person.address.city)    # Austin
print(person.address.state)   # TX

# Partial info -- missing fields become None
result = extractor(text="Got a call from Bob at 555-0199")
person = result.person
print(person.full_name)   # Bob
print(person.phone)       # 555-0199
print(person.email)       # None
print(person.company)     # None
```

Key points:
- Pydantic `BaseModel` gives you nested, validated, typed output from a single LM call
- `Optional` fields handle cases where information is missing in the source text
- `Predict` is the right module here because extraction is a direct mapping from text to structure
- The extracted `PersonInfo` object works like any Python object -- pass it to your database, API, or next pipeline step


## Example 3: Batch processing pattern

Process a list of items with `Predict` inside a `dspy.Module`. This pattern keeps your batch logic optimizable by DSPy.

```python
import dspy
from typing import Literal


class TagItem(dspy.Signature):
    """Tag a product listing with relevant categories for search."""
    title: str = dspy.InputField(desc="Product listing title")
    description: str = dspy.InputField(desc="Product listing description")
    primary_category: Literal[
        "electronics", "clothing", "home", "sports", "books", "other"
    ] = dspy.OutputField()
    tags: list[str] = dspy.OutputField(desc="3-5 search tags for this product")


class ProductTagger(dspy.Module):
    """Tag a batch of product listings for a search index."""

    def __init__(self):
        self.tag = dspy.Predict(TagItem)

    def forward(self, products: list[dict]):
        results = []
        for product in products:
            tagged = self.tag(
                title=product["title"],
                description=product["description"],
            )
            results.append({
                "id": product["id"],
                "title": product["title"],
                "primary_category": tagged.primary_category,
                "tags": tagged.tags,
            })
        return dspy.Prediction(tagged_products=results)


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

tagger = ProductTagger()

products = [
    {
        "id": "SKU-001",
        "title": "Wireless Noise-Canceling Headphones",
        "description": "Bluetooth over-ear headphones with 30hr battery and ANC.",
    },
    {
        "id": "SKU-002",
        "title": "Organic Cotton T-Shirt",
        "description": "Soft breathable crew neck tee, available in 6 colors.",
    },
    {
        "id": "SKU-003",
        "title": "Cast Iron Dutch Oven 6qt",
        "description": "Enameled cast iron pot, oven-safe to 500F, dishwasher safe.",
    },
]

result = tagger(products=products)

for item in result.tagged_products:
    print(f"\n{item['title']} ({item['id']})")
    print(f"  Category: {item['primary_category']}")
    print(f"  Tags: {item['tags']}")


# --- Optimization ---

# To optimize, define a metric and provide labeled examples:
#
# def tagging_metric(example, prediction, trace=None):
#     """Check if the primary category matches the gold label."""
#     gold = example.tagged_products
#     pred = prediction.tagged_products
#     correct = sum(
#         1 for g, p in zip(gold, pred)
#         if g["primary_category"] == p["primary_category"]
#     )
#     return correct / len(gold)
#
# optimizer = dspy.BootstrapFewShot(metric=tagging_metric, max_bootstrapped_demos=4)
# optimized_tagger = optimizer.compile(tagger, trainset=trainset)
# optimized_tagger.save("product_tagger.json")
```

Key points:
- Wrapping the loop in a `dspy.Module` makes the entire batch pipeline optimizable
- The `Predict` sub-module is declared in `__init__` so optimizers can discover and tune it
- Each item gets its own LM call -- DSPy handles the prompting for each one
- `dspy.Prediction` bundles the batch results into a clean return value
- The commented optimization section shows how to evaluate and tune the batch pipeline end-to-end
