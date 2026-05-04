# Examples - AI Understanding Images

---

## Example 1 - Product photo categorizer

Classify e-commerce product photos and extract structured attributes.

```python
import dspy
from pydantic import BaseModel
from typing import List, Literal

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class ProductAttributes(BaseModel):
    category: str
    subcategory: str
    primary_color: str
    secondary_colors: List[str]
    condition: Literal["new", "like_new", "used", "damaged"]
    material: str
    description: str

class CategorizeProduct(dspy.Signature):
    """
    Examine the product photo and extract structured attributes for catalog indexing.
    Be specific about category and subcategory. Use 'unknown' only if truly unclear.
    """
    image: dspy.Image = dspy.InputField(desc="Product photo")
    listing_title: str = dspy.InputField(desc="Seller-provided title, may be incomplete or misleading")
    attributes: ProductAttributes = dspy.OutputField(desc="Extracted product attributes")

categorizer = dspy.Predict(CategorizeProduct)

# Process a single product
result = categorizer(
    image=dspy.Image.from_url("https://example.com/jacket.jpg"),
    listing_title="vintage jacket size M great condition"
)
print(result.attributes.category)       # Clothing
print(result.attributes.subcategory)    # Jackets & Coats
print(result.attributes.primary_color)  # Brown
print(result.attributes.condition)      # like_new

# Batch process a catalog
import json

products = [
    {"url": "https://example.com/item1.jpg", "title": "old lamp"},
    {"url": "https://example.com/item2.jpg", "title": "ceramic bowl set"},
    {"url": "https://example.com/item3.jpg", "title": "running shoes"},
]

results = []
for product in products:
    res = categorizer(
        image=dspy.Image.from_url(product["url"]),
        listing_title=product["title"]
    )
    results.append({
        "title": product["title"],
        "category": res.attributes.category,
        "subcategory": res.attributes.subcategory,
        "condition": res.attributes.condition,
    })

print(json.dumps(results, indent=2))
```

### Reward function for optimization

```python
VALID_CONDITIONS = {"new", "like_new", "used", "damaged"}
REQUIRED_FIELDS = ["category", "subcategory", "primary_color", "condition", "description"]

def product_categorization_reward(example, prediction, trace=None):
    attrs = prediction.attributes
    # Check all required fields are populated
    for field in REQUIRED_FIELDS:
        if not getattr(attrs, field, None):
            return 0.0
    # Check condition is valid
    if attrs.condition not in VALID_CONDITIONS:
        return 0.0
    # Check description is meaningful (not just "unknown")
    if len(attrs.description) < 20:
        return 0.5
    return 1.0

# Optimize with a labeled dataset
optimizer = dspy.MIPROv2(metric=product_categorization_reward)
optimized = optimizer.compile(categorizer, trainset=labeled_examples)
```

---

## Example 2 - Alt text generator

Generate accessible alt text for images on web pages and in content management systems.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class AltTextOutput(dspy.Signature):
    """
    Generate concise, accurate alt text for web accessibility (WCAG 2.1 AA).
    Alt text should describe the image content and function, not appearance.
    Decorative images should receive empty alt text.
    Keep alt text under 125 characters.
    """
    image: dspy.Image = dspy.InputField(desc="Image to describe")
    page_context: str = dspy.InputField(
        desc="The surrounding page content or article topic where the image appears"
    )
    image_role: str = dspy.InputField(
        desc="Role of the image - informative, decorative, functional, or complex"
    )
    alt_text: str = dspy.OutputField(
        desc="Alt text under 125 characters. Empty string if decorative."
    )
    long_description: str = dspy.OutputField(
        desc="Extended description for complex images like charts. Empty if not needed."
    )
    is_decorative: bool = dspy.OutputField(
        desc="True if the image is purely decorative and alt should be empty"
    )

alt_gen = dspy.Predict(AltTextOutput)

# Single image
result = alt_gen(
    image=dspy.Image.from_url("https://example.com/team.jpg"),
    page_context="About page of a B2B SaaS company, section titled 'Our Team'",
    image_role="informative"
)
print(result.alt_text)          # "Five team members smiling in a modern office"
print(result.is_decorative)     # False

# CMS batch processing
cms_images = [
    {
        "url": "https://example.com/hero-bg.jpg",
        "context": "Homepage hero section background",
        "role": "decorative"
    },
    {
        "url": "https://example.com/quarterly-chart.png",
        "context": "Q4 investor report, revenue growth section",
        "role": "complex"
    },
    {
        "url": "https://example.com/cta-button.png",
        "context": "Sign up call to action",
        "role": "functional"
    },
]

for item in cms_images:
    res = alt_gen(
        image=dspy.Image.from_url(item["url"]),
        page_context=item["context"],
        image_role=item["role"]
    )
    if res.is_decorative:
        print(f'{item["url"]} -> alt=""')
    else:
        print(f'{item["url"]} -> alt="{res.alt_text}"')
        if res.long_description:
            print(f'  longdesc: {res.long_description[:80]}...')
```

### Evaluation

```python
def alt_text_reward(example, prediction, trace=None):
    alt = prediction.alt_text or ""
    # Decorative images should have empty alt
    if example.is_decorative and alt == "":
        return 1.0
    if example.is_decorative and alt != "":
        return 0.0
    # Non-decorative images must have meaningful alt text
    if len(alt) == 0:
        return 0.0
    if len(alt) > 125:
        return 0.5  # Too long for WCAG compliance
    # Penalize generic filler phrases
    filler = ["image of", "photo of", "picture of"]
    if any(alt.lower().startswith(f) for f in filler):
        return 0.7
    return 1.0
```

---

## Example 3 - Receipt and invoice OCR pipeline

Extract structured line items and totals from photos of receipts and invoices.

```python
import dspy
from pydantic import BaseModel
from typing import List, Optional

lm = dspy.LM("openai/gpt-4o")  # Use a higher-quality model for OCR accuracy
dspy.configure(lm=lm)

class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total: float

class ReceiptData(dspy.Signature):
    """
    Extract all line items, taxes, and totals from a receipt or invoice photo.
    Normalize prices to float. If quantity is not shown, assume 1.
    Use 0.0 for any monetary field that is not present or legible.
    """
    image: dspy.Image = dspy.InputField(desc="Photo of receipt or invoice")
    currency_hint: str = dspy.InputField(
        desc="Expected currency code if known, e.g. USD, EUR. Use 'unknown' if not sure."
    )
    merchant_name: str = dspy.OutputField(desc="Name of the merchant or vendor")
    date: str = dspy.OutputField(desc="Transaction date in ISO 8601 format if readable, else empty string")
    line_items: List[LineItem] = dspy.OutputField(desc="All line items on the receipt")
    subtotal: float = dspy.OutputField(desc="Subtotal before tax and tip")
    tax: float = dspy.OutputField(desc="Tax amount")
    tip: float = dspy.OutputField(desc="Tip amount, 0.0 if not present")
    total: float = dspy.OutputField(desc="Final total paid")
    currency: str = dspy.OutputField(desc="Currency code detected")

extractor = dspy.Predict(ReceiptData)

# Process a single receipt
result = extractor(
    image=dspy.Image.from_file("receipt.jpg"),
    currency_hint="USD"
)

print(f"Merchant - {result.merchant_name}")
print(f"Date - {result.date}")
print(f"Items -")
for item in result.line_items:
    print(f"  {item.description} x{item.quantity} @ ${item.unit_price:.2f} = ${item.total:.2f}")
print(f"Subtotal - ${result.subtotal:.2f}")
print(f"Tax - ${result.tax:.2f}")
print(f"Tip - ${result.tip:.2f}")
print(f"Total - ${result.total:.2f}")

# Validate totals (simple sanity check)
computed = sum(i.total for i in result.line_items) + result.tax + result.tip
if abs(computed - result.total) > 0.05:
    print(f"WARNING - Computed total ${computed:.2f} does not match extracted total ${result.total:.2f}")
```

### Pipeline with preprocessing and retry

```python
from PIL import Image as PILImage
import io, base64

def preprocess_receipt(image_path: str) -> dspy.Image:
    """Resize and enhance contrast for better OCR accuracy."""
    img = PILImage.open(image_path).convert("RGB")
    # Resize so longest side is 2048px (receipts benefit from higher resolution)
    ratio = min(2048 / img.width, 2048 / img.height, 1.0)
    if ratio < 1.0:
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            PILImage.LANCZOS
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return dspy.Image.from_base64(b64, media_type="image/jpeg")

def receipt_reward(example, prediction, trace=None):
    """Reward function for receipt extraction quality."""
    if not prediction.line_items:
        return 0.0
    # Check total reconciliation
    computed = sum(i.total for i in prediction.line_items) + prediction.tax + prediction.tip
    total_ok = abs(computed - prediction.total) < 0.10
    # Check required fields
    has_merchant = bool(prediction.merchant_name)
    has_items = len(prediction.line_items) > 0
    score = (0.5 * int(total_ok)) + (0.3 * int(has_merchant)) + (0.2 * int(has_items))
    return score

# Use dspy.Refine for retry on low-confidence extractions
refining_extractor = dspy.Refine(dspy.Predict(ReceiptData), N=2, reward_fn=receipt_reward)

image = preprocess_receipt("crumpled_receipt.jpg")
result = refining_extractor(image=image, currency_hint="USD")
print(f"Total - ${result.total:.2f} ({len(result.line_items)} items)")
```
