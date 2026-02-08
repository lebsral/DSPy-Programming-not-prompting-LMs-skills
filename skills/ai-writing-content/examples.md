# AI Writing Content — Worked Examples

## Example 1: Blog post generator

Generate SEO-friendly blog posts from a topic and target audience.

### Setup

```python
import dspy
from pydantic import BaseModel, Field

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

### Signatures and module

```python
class BlogSection(BaseModel):
    heading: str
    key_points: list[str]

class BlogOutline(BaseModel):
    title: str = Field(description="SEO-friendly blog post title")
    hook: str = Field(description="Opening hook — one compelling sentence")
    sections: list[BlogSection]

class PlanBlogPost(dspy.Signature):
    """Create an outline for a blog post."""
    topic: str = dspy.InputField()
    audience: str = dspy.InputField()
    outline: BlogOutline = dspy.OutputField()

class WriteBlogSection(dspy.Signature):
    """Write one section of a blog post. Engaging, clear, and actionable."""
    topic: str = dspy.InputField()
    heading: str = dspy.InputField()
    key_points: list[str] = dspy.InputField()
    prior_text: str = dspy.InputField(desc="Previously written sections for continuity")
    section_text: str = dspy.OutputField(desc="2-4 paragraphs for this section")

class BlogWriter(dspy.Module):
    def __init__(self):
        self.plan = dspy.ChainOfThought(PlanBlogPost)
        self.write = dspy.ChainOfThought(WriteBlogSection)

    def forward(self, topic, audience="developers"):
        outline = self.plan(topic=topic, audience=audience).outline

        parts = [f"# {outline.title}\n\n{outline.hook}\n"]
        running = outline.hook

        for section in outline.sections:
            result = self.write(
                topic=topic,
                heading=section.heading,
                key_points=section.key_points,
                prior_text=running[-1500:],
            )
            parts.append(f"## {section.heading}\n\n{result.section_text}")
            running += "\n" + result.section_text

        return dspy.Prediction(
            title=outline.title,
            article="\n\n".join(parts),
        )
```

### Usage

```python
writer = BlogWriter()
result = writer(topic="How to add AI features to your SaaS app", audience="SaaS founders")
print(result.title)
print(result.article[:500])
```

### Metric

```python
class JudgeBlogPost(dspy.Signature):
    """Judge a blog post's quality."""
    article: str = dspy.InputField()
    topic: str = dspy.InputField()
    has_clear_structure: bool = dspy.OutputField(desc="Has intro, body sections, conclusion")
    stays_on_topic: bool = dspy.OutputField(desc="Content is relevant to the topic")
    actionable: bool = dspy.OutputField(desc="Reader knows what to do next")

def blog_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeBlogPost)
    result = judge(article=prediction.article, topic=example.topic)
    score = (result.has_clear_structure + result.stays_on_topic + result.actionable) / 3
    # Bonus for reasonable length
    word_count = len(prediction.article.split())
    if 500 < word_count < 2000:
        score += 0.1
    return min(score, 1.0)
```

---

## Example 2: Product description writer

Generate consistent product descriptions for an e-commerce catalog.

### Signatures and module

```python
class ProductDescription(BaseModel):
    headline: str = Field(description="Short, catchy headline (under 10 words)")
    description: str = Field(description="2-3 sentence product description")
    key_features: list[str] = Field(description="3-5 bullet point features")
    call_to_action: str = Field(description="One-line CTA")

class WriteProductDescription(dspy.Signature):
    """Write a compelling product description for an e-commerce store."""
    product_name: str = dspy.InputField()
    product_details: str = dspy.InputField(desc="Raw product specs, features, materials")
    brand_voice: str = dspy.InputField(desc="e.g. 'friendly and casual' or 'premium and minimal'")
    description: ProductDescription = dspy.OutputField()

class ProductWriter(dspy.Module):
    def __init__(self):
        self.write = dspy.ChainOfThought(WriteProductDescription)

    def forward(self, product_name, product_details, brand_voice="friendly and helpful"):
        result = self.write(
            product_name=product_name,
            product_details=product_details,
            brand_voice=brand_voice,
        )

        # Enforce constraints
        dspy.Assert(
            len(result.description.headline.split()) <= 10,
            f"Headline is {len(result.description.headline.split())} words, must be under 10"
        )
        dspy.Suggest(
            len(result.description.key_features) >= 3,
            "Include at least 3 key features"
        )

        return result
```

### Usage

```python
writer = ProductWriter()
result = writer(
    product_name="CloudSync Pro Backpack",
    product_details="Water-resistant 600D polyester, 15.6 inch laptop compartment, USB charging port, 30L capacity, YKK zippers, padded shoulder straps, weight: 1.2kg",
    brand_voice="minimal and premium",
)
print(result.description.headline)
# "Your Office. Everywhere."
print(result.description.key_features)
# ["Water-resistant 600D polyester shell", "Fits laptops up to 15.6 inches", ...]
```

### Batch processing

```python
products = [
    {"product_name": "CloudSync Pro Backpack", "product_details": "..."},
    {"product_name": "AirDesk Standing Mat", "product_details": "..."},
    # ...
]

writer = ProductWriter()
for product in products:
    result = writer(**product, brand_voice="minimal and premium")
    save_to_catalog(product["product_name"], result.description)
```

---

## Example 3: Email / newsletter composer

Generate personalized email content from a brief.

### Signatures and module

```python
class EmailContent(BaseModel):
    subject_line: str = Field(description="Email subject line (under 60 characters)")
    preview_text: str = Field(description="Preview text shown in inbox (under 90 characters)")
    body: str = Field(description="Email body in plain text")

class ComposeEmail(dspy.Signature):
    """Compose an email or newsletter from the brief."""
    brief: str = dspy.InputField(desc="What the email should communicate")
    audience: str = dspy.InputField(desc="Who receives this email")
    tone: str = dspy.InputField(desc="e.g. 'professional', 'friendly', 'urgent'")
    email: EmailContent = dspy.OutputField()

class EmailComposer(dspy.Module):
    def __init__(self):
        self.compose = dspy.ChainOfThought(ComposeEmail)

    def forward(self, brief, audience="customers", tone="friendly"):
        result = self.compose(brief=brief, audience=audience, tone=tone)

        # Subject line constraints
        dspy.Assert(
            len(result.email.subject_line) <= 60,
            f"Subject line is {len(result.email.subject_line)} chars, must be under 60"
        )
        dspy.Assert(
            len(result.email.preview_text) <= 90,
            f"Preview text is {len(result.email.preview_text)} chars, must be under 90"
        )

        # No spammy patterns
        spam_words = ["free", "act now", "limited time", "click here"]
        subject_lower = result.email.subject_line.lower()
        dspy.Suggest(
            not any(word in subject_lower for word in spam_words),
            "Avoid spammy words in the subject line"
        )

        return result
```

### Usage

```python
composer = EmailComposer()
result = composer(
    brief="Announce our new API v2 with breaking changes. Migration guide available. Deadline is March 1.",
    audience="developers using our API",
    tone="professional but friendly",
)
print(result.email.subject_line)
# "API v2 is here — migrate by March 1"
print(result.email.body[:200])
```

### Metric and optimization

```python
class JudgeEmail(dspy.Signature):
    """Judge email quality."""
    email_body: str = dspy.InputField()
    brief: str = dspy.InputField()
    covers_brief: bool = dspy.OutputField(desc="All key points from the brief are mentioned")
    clear_cta: bool = dspy.OutputField(desc="There's a clear call to action")
    appropriate_tone: bool = dspy.OutputField(desc="Tone matches the target audience")

def email_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeEmail)
    result = judge(
        email_body=prediction.email.body,
        brief=example.brief,
    )
    return (result.covers_brief + result.clear_cta + result.appropriate_tone) / 3

optimizer = dspy.BootstrapFewShot(metric=email_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(EmailComposer(), trainset=trainset)
```
