---
name: ai-writing-content
description: Generate articles, reports, blog posts, or marketing copy with AI. Use when writing blog posts, creating product descriptions, generating newsletters, drafting reports, producing marketing copy, creating documentation, writing email campaigns, or any task where AI writes long-form content from a topic or brief. Powered by DSPy content generation pipelines.
---

# Build an AI Content Writer

Guide the user through building AI that writes articles, reports, and marketing copy. Uses DSPy to create a structured pipeline: outline, draft section-by-section, enrich with research, and polish with feedback loops.

## Step 1: Understand the content task

Ask the user:
1. **What type of content?** (blog post, product description, report, newsletter, docs?)
2. **What tone and voice?** (professional, casual, technical, marketing, brand-specific?)
3. **How long?** (tweet, paragraph, 500-word post, 2000-word article?)
4. **Does it need research?** (factual claims grounded in sources, or creative/opinion?)
5. **Any brand guidelines?** (words to avoid, style rules, required sections?)

## Step 2: Build an outline generator

Start with structure. An outline gives the writer a plan to follow:

```python
import dspy
from pydantic import BaseModel, Field

class Section(BaseModel):
    heading: str = Field(description="Section heading")
    key_points: list[str] = Field(description="Main points to cover in this section")

class ContentOutline(BaseModel):
    title: str
    sections: list[Section]

class GenerateOutline(dspy.Signature):
    """Create a structured outline for the content."""
    topic: str = dspy.InputField(desc="The topic or brief to write about")
    content_type: str = dspy.InputField(desc="Type: blog post, report, product description, etc.")
    audience: str = dspy.InputField(desc="Who will read this content")
    outline: ContentOutline = dspy.OutputField()

outliner = dspy.ChainOfThought(GenerateOutline)
```

### With research context

If the content needs to be grounded in facts:

```python
class GenerateResearchedOutline(dspy.Signature):
    """Create a structured outline grounded in the provided research."""
    topic: str = dspy.InputField()
    content_type: str = dspy.InputField()
    audience: str = dspy.InputField()
    research: list[str] = dspy.InputField(desc="Research sources and key facts")
    outline: ContentOutline = dspy.OutputField()
```

## Step 3: Generate section by section

Don't generate the whole article at once. Write one section at a time for better quality:

```python
class WriteSection(dspy.Signature):
    """Write one section of the article based on the outline."""
    topic: str = dspy.InputField(desc="Overall article topic")
    section_heading: str = dspy.InputField(desc="This section's heading")
    key_points: list[str] = dspy.InputField(desc="Points to cover in this section")
    previous_sections: str = dspy.InputField(desc="What's been written so far, for continuity")
    tone: str = dspy.InputField(desc="Writing tone and style")
    section_text: str = dspy.OutputField(desc="The written section (2-4 paragraphs)")

class ContentWriter(dspy.Module):
    def __init__(self):
        self.outline = dspy.ChainOfThought(GenerateOutline)
        self.write_section = dspy.ChainOfThought(WriteSection)

    def forward(self, topic, content_type="blog post", audience="general", tone="professional"):
        # Step 1: Generate outline
        plan = self.outline(topic=topic, content_type=content_type, audience=audience)

        # Step 2: Write each section
        sections = []
        running_text = ""

        for section in plan.outline.sections:
            result = self.write_section(
                topic=topic,
                section_heading=section.heading,
                key_points=section.key_points,
                previous_sections=running_text[-2000:],  # last 2000 chars for context
                tone=tone,
            )
            sections.append(f"## {section.heading}\n\n{result.section_text}")
            running_text += result.section_text + "\n\n"

        full_article = f"# {plan.outline.title}\n\n" + "\n\n".join(sections)

        return dspy.Prediction(
            title=plan.outline.title,
            outline=plan.outline,
            article=full_article,
        )
```

## Step 4: Add research grounding

For content that needs factual claims backed by sources:

### Retrieval-augmented content

```python
class ResearchTopic(dspy.Signature):
    """Generate search queries to research this topic."""
    topic: str = dspy.InputField()
    key_points: list[str] = dspy.InputField(desc="Points that need factual backing")
    queries: list[str] = dspy.OutputField(desc="Search queries to find supporting facts")

class WriteSectionWithSources(dspy.Signature):
    """Write a section using the provided sources for factual claims."""
    section_heading: str = dspy.InputField()
    key_points: list[str] = dspy.InputField()
    sources: list[str] = dspy.InputField(desc="Research passages to ground claims in")
    previous_sections: str = dspy.InputField()
    tone: str = dspy.InputField()
    section_text: str = dspy.OutputField(desc="Section text with claims grounded in sources")

class ResearchedWriter(dspy.Module):
    def __init__(self):
        self.outline = dspy.ChainOfThought(GenerateOutline)
        self.research = dspy.ChainOfThought(ResearchTopic)
        self.retrieve = dspy.Retrieve(k=3)
        self.write = dspy.ChainOfThought(WriteSectionWithSources)

    def forward(self, topic, content_type="blog post", audience="general", tone="professional"):
        plan = self.outline(topic=topic, content_type=content_type, audience=audience)

        sections = []
        running_text = ""

        for section in plan.outline.sections:
            # Research this section
            queries = self.research(
                topic=topic, key_points=section.key_points
            ).queries

            sources = []
            for query in queries:
                sources.extend(self.retrieve(query).passages)

            # Write with sources
            result = self.write(
                section_heading=section.heading,
                key_points=section.key_points,
                sources=sources,
                previous_sections=running_text[-2000:],
                tone=tone,
            )
            sections.append(f"## {section.heading}\n\n{result.section_text}")
            running_text += result.section_text + "\n\n"

        return dspy.Prediction(
            title=plan.outline.title,
            article=f"# {plan.outline.title}\n\n" + "\n\n".join(sections),
        )
```

## Step 5: Quality loop — generate, critique, improve

Add a feedback loop to iteratively improve drafts:

```python
class CritiqueContent(dspy.Signature):
    """Critique the written content and suggest improvements."""
    content: str = dspy.InputField(desc="The content to critique")
    content_type: str = dspy.InputField()
    audience: str = dspy.InputField()
    is_good_enough: bool = dspy.OutputField(desc="Is this ready to publish?")
    feedback: str = dspy.OutputField(desc="Specific feedback for improvement")

class ImproveContent(dspy.Signature):
    """Improve the content based on the feedback."""
    content: str = dspy.InputField(desc="Current draft")
    feedback: str = dspy.InputField(desc="Feedback to address")
    improved_content: str = dspy.OutputField(desc="Improved version")

class QualityWriter(dspy.Module):
    def __init__(self, max_revisions=2):
        self.writer = ContentWriter()
        self.critic = dspy.ChainOfThought(CritiqueContent)
        self.improver = dspy.ChainOfThought(ImproveContent)
        self.max_revisions = max_revisions

    def forward(self, topic, content_type="blog post", audience="general", tone="professional"):
        # Generate first draft
        draft = self.writer(
            topic=topic, content_type=content_type, audience=audience, tone=tone
        )
        article = draft.article

        # Critique-improve loop
        for _ in range(self.max_revisions):
            critique = self.critic(
                content=article, content_type=content_type, audience=audience
            )
            if critique.is_good_enough:
                break

            improved = self.improver(content=article, feedback=critique.feedback)
            article = improved.improved_content

        return dspy.Prediction(
            title=draft.title,
            article=article,
        )
```

## Step 6: Voice and style enforcement

Enforce brand voice and style rules:

```python
class StyledWriter(dspy.Module):
    def __init__(self, brand_rules=None):
        self.writer = ContentWriter()
        self.brand_rules = brand_rules or []

    def forward(self, topic, content_type="blog post", audience="general", tone="professional"):
        result = self.writer(topic=topic, content_type=content_type, audience=audience, tone=tone)

        # Enforce brand rules
        article_lower = result.article.lower()

        for rule in self.brand_rules:
            if rule.get("type") == "forbidden_word":
                dspy.Assert(
                    rule["word"].lower() not in article_lower,
                    f"Content must not use the word '{rule['word']}'. "
                    f"Use '{rule.get('alternative', 'a different word')}' instead."
                )
            elif rule.get("type") == "required_section":
                dspy.Assert(
                    rule["heading"].lower() in article_lower,
                    f"Content must include a '{rule['heading']}' section."
                )

        # General style checks
        sentences = result.article.split(".")
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        dspy.Suggest(
            avg_sentence_len < 25,
            "Sentences are too long on average. Aim for shorter, punchier sentences."
        )

        return result

# Usage
writer = StyledWriter(brand_rules=[
    {"type": "forbidden_word", "word": "utilize", "alternative": "use"},
    {"type": "forbidden_word", "word": "leverage", "alternative": "use"},
    {"type": "required_section", "heading": "Conclusion"},
])
```

## Step 7: Test and optimize

### Readability metric

```python
def readability_metric(example, prediction, trace=None):
    words = prediction.article.split()
    sentences = prediction.article.split(".")
    if not sentences or not words:
        return 0.0

    avg_sentence_len = len(words) / len(sentences)
    # Penalize very long or very short sentences
    readability = 1.0 if 10 < avg_sentence_len < 20 else 0.5
    # Penalize very short articles
    length_ok = 1.0 if len(words) > 200 else 0.5
    return (readability + length_ok) / 2
```

### AI-as-judge metric

```python
class JudgeContent(dspy.Signature):
    """Judge the quality of generated content."""
    content: str = dspy.InputField()
    content_type: str = dspy.InputField()
    topic: str = dspy.InputField()
    relevance: float = dspy.OutputField(desc="0.0-1.0 — stays on topic")
    coherence: float = dspy.OutputField(desc="0.0-1.0 — flows well, logically structured")
    engagement: float = dspy.OutputField(desc="0.0-1.0 — interesting to read")

def content_quality_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeContent)
    result = judge(
        content=prediction.article,
        content_type=example.content_type,
        topic=example.topic,
    )
    return (result.relevance + result.coherence + result.engagement) / 3
```

### Optimize

```python
optimizer = dspy.BootstrapFewShot(metric=content_quality_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(QualityWriter(), trainset=trainset)
```

## Key patterns

- **Outline first, then write** — structure prevents rambling and missed points
- **Section-by-section generation** — writing one section at a time produces better quality than generating the whole article at once
- **Retrieve for factual grounding** — pull in sources to back up claims
- **Critique-improve loop** — generate, critique, improve catches issues a single pass misses
- **Assert for brand rules** — DSPy retries when forbidden words or missing sections are detected
- **AI-as-judge for quality** — use a judge signature to score relevance, coherence, engagement

## Additional resources

- For worked examples (blog posts, product descriptions, newsletters), see [examples.md](examples.md)
- Need to summarize content instead of generating it? Use `/ai-summarizing`
- Need multi-step pipelines beyond content? Use `/ai-building-pipelines`
- Next: `/ai-improving-accuracy` to measure and improve your content writer
