# ai-translating-content - Examples

## Example 1 - Marketing copy translator (English to Spanish with brand glossary)

Translate landing page and product copy from English to Spanish (Mexico) while preserving brand terms like product names and feature labels.

### Setup

```python
import dspy
from pydantic import BaseModel
from typing import list

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

### Signatures and module

```python
class MarketingTranslationResult(BaseModel):
    translated_text: str
    glossary_terms_used: list[str]

class TranslateMarketing(dspy.Signature):
    """Translate English marketing copy to the target_language with the given tone.
    Terms in glossary must appear verbatim in the output — do not translate them."""
    source_text: str = dspy.InputField(desc="English marketing copy")
    target_language: str = dspy.InputField()
    tone: str = dspy.InputField(desc="casual, neutral, or formal")
    glossary: list[str] = dspy.InputField(
        desc="Brand terms, product names, and feature labels that must NOT be translated"
    )
    result: MarketingTranslationResult = dspy.OutputField()

translator = dspy.Predict(TranslateMarketing)
```

### Usage

```python
copy_strings = [
    "Get started with Acme Pro for free — no credit card required.",
    "Organize your work in Workspaces. Share with your team in seconds.",
    "The Dashboard gives you a real-time view of every project.",
]

glossary = ["Acme Pro", "Workspaces", "Dashboard"]

for text in copy_strings:
    result = translator(
        source_text=text,
        target_language="Spanish (Mexico)",
        tone="casual",
        glossary=glossary
    )
    print(f"EN: {text}")
    print(f"ES: {result.result.translated_text}")
    print(f"Terms preserved: {result.result.glossary_terms_used}")
    print()
```

Expected output:

```
EN: Get started with Acme Pro for free — no credit card required.
ES: Comienza con Acme Pro gratis, sin necesidad de tarjeta de crédito.
Terms preserved: ["Acme Pro"]

EN: Organize your work in Workspaces. Share with your team in seconds.
ES: Organiza tu trabajo en Workspaces. Comparte con tu equipo en segundos.
Terms preserved: ["Workspaces"]

EN: The Dashboard gives you a real-time view of every project.
ES: El Dashboard te da una vista en tiempo real de cada proyecto.
Terms preserved: ["Dashboard"]
```

### Metric

```python
def glossary_compliance_metric(example, pred, trace=None):
    glossary = example.glossary
    translated = pred.result.translated_text
    violations = [term for term in glossary if term not in translated]
    if violations:
        print(f"Glossary violations: {violations}")
    return len(violations) == 0
```

---

## Example 2 - i18n JSON batch translator

Translate a full `en.json` locale file to multiple target locales, preserving interpolation placeholders like `{count}` and `{name}`.

### Setup

```python
import dspy
import json
from pathlib import Path

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

### Signatures and module

```python
class TranslateI18nString(dspy.Signature):
    """Translate a single i18n string into target_language.
    Preserve interpolation placeholders exactly as written (e.g., {count}, {name}, %s).
    Glossary terms must appear verbatim."""
    source_text: str = dspy.InputField(desc="Single i18n string, may contain {placeholders}")
    target_language: str = dspy.InputField()
    glossary: list[str] = dspy.InputField(default=[])
    translated_text: str = dspy.OutputField()

translator = dspy.Predict(TranslateI18nString)

def translate_locale_file(
    source_path: str,
    output_path: str,
    target_language: str,
    glossary: list[str] = None
):
    glossary = glossary or []
    with open(source_path) as f:
        strings = json.load(f)

    translated = {}
    for key, text in strings.items():
        result = translator(
            source_text=text,
            target_language=target_language,
            glossary=glossary
        )
        translated[key] = result.translated_text

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    print(f"Translated {len(translated)} strings to {output_path}")
    return translated
```

### Usage

Given `locales/en.json`:

```json
{
  "welcome": "Welcome back, {name}!",
  "items_count": "You have {count} items in your cart.",
  "sign_out": "Sign out",
  "upgrade_cta": "Upgrade to Acme Pro",
  "empty_state": "No results found. Try a different search."
}
```

```python
glossary = ["Acme Pro"]

# Translate to German
translate_locale_file(
    source_path="locales/en.json",
    output_path="locales/de.json",
    target_language="German (formal)",
    glossary=glossary
)

# Translate to Japanese
translate_locale_file(
    source_path="locales/en.json",
    output_path="locales/ja.json",
    target_language="Japanese",
    glossary=glossary
)
```

Expected `locales/de.json`:

```json
{
  "welcome": "Willkommen zurück, {name}!",
  "items_count": "Sie haben {count} Artikel in Ihrem Warenkorb.",
  "sign_out": "Abmelden",
  "upgrade_cta": "Upgrade auf Acme Pro",
  "empty_state": "Keine Ergebnisse gefunden. Versuchen Sie eine andere Suche."
}
```

### Metric

```python
import re

def placeholder_preservation_metric(example, pred, trace=None):
    source = example.source_text
    translated = pred.translated_text
    # Extract all {placeholder} patterns from source
    placeholders = re.findall(r'\{[^}]+\}', source)
    missing = [p for p in placeholders if p not in translated]
    if missing:
        print(f"Missing placeholders: {missing}")
    return len(missing) == 0
```

---

## Example 3 - Support ticket translator with confidence scoring

Translate inbound support tickets from any language to English for your support team, flagging idiom-heavy or ambiguous messages for human review.

### Setup

```python
import dspy
from pydantic import BaseModel

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

### Signatures and module

```python
class SupportTranslationResult(BaseModel):
    translated_text: str
    detected_source_language: str
    confidence: float  # 0.0–1.0
    needs_review: bool
    review_reason: str  # empty string if needs_review is False

class TranslateSupportTicket(dspy.Signature):
    """Translate a support ticket into English. Auto-detect the source language.
    Estimate confidence: 1.0 = clear and literal, <0.7 = idioms, sarcasm, or ambiguous phrasing.
    Flag needs_review=True if confidence < 0.75 or if intent is ambiguous."""
    ticket_text: str = dspy.InputField(desc="Raw support ticket text in any language")
    result: SupportTranslationResult = dspy.OutputField()

translator = dspy.ChainOfThought(TranslateSupportTicket)

def process_ticket(ticket_text: str) -> dict:
    result = translator(ticket_text=ticket_text)
    r = result.result
    return {
        "original": ticket_text,
        "translation": r.translated_text,
        "source_language": r.detected_source_language,
        "confidence": r.confidence,
        "needs_review": r.needs_review,
        "review_reason": r.review_reason
    }
```

### Usage

```python
tickets = [
    "Bonjour, mon abonnement a été débité deux fois ce mois-ci. Pouvez-vous corriger cela?",
    "Das ist ja wohl ein Witz! Die App funktioniert überhaupt nicht mehr.",  # sarcasm
    "我无法登录我的账户，密码重置邮件也没有收到。",
    "Ayer todo funcionaba y hoy nada. No sé qué pasó.",
]

for ticket in tickets:
    result = process_ticket(ticket)
    print(f"[{result['source_language']}] Confidence: {result['confidence']:.2f}")
    print(f"Translation: {result['translation']}")
    if result['needs_review']:
        print(f"REVIEW NEEDED: {result['review_reason']}")
    print()
```

Expected output:

```
[French] Confidence: 0.97
Translation: Hello, my subscription was charged twice this month. Can you correct this?

[German] Confidence: 0.65
Translation: Are you serious? The app is completely broken.
REVIEW NEEDED: Sarcastic tone detected — "Das ist ja wohl ein Witz" is an idiom expressing frustration, not a literal question.

[Chinese (Simplified)] Confidence: 0.95
Translation: I cannot log in to my account and I have not received the password reset email.

[Spanish] Confidence: 0.82
Translation: Yesterday everything was working and today nothing is. I do not know what happened.
```

### Metric

```python
def translation_quality_metric(example, pred, trace=None):
    r = pred.result
    # Check that high-confidence translations do not get flagged as needing review
    if r.confidence >= 0.85 and r.needs_review:
        return False
    # Check that very low-confidence translations are always flagged
    if r.confidence < 0.6 and not r.needs_review:
        return False
    # Basic sanity - translated text must be non-empty
    return bool(r.translated_text.strip())
```
