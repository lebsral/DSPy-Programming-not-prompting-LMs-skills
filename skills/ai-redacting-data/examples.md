# ai-redacting-data - Examples

## Example 1 - Customer Support Email Redactor

Mask names, email addresses, and account numbers before routing tickets to a shared inbox or logging system.

```python
import dspy
import re
from dataclasses import dataclass
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

PATTERNS = {
    "EMAIL":   re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
    "PHONE":   re.compile(r'\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'),
    "ACCOUNT": re.compile(r'\baccount[:\s#]+\w+\b', re.IGNORECASE),
    "ORDER":   re.compile(r'\border[:\s#]+[A-Z0-9\-]+\b', re.IGNORECASE),
}

class DetectSupportPII(dspy.Signature):
    """Find names and personal identifiers in a customer support email.
    Return JSON list with pii_type (PERSON_NAME or ORG_NAME) and value."""

    text: str = dspy.InputField()
    pii_entities: list[dict] = dspy.OutputField()

class SupportEmailRedactor(dspy.Module):
    def __init__(self):
        self.detect = dspy.Predict(DetectSupportPII)
        self._seen: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def _placeholder(self, pii_type: str, value: str) -> str:
        if value in self._seen:
            return self._seen[value]
        n = self._counters.get(pii_type, 0) + 1
        self._counters[pii_type] = n
        label = f"[{pii_type}_{n}]"
        self._seen[value] = label
        return label

    def forward(self, email_text: str) -> dspy.Prediction:
        self._seen.clear()
        self._counters.clear()

        # Regex pass
        pre_masked = email_text
        regex_replacements = []
        for pii_type, pattern in PATTERNS.items():
            for m in pattern.finditer(email_text):
                label = self._placeholder(pii_type, m.group())
                regex_replacements.append((m.group(), label))

        for value, label in regex_replacements:
            pre_masked = pre_masked.replace(value, label)

        # LM pass on pre-masked text
        result = self.detect(text=pre_masked)
        for entity in (result.pii_entities or []):
            val = entity.get("value", "")
            pii_type = entity.get("pii_type", "PII")
            if val:
                self._placeholder(pii_type, val)

        # Final replacement on original
        final = email_text
        for value, label in sorted(self._seen.items(), key=lambda kv: len(kv[0]), reverse=True):
            final = final.replace(value, label)

        return dspy.Prediction(redacted=final, entity_map=dict(self._seen))


# Usage
redactor = SupportEmailRedactor()

email = """
Hi Support,

My name is Sarah Chen and I'm having trouble with my account #AC-88421.
I placed order #ORD-2024-99012 last Tuesday and it still hasn't shipped.
Please reach me at sarah.chen@gmail.com or 415-555-0192.

Thanks,
Sarah
"""

result = redactor(email_text=email.strip())
print(result.redacted)
# Hi Support,
#
# My name is [PERSON_NAME_1] and I'm having trouble with my account [ACCOUNT_1].
# I placed [ORDER_1] last Tuesday and it still hasn't shipped.
# Please reach me at [EMAIL_1] or [PHONE_1].
#
# Thanks,
# [PERSON_NAME_1]

print(result.entity_map)
# {"Sarah Chen": "[PERSON_NAME_1]", "AC-88421": "[ACCOUNT_1]", ...}
```

---

## Example 2 - Medical Record De-Identifier (HIPAA Safe Harbor)

Remove all 18 PHI categories required by HIPAA Safe Harbor before storing or sharing medical notes.

```python
import dspy
import re

lm = dspy.LM("openai/gpt-4o-mini")  # or use a local model to keep PHI off external APIs
dspy.configure(lm=lm)

# HIPAA Safe Harbor PHI patterns
HIPAA_PATTERNS = {
    "NAME":       None,  # handled by LM
    "DATE":       re.compile(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                             r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
                             r'Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
                             re.IGNORECASE),
    "PHONE":      re.compile(r'\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'),
    "FAX":        re.compile(r'\bfax[:\s]+[\d\-\(\)\s]+', re.IGNORECASE),
    "EMAIL":      re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
    "SSN":        re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "MRN":        re.compile(r'\bMRN[:\s#]+\w+\b', re.IGNORECASE),
    "ACCOUNT":    re.compile(r'\baccount[:\s#]+\w+\b', re.IGNORECASE),
    "ZIP":        re.compile(r'\b\d{5}(?:-\d{4})?\b'),
    "IP_ADDRESS": re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b'),
    "URL":        re.compile(r'https?://\S+'),
    "NPI":        re.compile(r'\bNPI[:\s#]+\d{10}\b', re.IGNORECASE),
    "DOB":        re.compile(r'\b(?:DOB|Date of Birth|born)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
                             re.IGNORECASE),
    "AGE_OVER_89": re.compile(r'\b(?:age[:\s]+)?(?:9\d|1[0-9]{2})\s*(?:years?\s*old|y/?o)\b',
                               re.IGNORECASE),
}

class DetectHIPAAPHI(dspy.Signature):
    """Identify HIPAA Protected Health Information requiring contextual understanding.
    Focus on - patient names, provider names, geographic subdivisions smaller than state,
    device identifiers, and certificate or license numbers.
    Do not flag medical conditions or treatment descriptions."""

    text: str = dspy.InputField(desc="Clinical note with structured PHI already masked")
    phi_entities: list[dict] = dspy.OutputField(
        desc='List of {"phi_type": "PATIENT_NAME", "value": "..."}'
    )

class HIPAADeidentifier(dspy.Module):
    def __init__(self):
        self.detect_phi = dspy.Predict(DetectHIPAAPHI)

    def forward(self, clinical_note: str) -> dspy.Prediction:
        seen: dict[str, str] = {}
        counters: dict[str, int] = {}

        def label(phi_type: str, value: str) -> str:
            if value in seen:
                return seen[value]
            n = counters.get(phi_type, 0) + 1
            counters[phi_type] = n
            tag = f"[{phi_type}]" if n == 1 else f"[{phi_type}_{n}]"
            seen[value] = tag
            return tag

        # Regex pass
        pre_masked = clinical_note
        for phi_type, pattern in HIPAA_PATTERNS.items():
            if pattern is None:
                continue
            for m in pattern.finditer(clinical_note):
                tag = label(phi_type, m.group())
                pre_masked = pre_masked.replace(m.group(), tag)

        # LM pass
        lm_result = self.detect_phi(text=pre_masked)
        for entity in (lm_result.phi_entities or []):
            val = entity.get("value", "")
            phi_type = entity.get("phi_type", "PHI")
            if val:
                label(phi_type, val)

        # Final pass on original text
        final = clinical_note
        for value, tag in sorted(seen.items(), key=lambda kv: len(kv[0]), reverse=True):
            final = final.replace(value, tag)

        return dspy.Prediction(deidentified=final, phi_removed=len(seen))


# Usage
deidentifier = HIPAADeidentifier()

note = """
Patient: Maria Gonzalez  DOB: 03/14/1962  MRN: 8821044
Provider: Dr. James Whitfield, NPI: 1234567890
Visit date: April 3, 2025

Chief complaint: Patient presents with chest pain since 02/28/2025.
Contact: (602) 555-7734  |  maria.gonzalez@yahoo.com
Address: 4521 W. Camelback Rd, Phoenix, AZ 85031
"""

result = deidentifier(clinical_note=note.strip())
print(result.deidentified)
print(f"PHI entities removed - {result.phi_removed}")
```

---

## Example 3 - Pre-LLM Sanitizer for Third-Party API Calls

Redact sensitive data before sending to an external AI API, then restore context in the response.

```python
import dspy
import re
import hashlib

lm = dspy.LM("openai/gpt-4o-mini")  # internal/trusted model for detection
dspy.configure(lm=lm)

PATTERNS = {
    "EMAIL":       re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
    "PHONE":       re.compile(r'\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'),
    "SSN":         re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "CREDIT_CARD": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "IP_ADDRESS":  re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b'),
}

class DetectNames(dspy.Signature):
    """Find person names in text. Return JSON list of name strings only."""
    text: str = dspy.InputField()
    names: list[str] = dspy.OutputField()

class PreLLMSanitizer(dspy.Module):
    """Redact PII, call an external LM, return response with placeholders intact."""

    def __init__(self):
        self.detect_names = dspy.Predict(DetectNames)
        # External LM (untrusted with PII)
        self.external_lm = dspy.LM("openai/gpt-4o")  # or any external provider
        self.external_task = dspy.ChainOfThought("sanitized_text -> summary")

    def _hash_token(self, value: str) -> str:
        return "[" + hashlib.sha256(value.encode()).hexdigest()[:8].upper() + "]"

    def forward(self, text: str, task: str = "summarize") -> dspy.Prediction:
        token_map: dict[str, str] = {}  # token → original value (for optional restore)
        seen: dict[str, str] = {}       # original value → token

        # Step 1 - regex redaction
        sanitized = text
        for pii_type, pattern in PATTERNS.items():
            for m in pattern.finditer(text):
                if m.group() not in seen:
                    token = self._hash_token(m.group())
                    seen[m.group()] = token
                    token_map[token] = m.group()

        # Step 2 - name detection on pre-sanitized text
        pre = text
        for val, tok in sorted(seen.items(), key=lambda kv: len(kv[0]), reverse=True):
            pre = pre.replace(val, tok)
        name_result = self.detect_names(text=pre)
        for name in (name_result.names or []):
            if name and name not in seen:
                token = self._hash_token(name)
                seen[name] = token
                token_map[token] = name

        # Step 3 - build fully sanitized text
        for val, tok in sorted(seen.items(), key=lambda kv: len(kv[0]), reverse=True):
            sanitized = sanitized.replace(val, tok)

        # Step 4 - call external LM with sanitized text
        with dspy.context(lm=self.external_lm):
            ext_result = self.external_task(sanitized_text=sanitized)

        return dspy.Prediction(
            sanitized_input=sanitized,
            external_response=ext_result.summary,
            token_map=token_map,  # keep for optional de-anonymization
        )


# Usage
sanitizer = PreLLMSanitizer()

doc = """
John Martinez submitted a support ticket about a billing issue.
His account was charged $299 twice on 2025-04-01.
Email: john.martinez@company.com | Phone: 312-555-8820
CC ending in 4532 was charged.
"""

result = sanitizer(text=doc.strip())
print("Sanitized input sent to external LM:")
print(result.sanitized_input)
print("\nExternal LM response (contains only tokens, no PII):")
print(result.external_response)
# All PII replaced with hash tokens - safe to log, cache, or display
```
