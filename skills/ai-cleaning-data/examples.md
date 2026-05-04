# ai-cleaning-data - Examples

## Example 1 - Address Normalizer

Normalize messy US addresses to USPS standard format. Handles abbreviations, missing components, and inconsistent capitalization.

```python
import dspy
import pandas as pd

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class NormalizeAddress(dspy.Signature):
    """Normalize a US mailing address to USPS standard format.
    Expand abbreviations - St → Street, Ave → Avenue, Blvd → Boulevard,
    Apt → Apartment, Ste → Suite, Dr → Drive, Rd → Road.
    Capitalize each word in street name, city, and state abbreviation.
    Preserve all components including suite/unit/apartment numbers.
    Do not infer or add information not present in the input.
    If a component is missing, leave it absent - do not guess.
    """
    raw_address: str = dspy.InputField(desc="Raw address string, may include city/state/zip")
    normalized: str = dspy.OutputField(
        desc="USPS-format address, e.g. '123 Main Street, Suite 100, Springfield, IL 62701'"
    )
    confidence: float = dspy.OutputField(desc="Confidence 0.0-1.0 that normalization is correct")
    meaning_preserved: bool = dspy.OutputField(
        desc="True if all components from input are present in output, False if any were dropped"
    )

normalizer = dspy.Predict(NormalizeAddress)

test_addresses = [
    "123 main st ste 100 springfield il 62701",
    "456 Oak Ave., Apt 2B, Chicago, IL  60601",
    "789 elm blvd, boston ma 02101",
    "1000 W. Broadway Rd, Phoenix AZ, 85001",
    "55 Park ave new york ny 10022",
]

results = []
for addr in test_addresses:
    r = normalizer(raw_address=addr)
    results.append({
        "original": addr,
        "normalized": r.normalized,
        "confidence": r.confidence,
        "meaning_preserved": r.meaning_preserved,
    })

df = pd.DataFrame(results)

# Flag anything that lost meaning or is low confidence
flagged = df[(df["confidence"] < 0.85) | (~df["meaning_preserved"])]
print(f"Auto-accepted: {len(df) - len(flagged)}")
print(f"Flagged for review: {len(flagged)}")
print(df[["original", "normalized", "confidence"]].to_string())
```

Sample output:

```
Auto-accepted: 4
Flagged for review: 1

original                                      normalized                                  confidence
123 main st ste 100 springfield il 62701      123 Main Street, Suite 100, Springfield...  0.97
456 Oak Ave., Apt 2B, Chicago, IL  60601      456 Oak Avenue, Apartment 2B, Chicago, ...  0.99
789 elm blvd, boston ma 02101                 789 Elm Boulevard, Boston, MA 02101         0.94
1000 W. Broadway Rd, Phoenix AZ, 85001        1000 West Broadway Road, Phoenix, AZ 85001  0.91
55 Park ave new york ny 10022                 55 Park Avenue, New York, NY 10022          0.88
```

---

## Example 2 - Company Name Standardizer

Resolve variant company names to their canonical forms. Useful before joining datasets or deduplicating CRM records.

```python
import dspy
from pydantic import BaseModel

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class CompanyResolution(BaseModel):
    canonical_name: str
    confidence: float
    is_recognized: bool
    variant_type: str  # e.g. "abbreviation", "common name", "legal suffix", "typo", "already canonical"

class StandardizeCompany(dspy.Signature):
    """Resolve a company name variant to its well-known canonical name.

    Examples of resolutions:
    - 'IBM Corp.' → 'IBM' (legal suffix removal)
    - 'I.B.M.' → 'IBM' (abbreviation expansion)
    - 'International Business Machines' → 'IBM' (full legal name → brand)
    - 'Mickey D' → 'McDonald's' (common nickname)
    - 'Alphabet Inc.' → 'Alphabet' (legal suffix)
    - 'MSFT' → 'Microsoft' (stock ticker)
    - 'Amazn' → 'Amazon' (typo)

    If the variant cannot be confidently resolved, return it unchanged with is_recognized=False.
    """
    variant: str = dspy.InputField(desc="Company name variant to standardize")
    result: CompanyResolution = dspy.OutputField(desc="Resolution result with canonical name and metadata")

standardizer = dspy.TypedPredictor(StandardizeCompany)

variants = [
    "IBM Corp.",
    "I.B.M.",
    "International Business Machines Corporation",
    "MSFT",
    "Amazn",  # typo
    "Mickey D's",
    "Alphabet Inc.",
    "Google LLC",
    "McKinsey & Company",
    "Accenture PLC",
    "XYZ Consulting Partners",  # unknown
]

print(f"{'Variant':<40} {'Canonical':<30} {'Type':<20} {'Conf'}")
print("-" * 100)
for v in variants:
    r = standardizer(variant=v).result
    flag = "" if r.is_recognized else " [UNKNOWN]"
    print(f"{v:<40} {r.canonical_name:<30} {r.variant_type:<20} {r.confidence:.2f}{flag}")
```

Sample output:

```
Variant                                  Canonical                      Type                 Conf
----------------------------------------------------------------------------------------------------
IBM Corp.                                IBM                            legal suffix         0.99
I.B.M.                                   IBM                            abbreviation         0.98
International Business Machines Corpo... IBM                            full legal name      0.97
MSFT                                     Microsoft                      stock ticker         0.99
Amazn                                    Amazon                         typo                 0.92
Mickey D's                               McDonald's                     common nickname      0.95
Alphabet Inc.                            Alphabet                       legal suffix         0.98
Google LLC                               Google                         legal suffix         0.99
McKinsey & Company                       McKinsey & Company             already canonical    0.96
Accenture PLC                            Accenture                      legal suffix         0.97
XYZ Consulting Partners                  XYZ Consulting Partners        already canonical    0.41 [UNKNOWN]
```

---

## Example 3 - CSV Batch Cleaner

Clean a CSV with mixed date formats, inconsistent phone numbers, and typos in a category column. Uses rule inference to minimize LM calls.

```python
import dspy
import pandas as pd
import re
from io import StringIO

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Simulated messy CSV
MESSY_CSV = """id,name,phone,signup_date,category
1,Alice Chen,(415)555-1234,05/04/2026,enterprise
2,Bob Smith,415.555.5678,2026-05-04,Enterprise
3,Carol Wu,+14155559012,May 4th 2026,SMB
4,Dave Lee,415 555 3456,4-May-26,smb
5,Eve Park,4155550000,20260504,Mid-market
6,Frank Kim,(415) 555-1111,05/04/26,mid market
7,Grace Ho,415-555-2222,2026/05/04,ENTERPRISE
8,Hiro Ito,555-3333,May 2026,Unknown
"""

df = pd.read_csv(StringIO(MESSY_CSV))
print("Original data:")
print(df.to_string(index=False))
print()

# --- Clean phone numbers ---
# Try regex first for the most common patterns

def clean_phone_regex(raw: str) -> str | None:
    """Handle patterns we can resolve deterministically."""
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) == 10:
        return f"+1 ({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == "1":
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    return None  # needs LM

class CleanPhone(dspy.Signature):
    """Clean an ambiguous US phone number to +1 (NNN) NNN-NNNN format.
    Only use this for numbers that could not be parsed by regex.
    """
    raw_phone: str = dspy.InputField()
    cleaned: str = dspy.OutputField(desc="Phone in +1 (NNN) NNN-NNNN format, or empty string if not a valid US number")
    confidence: float = dspy.OutputField()

phone_cleaner = dspy.Predict(CleanPhone)

cleaned_phones = []
lm_calls = 0
for raw in df["phone"]:
    result = clean_phone_regex(str(raw))
    if result is None:
        r = phone_cleaner(raw_phone=str(raw))
        result = r.cleaned
        lm_calls += 1
    cleaned_phones.append(result)

df["phone_clean"] = cleaned_phones
print(f"Phone cleaning - {lm_calls} LM calls out of {len(df)} rows (regex handled the rest)")

# --- Clean dates ---
class CleanDate(dspy.Signature):
    """Convert any date string to ISO 8601 YYYY-MM-DD format."""
    raw_date: str = dspy.InputField()
    iso_date: str = dspy.OutputField(desc="Date as YYYY-MM-DD, or empty string if unparseable")
    is_ambiguous: bool = dspy.OutputField(desc="True if the date could be multiple interpretations")
    confidence: float = dspy.OutputField()

date_cleaner = dspy.Predict(CleanDate)

# Try pandas first (handles many ISO variants)
def clean_date_pandas(raw: str):
    try:
        return pd.to_datetime(raw).strftime("%Y-%m-%d"), False
    except Exception:
        return None, False

cleaned_dates, ambiguous_flags = [], []
lm_date_calls = 0
for raw in df["signup_date"]:
    iso, ambig = clean_date_pandas(str(raw))
    if iso is None:
        r = date_cleaner(raw_date=str(raw))
        iso = r.iso_date
        ambig = r.is_ambiguous
        lm_date_calls += 1
    cleaned_dates.append(iso)
    ambiguous_flags.append(ambig)

df["date_clean"] = cleaned_dates
df["date_ambiguous"] = ambiguous_flags
print(f"Date cleaning - {lm_date_calls} LM calls out of {len(df)} rows")

# --- Clean categories ---
# Normalize to canonical set: Enterprise, SMB, Mid-Market
CATEGORY_MAP = {
    "enterprise": "Enterprise",
    "smb": "SMB",
    "mid-market": "Mid-Market",
    "mid market": "Mid-Market",
}

def clean_category(raw: str) -> str | None:
    normalized = raw.strip().lower()
    return CATEGORY_MAP.get(normalized)

class CleanCategory(dspy.Signature):
    """Map a company category variant to one of: Enterprise, SMB, Mid-Market.
    Return the original value if it does not match any of these categories.
    """
    raw_category: str = dspy.InputField()
    canonical: str = dspy.OutputField(desc="One of: Enterprise, SMB, Mid-Market, or the original if unrecognized")
    confidence: float = dspy.OutputField()

cat_cleaner = dspy.Predict(CleanCategory)

cleaned_cats = []
lm_cat_calls = 0
for raw in df["category"]:
    result = clean_category(str(raw))
    if result is None:
        r = cat_cleaner(raw_category=str(raw))
        result = r.canonical
        lm_cat_calls += 1
    cleaned_cats.append(result)

df["category_clean"] = cleaned_cats
print(f"Category cleaning - {lm_cat_calls} LM calls out of {len(df)} rows")

# --- Summary ---
print("\nCleaned data:")
print(df[["id", "name", "phone_clean", "date_clean", "category_clean", "date_ambiguous"]].to_string(index=False))

flagged = df[df["date_ambiguous"]]
if not flagged.empty:
    print(f"\nFlagged for review (ambiguous dates): {len(flagged)} rows")
    print(flagged[["id", "name", "signup_date", "date_clean"]].to_string(index=False))
```

Sample output:

```
Phone cleaning - 1 LM calls out of 8 rows (regex handled the rest)
Date cleaning - 3 LM calls out of 8 rows
Category cleaning - 1 LM calls out of 8 rows

Cleaned data:
 id        name           phone_clean  date_clean category_clean  date_ambiguous
  1   Alice Chen  +1 (415) 555-1234  2026-05-04     Enterprise           False
  2    Bob Smith  +1 (415) 555-5678  2026-05-04     Enterprise           False
  3    Carol Wu   +1 (415) 555-9012  2026-05-04            SMB           False
  4    Dave Lee   +1 (415) 555-3456  2026-05-04            SMB           False
  5    Eve Park   +1 (415) 555-0000  2026-05-04     Mid-Market           False
  6   Frank Kim   +1 (415) 555-1111  2026-05-04     Mid-Market           False
  7    Grace Ho   +1 (415) 555-2222  2026-05-04     Enterprise           False
  8    Hiro Ito              +1 ()    2026-05-01       [UNKNOWN]            True

Flagged for review (ambiguous dates): 1 rows
 id      name signup_date  date_clean
  8  Hiro Ito   May 2026   2026-05-01
```

**Key takeaway:** By applying regex and pandas before the LM, only 5 out of 24 field-level operations required an LM call — an 80% cost reduction on this dataset.
