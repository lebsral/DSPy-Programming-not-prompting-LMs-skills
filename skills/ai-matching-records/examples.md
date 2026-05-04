# Record Matching Examples

## CRM Contact Deduplication

Match people by name, email, and phone across messy imported data:

```python
import dspy
import jellyfish
from itertools import combinations

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class CompareContacts(dspy.Signature):
    """Determine if two CRM contact records refer to the same person.
    Email match is strong evidence. Name variations like 'Bob'/'Robert' are common.
    Phone numbers may differ in formatting but represent the same line."""
    record_a: str = dspy.InputField(desc="First contact as field: value pairs")
    record_b: str = dspy.InputField(desc="Second contact as field: value pairs")
    is_match: bool = dspy.OutputField(desc="True if both contacts are the same person")
    confidence: float = dspy.OutputField(desc="Confidence from 0.0 to 1.0")
    explanation: str = dspy.OutputField(desc="Key reason for match or non-match decision")

compare = dspy.ChainOfThought(CompareContacts)

# Sample CRM data with messy duplicates
contacts = [
    {"id": "1", "name": "Robert Johnson", "email": "rjohnson@acme.com", "phone": "415-555-0101", "company": "Acme Corp"},
    {"id": "2", "name": "Bob Johnson",    "email": "r.johnson@acme.com", "phone": "4155550101",   "company": "ACME"},
    {"id": "3", "name": "Sarah Lee",      "email": "sarah@techco.io",    "phone": "212-555-0199", "company": "TechCo"},
    {"id": "4", "name": "Sara Lee",       "email": "slee@techco.io",     "phone": "212-555-0199", "company": "TechCo Inc"},
    {"id": "5", "name": "James Park",     "email": "jpark@startup.com",  "phone": "",             "company": "Startup"},
]

def format_contact(c):
    return "\n".join(f"{k}: {v}" for k, v in c.items() if k != "id" and v)

def block_contacts(contacts):
    """Block by Soundex of last name to reduce pairs."""
    buckets = {}
    for c in contacts:
        last = c["name"].split()[-1] if c["name"] else ""
        key = jellyfish.soundex(last.lower())
        buckets.setdefault(key, []).append(c)
    pairs = []
    for bucket in buckets.values():
        if len(bucket) > 1:
            pairs.extend(combinations(bucket, 2))
    return pairs

# Block then score
candidate_pairs = block_contacts(contacts)
print(f"Comparing {len(candidate_pairs)} candidate pairs (down from {len(contacts)*(len(contacts)-1)//2} total)")

results = []
for a, b in candidate_pairs:
    result = compare(record_a=format_contact(a), record_b=format_contact(b))
    results.append({
        "ids": (a["id"], b["id"]),
        "names": (a["name"], b["name"]),
        "is_match": result.is_match,
        "confidence": result.confidence,
        "explanation": result.explanation,
    })
    print(f"  {a['name']} vs {b['name']} - match={result.is_match} ({result.confidence:.2f}): {result.explanation}")

# Route by confidence
auto_merge   = [r for r in results if r["confidence"] >= 0.9]
needs_review = [r for r in results if 0.6 <= r["confidence"] < 0.9]
rejected     = [r for r in results if r["confidence"] < 0.6]

print(f"\nAuto-merge: {len(auto_merge)}, Needs review: {len(needs_review)}, Rejected: {len(rejected)}")
```

Expected output:
```
Comparing 2 candidate pairs (down from 10 total)
  Robert Johnson vs Bob Johnson - match=True (0.92): Same email domain and phone number, common nickname Robert/Bob
  Sarah Lee vs Sara Lee - match=True (0.88): Same phone and company, name is a common spelling variant
```

## Company Name Matching

Match legal entity names to canonical records, handling abbreviations, suffixes, and acronyms:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class MatchCompanyName(dspy.Signature):
    """Determine whether two company name strings refer to the same legal entity.
    Common variations - abbreviations (IBM/International Business Machines),
    legal suffixes (Corp/Corporation/Inc/Ltd), punctuation, and parent/subsidiary names."""
    name_a: str = dspy.InputField(desc="First company name string")
    name_b: str = dspy.InputField(desc="Second company name string")
    is_match: bool = dspy.OutputField(desc="True if both names refer to the same company")
    confidence: float = dspy.OutputField(desc="Confidence from 0.0 to 1.0")
    explanation: str = dspy.OutputField()

company_matcher = dspy.ChainOfThought(MatchCompanyName)

# Test cases spanning easy to hard
test_pairs = [
    ("IBM", "International Business Machines Corp."),
    ("Apple Inc.", "Apple Computer, Inc."),
    ("3M", "Minnesota Mining and Manufacturing Company"),
    ("Goldman Sachs", "Goldman Sachs Group, Inc."),
    ("Amazon", "Amazon Web Services"),         # parent vs subsidiary - tricky
    ("Microsoft", "MicroSoft Corporation"),
    ("Google LLC", "Alphabet Inc."),            # subsidiary vs parent - distinct
]

for name_a, name_b in test_pairs:
    result = company_matcher(name_a=name_a, name_b=name_b)
    verdict = "MATCH" if result.is_match else "DISTINCT"
    print(f"[{verdict} {result.confidence:.2f}] '{name_a}' vs '{name_b}'")
    print(f"  {result.explanation}")

# Optimize with labeled examples
trainset = [
    dspy.Example(
        name_a="IBM",
        name_b="International Business Machines",
        is_match=True, confidence=1.0,
        explanation="IBM is the universally recognized abbreviation for International Business Machines"
    ).with_inputs("name_a", "name_b"),
    dspy.Example(
        name_a="Apple Inc.",
        name_b="Apple Records",
        is_match=False, confidence=0.95,
        explanation="Different companies - Apple Inc. is a tech company, Apple Records is a music label"
    ).with_inputs("name_a", "name_b"),
    dspy.Example(
        name_a="Meta Platforms Inc.",
        name_b="Facebook, Inc.",
        is_match=True, confidence=0.98,
        explanation="Facebook rebranded to Meta Platforms in 2021 - same legal entity"
    ).with_inputs("name_a", "name_b"),
]

def company_metric(example, pred, trace=None):
    return float(pred.is_match == example.is_match)

optimizer = dspy.BootstrapFewShot(metric=company_metric, max_bootstrapped_demos=3)
optimized = optimizer.compile(company_matcher, trainset=trainset)
optimized.save("company_name_matcher.json")
```

## Support Ticket Deduplication

Find duplicate tickets describing the same underlying issue so you do not work the same bug twice:

```python
import dspy
from itertools import combinations

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class CompareTickets(dspy.Signature):
    """Determine whether two support tickets describe the same underlying issue.
    Different users may describe the same bug in completely different words.
    Focus on the root problem, not surface wording or affected user."""
    ticket_a: str = dspy.InputField(desc="First ticket - title and description")
    ticket_b: str = dspy.InputField(desc="Second ticket - title and description")
    is_duplicate: bool = dspy.OutputField(desc="True if both tickets describe the same root issue")
    confidence: float = dspy.OutputField(desc="Confidence from 0.0 to 1.0")
    explanation: str = dspy.OutputField(desc="What makes them the same or different issue")

ticket_deduper = dspy.ChainOfThought(CompareTickets)

# Open tickets to deduplicate
tickets = [
    {
        "id": "T-101",
        "title": "Login page not loading",
        "description": "When I go to app.example.com/login the page spins forever and never loads. Started this morning.",
    },
    {
        "id": "T-102",
        "title": "Cannot access my account",
        "description": "The sign-in screen is broken. Tried Chrome and Firefox - both stuck on loading. Happening since 9am.",
    },
    {
        "id": "T-103",
        "title": "Password reset email not arriving",
        "description": "I requested a password reset 30 minutes ago and the email never came. Checked spam folder.",
    },
    {
        "id": "T-104",
        "title": "Authentication is down",
        "description": "Our whole team cannot log in. The login endpoint appears to be returning 503 errors.",
    },
    {
        "id": "T-105",
        "title": "Forgot password flow broken",
        "description": "Reset password emails are not being sent. Multiple users affected.",
    },
]

def format_ticket(t):
    return f"Title: {t['title']}\nDescription: {t['description']}"

def block_tickets_by_tokens(tickets, min_shared=2):
    """Block tickets sharing at least min_shared title tokens."""
    def tokens(t):
        stopwords = {"the", "a", "an", "is", "not", "my", "i", "and", "or", "to", "in"}
        return set(t["title"].lower().split()) - stopwords

    pairs = []
    for a, b in combinations(tickets, 2):
        if len(tokens(a) & tokens(b)) >= min_shared:
            pairs.append((a, b))
    return pairs

candidate_pairs = block_tickets_by_tokens(tickets)
print(f"Candidate pairs after blocking: {len(candidate_pairs)}")

# Score pairs
matches = []
for a, b in candidate_pairs:
    result = ticket_deduper(
        ticket_a=format_ticket(a),
        ticket_b=format_ticket(b),
    )
    if result.is_duplicate and result.confidence >= 0.7:
        matches.append((a["id"], b["id"], result.confidence, result.explanation))
        print(f"  DUPLICATE ({result.confidence:.2f}): {a['id']} + {b['id']}")
        print(f"    {result.explanation}")

# Transitive closure - group all related tickets
parent = {t["id"]: t["id"] for t in tickets}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    parent[find(x)] = find(y)

for a_id, b_id, _, _ in matches:
    union(a_id, b_id)

# Print duplicate groups
groups = {}
for t in tickets:
    root = find(t["id"])
    groups.setdefault(root, []).append(t["id"])

print("\nDuplicate groups:")
for root, members in groups.items():
    if len(members) > 1:
        print(f"  Group [{root}]: {', '.join(members)} - merge into one ticket")
    else:
        print(f"  Unique: {members[0]}")
```

Expected output:
```
Candidate pairs after blocking: 3
  DUPLICATE (0.91): T-101 + T-102
    Both describe login/authentication page failing to load, same timeframe
  DUPLICATE (0.88): T-101 + T-104
    Both report login system down with loading failures
  DUPLICATE (0.85): T-103 + T-105
    Both report password reset emails not being delivered

Duplicate groups:
  Group [T-101]: T-101, T-102, T-104 - merge into one ticket
  Group [T-103]: T-103, T-105 - merge into one ticket
  Unique: T-104
```
