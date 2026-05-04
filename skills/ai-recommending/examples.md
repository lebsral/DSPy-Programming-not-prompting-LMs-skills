# ai-recommending - Examples

## Example 1 - Product recommender for e-commerce

A user has a purchase history. Retrieve similar products from the catalog, then re-rank with the LM to surface the top 5 with friendly explanations.

```python
import dspy
import numpy as np

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# --- Item catalog ---
item_catalog = {
    "p001": {"title": "Trail Running Shoes", "description": "Lightweight shoes for off-road running"},
    "p002": {"title": "Waterproof Hiking Jacket", "description": "Breathable jacket for wet conditions"},
    "p003": {"title": "Trekking Poles", "description": "Collapsible carbon fibre poles"},
    "p004": {"title": "Running Socks 3-Pack", "description": "Moisture-wicking merino socks"},
    "p005": {"title": "Yoga Mat", "description": "Non-slip mat for studio or home use"},
    "p006": {"title": "Road Cycling Helmet", "description": "Aerodynamic helmet for road cyclists"},
}

# --- Stub embeddings (replace with your embedding model) ---
rng = np.random.default_rng(42)
item_embeddings = {pid: rng.random(64) for pid in item_catalog}

# User purchased trail shoes and hiking jacket — build profile
purchase_history = ["p001", "p002"]
history_titles = [item_catalog[pid]["title"] for pid in purchase_history]
user_profile = f"Recently purchased - {', '.join(history_titles)}. Interested in trail running and outdoor activities."

# Approximate user embedding as mean of purchased item embeddings
user_embedding = np.mean([item_embeddings[pid] for pid in purchase_history], axis=0)

# --- Re-ranker ---
class ReRankRecommendations(dspy.Signature):
    """Reorder the candidate items below, placing the most relevant first for this user.
    Do not reveal internal scoring. Write friendly explanations referencing why each item fits the user."""
    user_profile: str = dspy.InputField()
    candidate_items: str = dspy.InputField()
    num_results: int = dspy.InputField()
    ranked_items: list[str] = dspy.OutputField(desc="Item IDs in ranked order, most relevant first")
    explanations: list[str] = dspy.OutputField(desc="One friendly sentence per item")

reranker = dspy.Predict(ReRankRecommendations)

# --- Retrieve top-N candidates by cosine similarity ---
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

candidate_ids = sorted(item_embeddings, key=lambda pid: cosine_sim(user_embedding, item_embeddings[pid]), reverse=True)[:6]
candidates_text = "\n".join(
    f"{i+1}. [{cid}] {item_catalog[cid]['title']} - {item_catalog[cid]['description']}"
    for i, cid in enumerate(candidate_ids)
)

result = reranker(user_profile=user_profile, candidate_items=candidates_text, num_results=5)

print("Top 5 recommendations:")
for item_id, explanation in zip(result.ranked_items[:5], result.explanations[:5]):
    item_id = item_id.strip("[]").strip()
    title = item_catalog.get(item_id, {}).get("title", item_id)
    print(f"  {title} - {explanation}")
```

Expected output (varies by LM):
```
Top 5 recommendations:
  Trekking Poles - Perfect for your trail running and hiking adventures.
  Running Socks 3-Pack - Great pairing with your trail running shoes.
  Waterproof Hiking Jacket - Complements your existing outdoor kit for wet days.
  Trail Running Shoes - A natural follow-up if you need a second pair or a size up.
  Road Cycling Helmet - Expands your outdoor activity range beyond trails.
```

---

## Example 2 - Article recommender for a blog

A reader has viewed several articles. Recommend related articles from the content library based on their reading history.

```python
import dspy
import numpy as np

lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929")  # or "openai/gpt-4o-mini", etc.
dspy.configure(lm=lm)

article_catalog = {
    "a001": {"title": "Getting Started with DSPy", "description": "Intro to DSPy signatures and modules"},
    "a002": {"title": "Optimizing LM Pipelines", "description": "Using BootstrapFewShot and MIPROv2"},
    "a003": {"title": "Building RAG Systems", "description": "Retrieval-augmented generation patterns"},
    "a004": {"title": "DSPy Assertions Guide", "description": "Using Refine and BestOfN for quality control"},
    "a005": {"title": "Fine-tuning vs Prompting", "description": "When to fine-tune and when to prompt"},
    "a006": {"title": "Evaluating LLM Outputs", "description": "Metrics, judges, and eval frameworks"},
}

rng = np.random.default_rng(7)
article_embeddings = {aid: rng.random(64) for aid in article_catalog}

read_history = ["a001", "a003"]
history_titles = [article_catalog[aid]["title"] for aid in read_history]
user_profile = f"Read articles about - {', '.join(history_titles)}. Interested in practical DSPy usage and RAG systems."

user_embedding = np.mean([article_embeddings[aid] for aid in read_history], axis=0)

class ArticleRecommendation(dspy.Signature):
    """Reorder the candidate articles below, placing the most relevant first for this reader.
    Do not mention similarity scores. Explain why each article continues their learning journey."""
    user_profile: str = dspy.InputField()
    candidate_items: str = dspy.InputField()
    num_results: int = dspy.InputField()
    ranked_items: list[str] = dspy.OutputField(desc="Article IDs in ranked order")
    explanations: list[str] = dspy.OutputField(desc="One sentence per article - why it fits this reader")

reranker = dspy.Predict(ArticleRecommendation)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# Exclude already-read articles from candidates
unread_ids = [aid for aid in article_catalog if aid not in read_history]
candidate_ids = sorted(unread_ids, key=lambda aid: cosine_sim(user_embedding, article_embeddings[aid]), reverse=True)[:5]
candidates_text = "\n".join(
    f"{i+1}. [{aid}] {article_catalog[aid]['title']} - {article_catalog[aid]['description']}"
    for i, aid in enumerate(candidate_ids)
)

result = reranker(user_profile=user_profile, candidate_items=candidates_text, num_results=3)

print("Recommended articles:")
for aid, explanation in zip(result.ranked_items[:3], result.explanations[:3]):
    aid = aid.strip("[]").strip()
    title = article_catalog.get(aid, {}).get("title", aid)
    print(f"  {title}")
    print(f"    {explanation}")
```

---

## Example 3 - Support article suggester

A user submits a support ticket. Match the ticket text to relevant help docs before routing to a human agent, reducing ticket volume.

```python
import dspy
import numpy as np

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

help_docs = {
    "h001": {"title": "How to reset your password", "description": "Step-by-step password reset instructions"},
    "h002": {"title": "Billing and invoice questions", "description": "How to download invoices and update payment methods"},
    "h003": {"title": "Cancelling your subscription", "description": "How to cancel, pause, or downgrade your plan"},
    "h004": {"title": "Two-factor authentication setup", "description": "Enable and manage 2FA on your account"},
    "h005": {"title": "Exporting your data", "description": "Download your account data in CSV or JSON format"},
    "h006": {"title": "Connecting third-party integrations", "description": "Set up Slack, Zapier, and other integrations"},
}

rng = np.random.default_rng(99)
doc_embeddings = {did: rng.random(64) for did in help_docs}

# Simulate ticket text as user profile signal
ticket_text = "I cannot log into my account. I forgot my password and the reset email is not arriving."
user_profile = f"Support ticket - {ticket_text}"

# In production, embed the ticket text with your embedding model
# Here we use a random vector as a placeholder
ticket_embedding = rng.random(64)

class SupportDocSuggestion(dspy.Signature):
    """Given a support ticket, reorder the candidate help articles below to surface the most useful ones first.
    Do not mention internal scores. Write a short sentence explaining how each article addresses the user's issue."""
    user_profile: str = dspy.InputField(desc="Support ticket text describing the user's problem")
    candidate_items: str = dspy.InputField(desc="Numbered list of help articles with descriptions")
    num_results: int = dspy.InputField()
    ranked_items: list[str] = dspy.OutputField(desc="Help doc IDs in ranked order, most relevant first")
    explanations: list[str] = dspy.OutputField(desc="One sentence per doc explaining how it helps")

reranker = dspy.Predict(SupportDocSuggestion)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

candidate_ids = sorted(doc_embeddings, key=lambda did: cosine_sim(ticket_embedding, doc_embeddings[did]), reverse=True)[:5]
candidates_text = "\n".join(
    f"{i+1}. [{did}] {help_docs[did]['title']} - {help_docs[did]['description']}"
    for i, did in enumerate(candidate_ids)
)

result = reranker(user_profile=user_profile, candidate_items=candidates_text, num_results=3)

print(f"Suggested articles for ticket: '{ticket_text[:60]}...'")
for did, explanation in zip(result.ranked_items[:3], result.explanations[:3]):
    did = did.strip("[]").strip()
    title = help_docs.get(did, {}).get("title", did)
    print(f"  [{did}] {title}")
    print(f"    {explanation}")
```

Expected output:
```
Suggested articles for ticket: 'I cannot log into my account. I forgot my password and...'
  [h001] How to reset your password
    Directly addresses your forgotten password and reset email issue.
  [h004] Two-factor authentication setup
    Relevant if 2FA is blocking login after the password reset.
  [h002] Billing and invoice questions
    Less likely to be relevant but included as a fallback.
```
