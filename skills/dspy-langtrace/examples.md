# Langtrace Examples

## Trace a RAG pipeline and find slow requests

### Setup

```python
from langtrace_python_sdk import langtrace

langtrace.init(api_key="your-key")

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class SupportBot(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)

bot = SupportBot()
```

### Run queries (all automatically traced)

```python
questions = [
    "How do I reset my password?",
    "What's your refund policy?",
    "Can I upgrade my plan mid-cycle?",
    "How do I export my data?",
]

for q in questions:
    result = bot(question=q)
    print(f"Q: {q}\nA: {result.answer}\n")
```

### Find slow requests in the Langtrace UI

1. Go to [app.langtrace.ai](https://app.langtrace.ai) (or your self-hosted URL)
2. Open your project
3. Sort traces by latency (descending)
4. Click a slow trace to see the waterfall view
5. Check which step is the bottleneck:
   - **Retrieve slow?** Your vector DB may need optimization or the query is too broad
   - **LM call slow?** The model may be overloaded or the prompt is too long

## Trace with custom metadata for production filtering

```python
from langtrace_python_sdk import langtrace, with_langtrace_root_span

langtrace.init(api_key="your-key")

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

bot = SupportBot()

@with_langtrace_root_span("support-request")
def handle_support(user_id, plan, question):
    langtrace.inject_additional_attributes({
        "user_id": user_id,
        "plan": plan,
        "source": "api",
    })
    return bot(question=question)

# In production, filter traces by plan="enterprise" to debug issues for key accounts
result = handle_support("user-123", "enterprise", "How do I set up SSO?")
```

## Self-hosted: Docker Compose with Postgres persistence

```yaml
# docker-compose.override.yml — add to the Langtrace repo's docker-compose.yml
version: "3.8"
services:
  langtrace:
    environment:
      - DATABASE_URL=postgresql://langtrace:secret@db:5432/langtrace
    depends_on:
      - db
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: langtrace
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: langtrace
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

```bash
docker compose up -d
```

```python
from langtrace_python_sdk import langtrace

langtrace.init(api_host="http://localhost:3000")

# Traces persist across restarts in the Postgres volume
```
