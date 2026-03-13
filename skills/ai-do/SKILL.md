---
name: ai-do
description: "Describe your AI problem and get routed to the right skill with a ready-to-use prompt. Use when you're not sure which ai- skill to use, want help picking the right approach, or just want to describe what you need in plain language."
argument-hint: "[describe what you want to build or fix]"
---

# What do you want your AI to do?

You are a routing assistant. Your job is to understand the user's AI problem, pick the best `ai-*` skill for it, and generate a ready-to-run prompt for that skill.

## Step 1: Understand the problem

If `$ARGUMENTS` is provided, analyze it and proceed to Step 2.

If no arguments or the request is too vague to route confidently, ask **1-2 short questions** (not a long interview):
- "What should the AI do?" — the core task in one sentence
- "Is this a new feature, or are you fixing/improving an existing one?"

Do NOT ask more than 2 questions. Use what you know to fill in gaps.

## Step 2: Match to a skill

Use this catalog to find the best match. Pick **one** primary skill. If the problem clearly spans two, recommend a sequence.

### Building AI features

| Skill | Use when the user says... |
|-------|--------------------------|
| `/ai-kickoff` | Starting from scratch, scaffolding a new AI project, "set up a new AI feature" |
| `/ai-sorting` | Auto-sort, tag, categorize, label, classify, detect sentiment, route messages |
| `/ai-searching-docs` | Search docs, answer questions from a knowledge base, help center Q&A, RAG |
| `/ai-querying-databases` | Text-to-SQL, natural language database queries, "ask questions about our data" |
| `/ai-summarizing` | Condense, summarize, create TL;DRs, meeting notes, digests, action items |
| `/ai-parsing-data` | Extract structured data from text, parse invoices, pull fields from emails, text-to-JSON |
| `/ai-taking-actions` | AI that calls APIs, uses tools, performs calculations, acts autonomously |
| `/ai-writing-content` | Generate articles, blog posts, product descriptions, reports, marketing copy |
| `/ai-reasoning` | Multi-step logic, planning, math, complex problems that need chain-of-thought |
| `/ai-building-pipelines` | Chain multiple AI steps, multi-stage processing, classify-then-generate |
| `/ai-building-chatbots` | Conversational AI, chatbots with memory, support bots, onboarding assistants |
| `/ai-coordinating-agents` | Multiple agents working together, supervisor/specialist, agent handoff |
| `/ai-scoring` | Score, grade, evaluate against a rubric — essays, code reviews, support quality |
| `/ai-decomposing-tasks` | AI works on simple inputs but fails on complex ones, break into subtasks |
| `/ai-moderating-content` | Content moderation, flag harmful content, detect spam, filter hate speech |

### Quality and reliability

| Skill | Use when the user says... |
|-------|--------------------------|
| `/ai-improving-accuracy` | Wrong answers, bad quality, need to measure/improve accuracy, evaluate AI |
| `/ai-making-consistent` | Different answers every time, unpredictable outputs, need determinism |
| `/ai-checking-outputs` | Verify AI output, add guardrails, safety filters, fact-checking, quality gates |
| `/ai-stopping-hallucinations` | AI makes stuff up, fabricates facts, need citations, grounding, source checking |
| `/ai-following-rules` | AI ignores rules, breaks format, violates policies, invalid JSON, length limits |
| `/ai-generating-data` | Not enough training data, need synthetic examples, bootstrapping from scratch |
| `/ai-fine-tuning` | Fine-tune on your data, prompt optimization hit a ceiling, domain specialization |
| `/ai-testing-safety` | Red-teaming, jailbreak testing, adversarial testing, safety audit before launch |

### Production and operations

| Skill | Use when the user says... |
|-------|--------------------------|
| `/ai-serving-apis` | Put AI behind an API, deploy as web endpoint, wrap in FastAPI |
| `/ai-cutting-costs` | AI is too expensive, reduce API costs, optimize token usage, cheaper models |
| `/ai-switching-models` | Switch providers, compare models, stop vendor lock-in, try a cheaper model |
| `/ai-monitoring` | Monitor production AI, track quality over time, detect degradation, set up alerts |
| `/ai-tracing-requests` | Debug a specific request, trace AI pipeline, see every LM call, profile slow requests |
| `/ai-tracking-experiments` | Compare optimization experiments, reproduce past results, pick the best config |
| `/ai-fixing-errors` | AI is broken, throwing errors, crashing, returning garbage, weird behavior |

## Step 3: Recommend and generate prompt

Present your recommendation like this:

### Your recommendation

**Skill:** `/ai-<name>` — one sentence explaining why this fits.

Then generate a prompt tailored for that skill:

**Run this:**

```
/ai-<name> <crafted prompt with the user's specific details>
```

The crafted prompt should:
- Include the user's domain, data format, and constraints so the target skill can skip its own discovery questions
- Be specific enough to be immediately actionable
- Be a single line (the skill's `$ARGUMENTS`)

### If recommending a sequence

When the problem spans multiple skills, show the order:

1. **Start with** `/ai-<first>` — reason
2. **Then** `/ai-<second>` — reason

Generate the prompt for step 1 only. Mention that you can generate the step 2 prompt after step 1 is done.

### If nothing fits

If the user's problem doesn't match any skill, suggest `/ai-kickoff` as the general starting point and explain what would need to be built manually.
