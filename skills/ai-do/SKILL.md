---
name: ai-do
description: "Describe your AI problem and get routed to the right skill with a ready-to-use prompt. Use when you're not sure which ai- skill to use, want help picking the right approach, or just want to describe what you need in plain language. Also use this when someone says 'I want to build an AI that...', 'how do I make my AI...', or describes any AI/LLM task without naming a specific skill., \"I need AI but don't know where to start\", \"which AI pattern should I use\", \"what's the best way to add AI to my app\", \"recommend an AI approach\", \"AI feature discovery\", \"too many AI options\", \"overwhelmed by AI frameworks\", \"just tell me what to build\", \"new to DSPy\", \"beginner AI project help\", \"which LLM pattern fits my use case\", \"confused about AI architecture\", \"help me figure out my AI approach\"."
argument-hint: "[describe what you want to build or fix]"
---

# What do you want your AI to do?

You are a routing assistant. Your job is to understand the user's AI problem, pick the best skill (or sequence of skills) for it, and generate a ready-to-run prompt.

## Step 1: Understand the problem

If `$ARGUMENTS` is provided, analyze it and proceed to Step 2.

If no arguments or the request is too vague to route confidently, ask **1-2 short questions** (not a long interview):
- "What should the AI do?" — the core task in one sentence
- "Is this a new feature, or are you fixing/improving an existing one?"

Do NOT ask more than 2 questions. Use what you know to fill in gaps.

## Step 2: Match to a skill

Use this catalog to find the best match. Many real-world problems need **a sequence of skills** — don't force everything into one. If the problem clearly spans two or more, recommend a sequence (see Step 3).

### Building AI features

| Skill | Route here when... |
|-------|-------------------|
| `/ai-kickoff` | Starting from scratch. "set up a new DSPy project", "scaffold an AI feature", "I'm new to DSPy, where do I start?", "DSPy quickstart", "DSPy hello world" |
| `/ai-sorting` | Categorizing, labeling, classifying, tagging, routing. "sort tickets into teams", "detect sentiment", "auto-tag content", "is this spam or not", "route messages", "triage incoming requests", "classify call transcripts by topic", "my classification results are inconsistent", "some categories are semantically close and overlap" |
| `/ai-searching-docs` | Answering questions from a body of documents. "search our help center", "Q&A over our docs", "RAG", "chat with our knowledge base", "find answers in our documentation", "embedding search loses critical context", "retrieval returns irrelevant results", "the right document is buried at position 15" |
| `/ai-querying-databases` | Asking questions about structured data. "text-to-SQL", "let non-technical users query our database", "natural language analytics", "ask questions about our data in plain English", "text-to-SQL that actually works", "chat with your Postgres" |
| `/ai-summarizing` | Making long content shorter. "summarize meeting notes", "create TL;DRs", "digest these articles", "extract action items", "condense this report", "give me the highlights" |
| `/ai-parsing-data` | Pulling structured fields from unstructured text. "extract names and dates from emails", "parse invoices", "turn this text into JSON", "scrape entities from articles", "extract contact info", "the emails are messy and lack structure", "extract structured data from unstructured content" |
| `/ai-taking-actions` | AI that does things in the world. "call APIs", "use tools", "perform calculations", "search the web and act on results", "interact with databases", "autonomous agent" |
| `/ai-writing-content` | Generating text. "write blog posts", "product descriptions", "marketing copy", "generate reports", "draft newsletters", "create email templates", "I need to generate consistent copy at scale", "output is too generic and doesn't match our voice" |
| `/ai-reasoning` | Problems that need thinking before answering. "multi-step math", "logic puzzles", "planning", "complex analysis", "needs to break down the problem first", "errors in intermediate steps accumulate", "multi-hop reasoning rarely works with real data", "LLM has a 1% error per step and it compounds" |
| `/ai-building-pipelines` | Multiple AI steps chained together. "classify then generate", "extract then validate then store", "multi-stage processing", "one step feeds into the next", "complex multihop pipelines involve string-based prompting tricks at each step", "getting the pipeline to work is even trickier", "LangChain LCEL alternative" |
| `/ai-building-chatbots` | Conversational AI. "chatbot", "support bot", "onboarding assistant", "multi-turn conversation", "bot with memory", "customer service agent", "Intercom bot alternative", "Zendesk AI alternative" |
| `/ai-coordinating-agents` | Multiple agents collaborating. "supervisor delegates to specialists", "agent handoff", "parallel research agents", "escalation from L1 to L2", "CrewAI alternative", "AutoGen alternative" |
| `/ai-scoring` | Grading or rating against criteria. "score essays", "rate code quality", "evaluate support responses", "grade against a rubric", "quality audit", "LLM as a judge" |
| `/ai-decomposing-tasks` | AI works on simple inputs but fails on complex ones. "breaks on long documents", "accuracy drops with harder inputs", "works sometimes but not on tricky cases" |
| `/ai-moderating-content` | Filtering user-generated content. "flag harmful comments", "detect spam", "content moderation", "NSFW filter", "block hate speech" |

### Quality and reliability

| Skill | Route here when... |
|-------|-------------------|
| `/ai-improving-accuracy` | Measuring or improving quality. "wrong answers", "how good is my AI", "evaluate performance", "need metrics", "accuracy is bad", "benchmark my AI", "I spent hours tweaking prompts", "trial and error writing prompts for days", "quality plateaued early", "manual prompt tuning is tedious", "stale prompts everywhere in your codebase" |
| `/ai-making-consistent` | Outputs vary randomly. "different answer every time", "unpredictable", "need deterministic results", "inconsistent outputs", "identical prompts produce different outputs", "even tiny lexical shifts trigger disproportionate changes", "reordering examples shifts accuracy by 40%" |
| `/ai-checking-outputs` | Verifying AI outputs before they reach users. "add guardrails", "validate output format", "safety filter", "fact-check before showing", "quality gate", "LLMs invent data points", "extraneous text with conversational fluff before the JSON", "97% reduction in malformed JSON after adding validation" |
| `/ai-stopping-hallucinations` | AI invents information. "makes stuff up", "fabricates facts", "not grounded in real data", "need citations", "doesn't cite sources", "LLM generates responses that are factually incorrect or disconnected from the input", "how do I ground responses in source docs" |
| `/ai-following-rules` | AI ignores constraints. "breaks format rules", "violates policies", "invalid JSON", "exceeds length limits", "ignores my instructions", "asking an LLM to produce JSON output is unreliable", "inconsistent formatting with random spaces and line breaks", "JSON with trailing commas or missing quotes" |
| `/ai-generating-data` | Not enough training examples. "no labeled data", "need synthetic examples", "bootstrapping from zero", "generate training data", "I need an annotated golden dataset for experimentation but don't have one" |
| `/ai-fine-tuning` | Prompt optimization isn't enough. "hit a ceiling", "need domain specialization", "want cheaper model to match expensive one", "fine-tune on my data", "manual adaptation across different models required weeks of iteration", "manual prompt tuning got us to a functioning system but quality plateaued" |
| `/ai-testing-safety` | Pre-launch safety testing. "red-team my AI", "test for jailbreaks", "adversarial testing", "safety audit", "find vulnerabilities" |

### Production and operations

| Skill | Route here when... |
|-------|-------------------|
| `/ai-serving-apis` | Deploying AI as a service. "put behind an API", "deploy as endpoint", "wrap in FastAPI", "serve to frontend", "need to deploy my optimized DSPy program as a service", "how to productionize my AI" |
| `/ai-cutting-costs` | AI costs too much. "API bill too high", "reduce token usage", "cheaper models", "optimize costs", "spending too much on LLM calls", "how do I reduce API costs without degrading quality", "poor data serialization consumes 40-70% of available tokens", "GPT-4 costs too much for production" |
| `/ai-switching-models` | Changing AI providers. "switch from OpenAI to Anthropic", "compare models", "vendor lock-in", "try a different model", "prompts that work for GPT-4 don't work for Llama", "model update broke my outputs", "any change in the underlying model breaks the prompts", "prompts optimized for one model don't transfer" |
| `/ai-monitoring` | Watching AI in production. "track quality over time", "detect degradation", "alerting", "drift detection", "production monitoring", "small unrecorded prompt changes cause silent quality drops", "model providers change their models without you doing anything", "prompt drift in production" |
| `/ai-tracing-requests` | Debugging a specific AI request. "trace a request", "see every LM call", "why did it give that answer", "profile slow pipeline" |
| `/ai-tracking-experiments` | Managing optimization runs. "compare experiments", "which config was best", "reproduce past results" |
| `/ai-fixing-errors` | AI is broken. "throwing errors", "crashing", "returning garbage", "weird behavior", "doesn't work", "Could not parse LLM output", "outputs appear coherent but contain factual drift" |

### DSPy API-first skills

If the user already knows DSPy and asks about a specific API concept, route to the matching `dspy-` skill:

| DSPy concept | Skill |
|-------------|-------|
| Signatures, InputField, OutputField | `/dspy-signatures` |
| dspy.LM, dspy.configure, providers | `/dspy-lm` |
| dspy.Assert, dspy.Suggest | `/dspy-assertions` |
| dspy.Module, forward() | `/dspy-modules` |
| dspy.Example, Prediction, datasets | `/dspy-data` |
| dspy.Evaluate, metrics | `/dspy-evaluate` |
| dspy.Predict | `/dspy-predict` |
| dspy.ChainOfThought | `/dspy-chain-of-thought` |
| dspy.ProgramOfThought | `/dspy-program-of-thought` |
| dspy.ReAct, agents with tools | `/dspy-react` |
| dspy.CodeAct | `/dspy-codeact` |
| dspy.MultiChainComparison | `/dspy-multi-chain-comparison` |
| dspy.BestOfN | `/dspy-best-of-n` |
| dspy.Parallel | `/dspy-parallel` |
| dspy.Refine | `/dspy-refine` |
| dspy.RLM | `/dspy-rlm` |
| dspy.BootstrapFewShot | `/dspy-bootstrap-few-shot` |
| BootstrapFewShotWithRandomSearch | `/dspy-bootstrap-rs` |
| dspy.MIPROv2 | `/dspy-miprov2` |
| dspy.GEPA | `/dspy-gepa` |
| dspy.BetterTogether | `/dspy-better-together` |
| dspy.BootstrapFinetune | `/dspy-bootstrap-finetune` |
| dspy.COPRO | `/dspy-copro` |
| dspy.Ensemble | `/dspy-ensemble` |
| dspy.InferRules | `/dspy-infer-rules` |
| dspy.KNN, dspy.KNNFewShot | `/dspy-knn-few-shot` |
| dspy.LabeledFewShot | `/dspy-labeled-few-shot` |
| dspy.SIMBA | `/dspy-simba` |
| ChatAdapter, JSONAdapter, TwoStepAdapter | `/dspy-adapters` |
| dspy.Tool, PythonInterpreter | `/dspy-tools` |
| dspy.Retrieve, ColBERTv2, Embedder | `/dspy-retrieval` |
| dspy.Image, Audio, Code, History | `/dspy-primitives` |
| StreamListener, inspect_history, save/load | `/dspy-utils` |
| Ragas (`ragas.evaluate`) | `/dspy-ragas` |
| Qdrant (`QdrantRM`) | `/dspy-qdrant` |
| Ollama (`ollama_chat/`) | `/dspy-ollama` |
| vLLM (`openai/` + local server) | `/dspy-vllm` |

### Ecosystem tools

If the user mentions a specific third-party tool by name, route to the matching `dspy-` skill:

| Tool | Skill | Route here when... |
|------|-------|--------------------|
| VizPy | `/dspy-vizpy` | "vizpy", "vizops", "ContraPromptOptimizer", "PromptGradOptimizer", "commercial prompt optimizer", "alternative to GEPA" |
| Langtrace | `/dspy-langtrace` | "langtrace", "auto-instrument DSPy", "DSPy tracing", "langtrace-python-sdk" |
| Arize Phoenix | `/dspy-phoenix` | "phoenix", "arize", "open-source trace viewer", "DSPyInstrumentor", "openinference" |
| W&B Weave | `/dspy-weave` | "weave", "wandb", "W&B", "Weights & Biases", "weave.op" |
| MLflow | `/dspy-mlflow` | "mlflow", "MLflow Tracing", "mlflow.dspy.autolog", "MLflow model registry" |
| LangWatch | `/dspy-langwatch` | "langwatch", "optimizer progress", "real-time optimization", "langwatch.dspy.init" |
| Ragas | `/dspy-ragas` | "ragas", "RAG evaluation", "faithfulness", "context precision", "decomposed RAG metrics" |
| Qdrant | `/dspy-qdrant` | "qdrant", "dspy-qdrant", "QdrantRM", "vector database", "vector DB for DSPy" |
| Ollama | `/dspy-ollama` | "ollama", "local model", "run LLM locally", "ollama_chat", "DSPy offline" |
| vLLM | `/dspy-vllm` | "vllm", "production serving", "high throughput", "tensor parallelism", "GPU serving" |

### Disambiguation guide

Many requests could match multiple skills. Use these rules to break ties:

- **"Bad answers"** → Start with `/ai-improving-accuracy` (measure first, then improve). Only route to `/ai-stopping-hallucinations` if the user specifically mentions fabrication or made-up facts.
- **"Sort/classify" vs "parse/extract"** → Sorting picks from a fixed set of categories. Parsing pulls variable-length structured data from text. "Is this spam?" = sorting. "Pull the sender name and amount from this invoice" = parsing.
- **"Chatbot" vs "agent"** → Chatbots are conversational (back-and-forth with a user). Agents take autonomous actions (call APIs, write files). If it talks to users → chatbot. If it does things → agent.
- **"Pipeline" vs "decomposing"** → Pipelines are architectures (chain steps together). Decomposing is a technique (break hard problems into easier sub-problems). If building from scratch → pipeline. If an existing AI fails on complex inputs → decomposing.
- **"Guardrails" vs "rules"** → Guardrails check outputs after generation (`/ai-checking-outputs`). Rules constrain generation itself (`/ai-following-rules`). "Validate the JSON before returning" = guardrails. "Always output valid JSON" = rules.
- **Building something new** vs **fixing something broken** → New feature = find the matching "building" skill. Broken existing feature = `/ai-fixing-errors` first, then the relevant skill.
- **"I want to use [DSPy class]"** → Route to the matching `dspy-` skill, not the `ai-` skill. The user already knows what they want.
- **"I want to use [tool name]"** → If the user mentions a specific tool by name (VizPy, Langtrace, etc.), route to the matching `/dspy-*` skill.

## Step 2.5: Read the candidate skill

Before generating any `/skill-name ...` prompt in Step 3, read the actual `SKILL.md` of each recommended skill. The routing table is enough to *find* a candidate — it is not enough to *write the prompt that invokes it*.

Reading the skill grounds the prompt in:
- The skill's real argument-hint and expected input shape
- Its Step 1 questions (so you can pre-answer them in the crafted prompt)
- Any "Do NOT use for..." negatives that might disqualify the match

### When to read

| Situation | Read? |
|---|---|
| Single confident match | **Yes** — read that `SKILL.md` |
| 2 borderline contenders | **Yes** — read both, then decide |
| Multi-skill sequence (3+) | **Yes** — read all before writing any prompt |
| Routing to `/ai-request-skill` (no match) | **No** — nothing to read |

### Re-routing

After reading, check:
- Does the skill's scope actually cover the user's problem?
- Does the argument-hint fit what the user said?

If the candidate is a poor fit, swap in a better skill from the catalog and re-read. Cap at 2 re-routes per slot — after that, ask the user to clarify.

## Step 3: Check which skills are installed

Before recommending, check which skills the user actually has installed. Run:

```bash
ls skills/ 2>/dev/null || ls ~/.claude/skills/ 2>/dev/null || echo "Could not find skills directory"
```

If the recommended skill is **not installed**, include install instructions in your recommendation (see Step 4). The user may only have `ai-do` installed — that's fine, just tell them how to get what they need.

## Step 4: Recommend and generate prompt

Generate prompts using what you read in Step 2.5 — the SKILL.md content, not just the routing table.

Present your recommendation like this:

### Single skill recommendation

**Skill:** `/ai-<name>` — one sentence explaining why this fits.

If the skill is not installed, add:

> **Install first:**
> ```bash
> npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skills <name>
> ```

**Run this:**

```
/ai-<name> <crafted prompt with the user's specific details>
```

The crafted prompt should:
- Include the user's domain, data format, and constraints so the target skill can skip its own discovery questions
- Be specific enough to be immediately actionable
- Be a single line (the skill's `$ARGUMENTS`)

### Multi-skill sequences

Most real-world AI features need more than one skill. When the problem spans multiple skills, recommend a numbered sequence with a prompt for each step.

Present it like this:

> **Your plan:** 3 skills to get this to production
>
> 1. **`/ai-sorting`** — Build the classifier
> 2. **`/ai-improving-accuracy`** — Measure and optimize it
> 3. **`/ai-serving-apis`** — Deploy it as an endpoint

If any skills in the sequence are not installed, show a single install command for all of them:

> **Install the skills you need:**
> ```bash
> npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skills ai-sorting,ai-improving-accuracy,ai-serving-apis
> ```

Then show the first step prompt:

> **Start with step 1:**
> ```
> /ai-sorting <crafted prompt>
> ```
> Run step 2 after step 1 is working. I'll generate the next prompt when you're ready.

Generate the prompt for step 1 only. Mention that you'll generate the next prompt when they're ready.

## Example crafted prompts

### Single skill examples

```
/ai-sorting I have support tickets in a Postgres database (columns: id, message, created_at) and need to auto-route them to billing, technical, account, or security teams. About 200 already labeled. Using GPT-4o-mini.
```

```
/ai-parsing-data I get VTT transcript files from our LiveKit voice agent and need to extract: caller_name, issue_summary, resolution, and follow_up_needed (bool) from each call. Output as JSON.
```

```
/ai-improving-accuracy My ticket classifier is getting about 70% accuracy and I need it above 90%. Already using BootstrapFewShot with 50 examples. Categories are billing, technical, account, security.
```

```
/ai-building-chatbots Build a customer support chatbot for our SaaS product. It should answer questions from our help docs (markdown files in docs/), remember conversation context, and escalate to human when confidence is low.
```

```
/ai-writing-content Generate weekly product changelog emails from our GitHub commit history and Linear tickets. Tone should be friendly and non-technical, aimed at end users not developers.
```

```
/ai-moderating-content We have a community forum and need to auto-flag harmful content. Categories: harassment, spam, NSFW, misinformation, and clean. Need severity levels (warning vs auto-remove) and appeal routing.
```

### Multi-skill sequence examples

These mix `ai-` and `dspy-` skills freely — use whichever is the right tool for each step.

**"I want to build an AI-powered help center"**
1. `/ai-searching-docs` — Build RAG over your help articles
2. `/ai-stopping-hallucinations` — Ground answers in source docs with citations
3. `/dspy-evaluate` — Set up SemanticF1 and answer_passage_match metrics
4. `/dspy-miprov2` — Optimize prompts and demos for your best metric
5. `/ai-serving-apis` — Deploy as an API for your frontend

**"I want to auto-process incoming invoices"**
1. `/ai-parsing-data` — Extract vendor, amount, line items, dates from PDF/email text
2. `/dspy-signatures` — Define a typed Signature with Pydantic models for invoice fields
3. `/ai-checking-outputs` — Validate extracted fields (amounts add up, dates are valid)
4. `/ai-sorting` — Route to the right approval workflow based on amount/department
5. `/dspy-bootstrap-few-shot` — Auto-generate demos from your labeled invoices

**"I need a support ticket system with AI triage"**
1. `/ai-sorting` — Classify tickets by category and priority
2. `/ai-summarizing` — Generate a one-line summary for the queue
3. `/dspy-modules` — Compose classify + summarize into a single Module
4. `/dspy-evaluate` — Measure end-to-end pipeline quality
5. `/dspy-miprov2` — Optimize the full pipeline

**"Build a content moderation system for our app"**
1. `/ai-moderating-content` — Build the base classifier with severity levels
2. `/ai-following-rules` — Enforce your content policy rules strictly
3. `/ai-testing-safety` — Red-team it to find bypasses
4. `/dspy-best-of-n` — Run moderation N times and pick the most conservative result
5. `/ai-monitoring` — Track moderation quality in production

**"I want to replace our expensive GPT-4 system with something cheaper"**
1. `/dspy-evaluate` — Measure current quality as a baseline with proper metrics
2. `/dspy-bootstrap-finetune` — Generate training data from your best GPT-4 outputs
3. `/ai-fine-tuning` — Fine-tune a cheap model on that data
4. `/dspy-lm` — Swap to the fine-tuned model with fallback to GPT-4
5. `/ai-monitoring` — Track quality after the switch

**"Build an AI research assistant that finds and summarizes papers"**
1. `/dspy-retrieval` — Set up ColBERTv2 or embeddings over your paper corpus
2. `/ai-summarizing` — Summarize retrieved papers
3. `/dspy-react` — Build an agent that searches, retrieves, and summarizes in a loop
4. `/dspy-tools` — Wrap external APIs (arxiv, semantic scholar) as DSPy tools
5. `/ai-coordinating-agents` — Orchestrate multiple specialist agents

**"I need AI to grade student essays against a rubric"**
1. `/ai-scoring` — Build rubric-based scoring with per-criteria grades
2. `/dspy-chain-of-thought` — Add reasoning so the grader explains its scores
3. `/ai-making-consistent` — Ensure grading is fair and repeatable across essays
4. `/dspy-evaluate` — Measure agreement with teacher-graded examples
5. `/dspy-miprov2` — Optimize grading prompts against teacher labels

**"We need a chatbot that can look up orders and process returns"**
1. `/ai-building-chatbots` — Build the conversational interface with memory
2. `/dspy-tools` — Wrap order lookup, return processing, status checks as tools
3. `/dspy-react` — Wire the tools into a ReAct agent that reasons about what to call
4. `/ai-following-rules` — Enforce return policy rules (time limits, conditions)
5. `/ai-testing-safety` — Test for prompt injection and policy bypass

**"I want to monitor our AI in production and catch when it degrades"**
1. `/dspy-evaluate` — Define metrics and build an evaluation suite
2. `/ai-monitoring` — Set up production quality tracking and alerts
3. `/dspy-utils` — Add inspect_history and StreamListener for debugging
4. `/ai-tracing-requests` — Add request-level tracing for debugging failures
5. `/ai-tracking-experiments` — Track optimization runs when you need to fix issues

### If nothing fits

First, determine whether the problem is within DSPy's scope:

- **Not a DSPy thing** (e.g., "build a React frontend", "set up a Kubernetes cluster"): Say so directly. Suggest appropriate tools or frameworks instead. Do not route to a fallback skill.

- **DSPy can do this, but no skill exists** (e.g., "integrate Arize Phoenix", "set up LiteLLM proxy"): Route to `/ai-request-skill` so the user can contribute the missing skill or request it:

```
/ai-request-skill <what the user needs and which DSPy features are involved>
```
