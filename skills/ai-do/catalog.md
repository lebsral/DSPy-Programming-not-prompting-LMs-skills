# Complete Skill Catalog

Every skill in the repo, organized by category. This is the authoritative list — if a skill isn't here, `ai-do` can't route to it.

## Problem-first skills (`ai-` prefix)

For users who describe what they need in plain language.

### Building AI features

| Skill | What it does |
|-------|-------------|
| `/ai-kickoff` | Scaffold a new AI feature powered by DSPy |
| `/ai-sorting` | Auto-sort, categorize, or label content using AI |
| `/ai-searching-docs` | Build AI that searches your documents and answers questions |
| `/ai-querying-databases` | Build AI that answers questions about your database |
| `/ai-summarizing` | Condense long content into short summaries using AI |
| `/ai-parsing-data` | Pull structured data from messy text using AI |
| `/ai-taking-actions` | Build AI that takes actions, calls APIs, and does things autonomously |
| `/ai-writing-content` | Generate articles, reports, blog posts, or marketing copy with AI |
| `/ai-reasoning` | Make AI solve hard problems that need planning and multi-step thinking |
| `/ai-building-pipelines` | Chain multiple AI steps into one reliable pipeline |
| `/ai-building-chatbots` | Build a conversational AI assistant with memory and state |
| `/ai-coordinating-agents` | Build multiple AI agents that work together |
| `/ai-scoring` | Score, grade, or evaluate things using AI against a rubric |
| `/ai-decomposing-tasks` | Break a failing complex AI task into reliable subtasks |
| `/ai-moderating-content` | Auto-moderate what users post on your platform |

### Quality and reliability

| Skill | What it does |
|-------|-------------|
| `/ai-improving-accuracy` | Measure and improve how well your AI works |
| `/ai-making-consistent` | Make your AI give the same answer every time |
| `/ai-checking-outputs` | Verify and validate AI output before it reaches users |
| `/ai-stopping-hallucinations` | Stop your AI from making things up |
| `/ai-following-rules` | Make your AI follow rules and policies |
| `/ai-generating-data` | Generate synthetic training data when you don't have enough real examples |
| `/ai-fine-tuning` | Fine-tune models on your data to maximize quality and cut costs |
| `/ai-testing-safety` | Find every way users can break your AI before they do |

### Production and operations

| Skill | What it does |
|-------|-------------|
| `/ai-serving-apis` | Put your AI behind an API |
| `/ai-cutting-costs` | Reduce your AI API bill |
| `/ai-switching-models` | Switch AI providers or models without breaking things |
| `/ai-monitoring` | Know when your AI breaks in production |
| `/ai-tracing-requests` | See exactly what your AI did on a specific request |
| `/ai-tracking-experiments` | Track which optimization experiment was best |
| `/ai-fixing-errors` | Fix broken AI features |

### Meta

| Skill | What it does |
|-------|-------------|
| `/ai-request-skill` | Request or contribute a new AI skill that doesn't exist yet |

## API-first skills (`dspy-` prefix)

For users who already know DSPy and want a specific concept or tool.

### Core modules

| DSPy concept | Skill |
|-------------|-------|
| Signatures, InputField, OutputField | `/dspy-signatures` |
| dspy.LM, dspy.configure, providers | `/dspy-lm` |
| dspy.Assert, dspy.Suggest | `/dspy-assertions` |
| dspy.Module, forward() | `/dspy-modules` |
| dspy.Example, Prediction, datasets | `/dspy-data` |
| dspy.Evaluate, metrics | `/dspy-evaluate` |

### Prompting strategies

| DSPy concept | Skill |
|-------------|-------|
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

### Optimizers

| DSPy concept | Skill |
|-------------|-------|
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

### Infrastructure

| DSPy concept | Skill |
|-------------|-------|
| ChatAdapter, JSONAdapter, TwoStepAdapter | `/dspy-adapters` |
| dspy.Tool, PythonInterpreter | `/dspy-tools` |
| dspy.Retrieve, ColBERTv2, Embedder | `/dspy-retrieval` |
| dspy.Image, Audio, Code, History | `/dspy-primitives` |
| StreamListener, inspect_history, save/load | `/dspy-utils` |

### Ecosystem tools

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

## Example multi-skill sequences

These show how to combine `ai-` and `dspy-` skills for real-world projects.

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
