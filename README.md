# AI Skills for Claude Code

Build reliable AI features. Powered by [DSPy](https://dspy.ai/) — a framework that lets you program language models with composable modules instead of hand-writing prompts.

## Quick start

The only skill you need is `/ai-do`. Describe what you want to build and it tells you which skill to use next.

```bash
npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do
```

Then in Claude Code:

```
/ai-do I want to build a support ticket classifier
```

It picks the right skill, generates a ready-to-run prompt, and tells you what to install. Or install everything at once:

```bash
npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --all
```

## What problem are you solving?

| Your problem | Skill | What it does |
|-------------|-------|--------------|
| "I want to build something with AI but not sure where to start" | [`/ai-do`](skills/ai-do/SKILL.md) | Describe your problem, get routed to the right skill with a ready-to-use prompt |
| "I'm starting a new AI feature" / "DSPy quickstart" | [`/ai-kickoff`](skills/ai-kickoff/SKILL.md) | Scaffold a complete AI project with the right structure |
| "I need to auto-sort/tag/categorize content" / "classification results are inconsistent" | [`/ai-sorting`](skills/ai-sorting/SKILL.md) | Build AI that sorts tickets, tags emails, detects sentiment |
| "I need to search docs and answer questions" / "retrieval returns irrelevant results" / "RAG pipeline tutorial" | [`/ai-searching-docs`](skills/ai-searching-docs/SKILL.md) | Build AI-powered knowledge base, help center, or doc Q&A |
| "I need AI to answer questions about our database" / "chat with your Postgres" | [`/ai-querying-databases`](skills/ai-querying-databases/SKILL.md) | Text-to-SQL: plain English questions over Postgres, MySQL, Snowflake |
| "I need to condense long content into summaries" | [`/ai-summarizing`](skills/ai-summarizing/SKILL.md) | Summarize meetings, articles, threads — with length control |
| "I need to pull structured data from messy text" / "the emails are messy and lack structure" | [`/ai-parsing-data`](skills/ai-parsing-data/SKILL.md) | Parse invoices, extract entities, convert text to JSON |
| "I need AI to take actions and call APIs" / "LLM function calling" | [`/ai-taking-actions`](skills/ai-taking-actions/SKILL.md) | Build AI that calls APIs, uses tools, and completes tasks |
| "I need AI to write articles, reports, or copy" | [`/ai-writing-content`](skills/ai-writing-content/SKILL.md) | Generate blog posts, product descriptions, newsletters |
| "My AI fails on hard problems that need planning" / "LLM can't do multi-step logic" | [`/ai-reasoning`](skills/ai-reasoning/SKILL.md) | Add multi-step reasoning, Self-Discovery, chain-of-thought |
| "My task needs multiple AI steps" / "LangChain LCEL alternative" | [`/ai-building-pipelines`](skills/ai-building-pipelines/SKILL.md) | Chain classify, retrieve, generate, verify into one pipeline |
| "I need to verify AI output before users see it" / "LLMs invent data points" | [`/ai-checking-outputs`](skills/ai-checking-outputs/SKILL.md) | Add guardrails, fact-checking, safety filters, and quality gates |
| "My AI makes stuff up / hallucinates" / "LLM makes up facts" | [`/ai-stopping-hallucinations`](skills/ai-stopping-hallucinations/SKILL.md) | Ground AI in facts with citations, verification, and source checking |
| "My AI doesn't follow our rules" / "LLM JSON output is unreliable" | [`/ai-following-rules`](skills/ai-following-rules/SKILL.md) | Enforce content policies, format rules, and business constraints |
| "My AI gives wrong answers" / "I spent hours tweaking prompts" | [`/ai-improving-accuracy`](skills/ai-improving-accuracy/SKILL.md) | Measure quality, then systematically improve it |
| "My AI gives different answers every time" / "same prompt, different results every run" | [`/ai-making-consistent`](skills/ai-making-consistent/SKILL.md) | Lock down outputs so they're predictable and reliable |
| "My AI is too expensive" / "LLM API costs too high" / "GPT-4 costs too much" | [`/ai-cutting-costs`](skills/ai-cutting-costs/SKILL.md) | Reduce API costs with smart routing, caching, fine-tuning |
| "Let's fine-tune on our data" / "prompt optimization hit a ceiling" | [`/ai-fine-tuning`](skills/ai-fine-tuning/SKILL.md) | Train models on your data for max quality or cost savings |
| "Can we switch to a different model?" / "prompt broke after model update" | [`/ai-switching-models`](skills/ai-switching-models/SKILL.md) | Switch providers, compare models, re-optimize automatically |
| "We don't have enough training data" / "no labeled data, need to bootstrap" | [`/ai-generating-data`](skills/ai-generating-data/SKILL.md) | Generate synthetic examples, fill data gaps, bootstrap from scratch |
| "How do I put my AI behind an API?" / "deploy LLM as API" / "productionize my AI" | [`/ai-serving-apis`](skills/ai-serving-apis/SKILL.md) | Wrap your AI in FastAPI endpoints for production serving |
| "Is our AI safe to launch?" / "prevent prompt injection" | [`/ai-testing-safety`](skills/ai-testing-safety/SKILL.md) | Automatically find vulnerabilities with adversarial testing |
| "We need to moderate user content" | [`/ai-moderating-content`](skills/ai-moderating-content/SKILL.md) | Build AI content moderation with severity levels and routing |
| "Is our AI still working in production?" / "silent quality drops, prompt drift" | [`/ai-monitoring`](skills/ai-monitoring/SKILL.md) | Monitor quality, safety, and cost — catch degradation early |
| "Why did my AI give that wrong answer?" | [`/ai-tracing-requests`](skills/ai-tracing-requests/SKILL.md) | Trace individual requests — see every LM call, retrieval, and step |
| "Which of our optimization experiments was best?" | [`/ai-tracking-experiments`](skills/ai-tracking-experiments/SKILL.md) | Log, compare, and promote optimization runs |
| "I need AI to score, grade, or evaluate things" / "LLM as a judge" | [`/ai-scoring`](skills/ai-scoring/SKILL.md) | Score essays, audit support quality, rate code reviews against rubrics |
| "My AI works on simple inputs but fails on complex ones" / "works on simple inputs but fails on complex ones" | [`/ai-decomposing-tasks`](skills/ai-decomposing-tasks/SKILL.md) | Break unreliable single-step tasks into reliable subtasks |
| "I need a conversational AI assistant" / "how do I build a chatbot" / "Intercom bot alternative" | [`/ai-building-chatbots`](skills/ai-building-chatbots/SKILL.md) | Build chatbots with memory, state, and doc-grounded responses |
| "I need multiple AI agents working together" / "CrewAI alternative" | [`/ai-coordinating-agents`](skills/ai-coordinating-agents/SKILL.md) | Supervisor agents, specialist handoff, parallel research teams |
| "My AI is broken/erroring" / "Could not parse LLM output" | [`/ai-fixing-errors`](skills/ai-fixing-errors/SKILL.md) | Diagnose and fix crashes, wrong outputs, and weird behavior |
| "DSPy can do X but there's no skill for it" | [`/ai-request-skill`](skills/ai-request-skill/SKILL.md) | Build a missing skill and submit a PR, or file a GitHub issue requesting it |

## Using a specific tool with DSPy?

| Tool | Skill | What it covers |
|------|-------|----------------|
| VizPy (prompt optimizer) | [`/dspy-vizpy`](skills/dspy-vizpy/SKILL.md) | Drop-in ContraPrompt/PromptGrad optimizers as alternative to GEPA/MIPROv2 |
| Langtrace | [`/dspy-langtrace`](skills/dspy-langtrace/SKILL.md) | Auto-instrument DSPy with one line, cloud + self-hosted tracing |
| Arize Phoenix | [`/dspy-phoenix`](skills/dspy-phoenix/SKILL.md) | Open-source trace viewer with built-in evals, local UI at localhost:6006 |
| W&B Weave | [`/dspy-weave`](skills/dspy-weave/SKILL.md) | Cloud experiment tracking and team dashboards via `@weave.op()` decorator |
| MLflow | [`/dspy-mlflow`](skills/dspy-mlflow/SKILL.md) | Auto-tracing, experiment tracking, and model registry for DSPy |
| LangWatch | [`/dspy-langwatch`](skills/dspy-langwatch/SKILL.md) | Auto-tracing + real-time optimizer progress dashboard |
| Langfuse | [`/dspy-langfuse`](skills/dspy-langfuse/SKILL.md) | Tracing + scoring + annotation queues + experiment tracking |
| Ragas | [`/dspy-ragas`](skills/dspy-ragas/SKILL.md) | Decomposed RAG evaluation: faithfulness, context precision/recall |
| Qdrant | [`/dspy-qdrant`](skills/dspy-qdrant/SKILL.md) | Official vector DB integration + custom retriever pattern for any DB |
| Ollama | [`/dspy-ollama`](skills/dspy-ollama/SKILL.md) | Run DSPy with local models, no API key needed |
| vLLM | [`/dspy-vllm`](skills/dspy-vllm/SKILL.md) | High-throughput production serving for self-hosted models |

## Know which DSPy concept you need?

If you already know DSPy and think in its vocabulary, use these API-first skills instead:

| DSPy concept | Skill | What it covers |
|-------------|-------|----------------|
| `Signature`, `InputField`, `OutputField` | [`/dspy-signatures`](skills/dspy-signatures/SKILL.md) | Inline and class-based signatures, typed fields, Pydantic models |
| `dspy.LM`, `dspy.configure` | [`/dspy-lm`](skills/dspy-lm/SKILL.md) | Provider strings, temperature/max_tokens, per-module LM assignment |
| `dspy.Assert`, `dspy.Suggest` | [`/dspy-assertions`](skills/dspy-assertions/SKILL.md) | Hard/soft constraints, backtracking, retry behavior, optimizer integration |
| `dspy.Module`, `forward()` | [`/dspy-modules`](skills/dspy-modules/SKILL.md) | Custom modules, composing sub-modules, save/load state |
| `dspy.Example`, `Prediction` | [`/dspy-data`](skills/dspy-data/SKILL.md) | `with_inputs()`, train/dev splits, loading from CSV/JSON/HuggingFace |
| `dspy.Evaluate`, metrics | [`/dspy-evaluate`](skills/dspy-evaluate/SKILL.md) | SemanticF1, exact match, LM-as-judge, composite metrics |
| `dspy.Predict` | [`/dspy-predict`](skills/dspy-predict/SKILL.md) | Direct LM calls, simplest inference module |
| `dspy.ChainOfThought` | [`/dspy-chain-of-thought`](skills/dspy-chain-of-thought/SKILL.md) | Step-by-step reasoning, `reasoning` field |
| `dspy.ProgramOfThought` | [`/dspy-program-of-thought`](skills/dspy-program-of-thought/SKILL.md) | Code generation + execution for math/computation |
| `dspy.ReAct` | [`/dspy-react`](skills/dspy-react/SKILL.md) | Tool-using agents, Reasoning-Action-Observation loop |
| `dspy.CodeAct` | [`/dspy-codeact`](skills/dspy-codeact/SKILL.md) | Agents that write and execute code to act |
| `dspy.MultiChainComparison` | [`/dspy-multi-chain-comparison`](skills/dspy-multi-chain-comparison/SKILL.md) | Multiple reasoning chains, pick the best |
| `dspy.BestOfN` | [`/dspy-best-of-n`](skills/dspy-best-of-n/SKILL.md) | Rejection sampling with a reward function |
| `dspy.Parallel` | [`/dspy-parallel`](skills/dspy-parallel/SKILL.md) | Concurrent LM calls, batch processing |
| `dspy.Refine` | [`/dspy-refine`](skills/dspy-refine/SKILL.md) | Iterative self-improvement with feedback |
| `dspy.RLM` | [`/dspy-rlm`](skills/dspy-rlm/SKILL.md) | Reinforcement-learning-style refinement |
| `dspy.BootstrapFewShot` | [`/dspy-bootstrap-few-shot`](skills/dspy-bootstrap-few-shot/SKILL.md) | Auto-generate few-shot demos, first optimizer to try |
| `dspy.BootstrapFewShotWithRandomSearch` | [`/dspy-bootstrap-rs`](skills/dspy-bootstrap-rs/SKILL.md) | Random search over candidate demo sets |
| `dspy.MIPROv2` | [`/dspy-miprov2`](skills/dspy-miprov2/SKILL.md) | Best prompt optimizer, instructions + demos jointly |
| `dspy.GEPA` | [`/dspy-gepa`](skills/dspy-gepa/SKILL.md) | Instruction generation and selection |
| `dspy.BetterTogether` | [`/dspy-better-together`](skills/dspy-better-together/SKILL.md) | Combined prompt + weight tuning |
| `dspy.BootstrapFinetune` | [`/dspy-bootstrap-finetune`](skills/dspy-bootstrap-finetune/SKILL.md) | Fine-tune weights from bootstrapped data |
| `dspy.COPRO` | [`/dspy-copro`](skills/dspy-copro/SKILL.md) | Instruction candidates with breadth search |
| `dspy.Ensemble` | [`/dspy-ensemble`](skills/dspy-ensemble/SKILL.md) | Combine multiple optimized programs |
| `dspy.InferRules` | [`/dspy-infer-rules`](skills/dspy-infer-rules/SKILL.md) | Extract decision logic from examples |
| `dspy.KNN`, `dspy.KNNFewShot` | [`/dspy-knn-few-shot`](skills/dspy-knn-few-shot/SKILL.md) | Embedding-based demo retrieval |
| `dspy.LabeledFewShot` | [`/dspy-labeled-few-shot`](skills/dspy-labeled-few-shot/SKILL.md) | Hand-picked demonstrations |
| `dspy.SIMBA` | [`/dspy-simba`](skills/dspy-simba/SKILL.md) | Small-step incremental optimization |
| `ChatAdapter`, `JSONAdapter`, `TwoStepAdapter` | [`/dspy-adapters`](skills/dspy-adapters/SKILL.md) | Prompt formatting, structured output, reasoning models |
| `dspy.Tool`, `PythonInterpreter` | [`/dspy-tools`](skills/dspy-tools/SKILL.md) | Wrapping functions as tools, code execution |
| `dspy.Retrieve`, `ColBERTv2`, `Embedder` | [`/dspy-retrieval`](skills/dspy-retrieval/SKILL.md) | Search, RAG pipelines, embeddings |
| `dspy.Image`, `dspy.Audio`, `dspy.Code`, `dspy.History` | [`/dspy-primitives`](skills/dspy-primitives/SKILL.md) | Multimodal inputs, conversation history |
| `StreamListener`, `inspect_history`, `save`/`load` | [`/dspy-utils`](skills/dspy-utils/SKILL.md) | Streaming, caching, debugging, persistence, async |
| VizPy (`ContraPromptOptimizer`, `PromptGradOptimizer`) | [`/dspy-vizpy`](skills/dspy-vizpy/SKILL.md) | Commercial drop-in prompt optimizer, alternative to GEPA |
| Langtrace (`langtrace.init`) | [`/dspy-langtrace`](skills/dspy-langtrace/SKILL.md) | Auto-instrument DSPy, cloud + self-hosted LLM observability |
| Arize Phoenix (`DSPyInstrumentor`) | [`/dspy-phoenix`](skills/dspy-phoenix/SKILL.md) | Open-source trace viewer with evals, local UI |
| W&B Weave (`@weave.op()`) | [`/dspy-weave`](skills/dspy-weave/SKILL.md) | Cloud experiment tracking and team dashboards |
| MLflow (`mlflow.dspy.autolog()`) | [`/dspy-mlflow`](skills/dspy-mlflow/SKILL.md) | Auto-tracing, experiment tracking, model registry |
| LangWatch (`langwatch.dspy.init`) | [`/dspy-langwatch`](skills/dspy-langwatch/SKILL.md) | Auto-tracing and real-time optimizer progress |
| Langfuse (`DSPyInstrumentor`, `@observe`) | [`/dspy-langfuse`](skills/dspy-langfuse/SKILL.md) | Tracing + scoring + annotation queues + experiments |
| Ragas (`ragas.evaluate`) | [`/dspy-ragas`](skills/dspy-ragas/SKILL.md) | Decomposed RAG evaluation with LLM-as-judge metrics |
| Qdrant (`QdrantRM`) | [`/dspy-qdrant`](skills/dspy-qdrant/SKILL.md) | Vector DB retriever with hybrid search |
| Ollama (`ollama_chat/`) | [`/dspy-ollama`](skills/dspy-ollama/SKILL.md) | Local model serving for development |
| vLLM (`openai/` + local server) | [`/dspy-vllm`](skills/dspy-vllm/SKILL.md) | Production self-hosted model serving |

## Install

### Option 1: `npx skills` (recommended — works with any AI coding agent)

Install all 80 skills in one command. Works with Claude Code, Cursor, Codex, Cline, Windsurf, and [35+ other agents](https://agentskills.io).

```bash
npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills
```

The CLI will prompt you to pick which skills and which agents to install. Or install everything non-interactively:

```bash
npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --all -y
```

To install globally (available in all your projects):

```bash
npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills -g
```

### Option 2: Claude Code plugin marketplace

```bash
# In Claude Code, run:
/plugin marketplace add lebsral/DSPy-Programming-not-prompting-LMs-skills
```

Then install a skill group:

```bash
/plugin install dspy-build-skills@dspy-skills     # Building AI features (17 skills)
/plugin install dspy-quality-skills@dspy-skills    # Quality and reliability (8 skills)
/plugin install dspy-ops-skills@dspy-skills        # Production operations (7 skills)
/plugin install dspy-api-skills@dspy-skills        # DSPy API-first skills (32 skills)
```

### Option 3: Manual (git clone)

```bash
git clone https://github.com/lebsral/DSPy-Programming-not-prompting-LMs-skills.git
```

Copy skills to your agent's skill directory:

```bash
# Claude Code
cp -r DSPy-Programming-not-prompting-LMs-skills/skills/* ~/.claude/skills/

# Cursor
cp -r DSPy-Programming-not-prompting-LMs-skills/skills/* ~/.cursor/skills/
```

Or symlink to stay in sync with updates:

```bash
ln -s "$(pwd)/DSPy-Programming-not-prompting-LMs-skills/skills/"* ~/.claude/skills/
```

### Managing skills

```bash
npx skills list          # See what you have installed
npx skills check         # Check for updates
npx skills update        # Update all installed skills
npx skills remove        # Uninstall skills
```

### Keeping skills up to date

These skills are actively improved. To get the latest versions:

```bash
npx skills check          # See what's changed
npx skills update         # Pull latest versions
```

For manual/symlink installs, run `git pull` in your cloned repo.

> **Migrating from v1.12.1 or earlier?** A YAML formatting change in v1.12.2 means `npx skills update` may fail on some skills. Reinstall to fix:
> ```bash
> npx skills remove dspy-skills
> npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills -g -s '*'
> ```

### Adding new skills after initial install

`npx skills update` only updates skills you already have. When new skills are added to this repo (like the `dspy-` API-first skills), re-run the add command to pick them up:

```bash
# Interactive — choose which new skills to add
npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills

# Non-interactive — add all new skills automatically
npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --all -y
```

This won't duplicate skills you already have — it only adds the ones that are missing.

## Use a skill

In Claude Code (or any agent that supports the [Agent Skills](https://agentskills.io) standard):

- **Invoke directly:** `/ai-sorting` or `/ai-kickoff my-project`
- **Ask naturally:** "Help me sort support tickets into categories" — the agent picks the right skill

## How It Works

Each skill is a directory under `skills/` containing:

- **`SKILL.md`** — Main instructions Claude follows (YAML frontmatter + markdown)
- **`examples.md`** — Worked examples (loaded on demand)
- **`reference.md`** — Detailed reference material (loaded on demand)

Under the hood, skills use [DSPy](https://dspy.ai/) — a framework for building AI features with composable modules that compile into optimized prompts. You don't need to know DSPy to use these skills; they guide you through everything.

Skills follow the [Claude Code skills format](https://code.claude.com/docs/en/skills) and the [Agent Skills](https://agentskills.io) open standard.

## Reference Docs

- [`docs/dspy-reference.md`](docs/dspy-reference.md) — DSPy API quick reference (modules, optimizers, patterns)
- [`docs/langchain-langgraph-reference.md`](docs/langchain-langgraph-reference.md) — LangChain & LangGraph API quick reference (loaders, tools, StateGraph)
- [`docs/skills-spec.md`](docs/skills-spec.md) — Claude Code skills specification (for contributors)

## Contributing

### Adding a new skill

1. Create `skills/ai-<problem>/SKILL.md` — name it after the problem, not the DSPy concept
2. Add YAML frontmatter with `name` and `description` (include phrases users would naturally say)
3. Write step-by-step instructions in the markdown body
4. Add `examples.md` and/or `reference.md` for supporting content
5. Update the problem catalog table in this README
6. Test with `/ai-<problem>` in Claude Code
7. Bump the patch version in `.claude-plugin/marketplace.json`

See [`docs/skills-spec.md`](docs/skills-spec.md) for the full skill format specification.

## Links

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Claude Code Skills](https://code.claude.com/docs/en/skills)
- [Agent Skills Standard](https://agentskills.io)
