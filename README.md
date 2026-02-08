# AI Skills for Claude Code

```bash
npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills -g --all -y
```

Build reliable AI features. Powered by [DSPy](https://dspy.ai/) — a framework that lets you program language models with composable modules instead of hand-writing prompts.

## What problem are you solving?

| Your problem | Skill | What it does |
|-------------|-------|--------------|
| "I'm starting a new AI feature" | [`/ai-kickoff`](skills/ai-kickoff/SKILL.md) | Scaffold a complete AI project with the right structure |
| "I need to auto-sort/tag/categorize content" | [`/ai-sorting`](skills/ai-sorting/SKILL.md) | Build AI that sorts tickets, tags emails, detects sentiment |
| "I need to search docs and answer questions" | [`/ai-searching-docs`](skills/ai-searching-docs/SKILL.md) | Build AI-powered knowledge base, help center, or doc Q&A |
| "I need AI to answer questions about our database" | [`/ai-querying-databases`](skills/ai-querying-databases/SKILL.md) | Text-to-SQL: plain English questions over Postgres, MySQL, Snowflake |
| "I need to condense long content into summaries" | [`/ai-summarizing`](skills/ai-summarizing/SKILL.md) | Summarize meetings, articles, threads — with length control |
| "I need to pull structured data from messy text" | [`/ai-parsing-data`](skills/ai-parsing-data/SKILL.md) | Parse invoices, extract entities, convert text to JSON |
| "I need AI to take actions and call APIs" | [`/ai-taking-actions`](skills/ai-taking-actions/SKILL.md) | Build AI that calls APIs, uses tools, and completes tasks |
| "I need AI to write articles, reports, or copy" | [`/ai-writing-content`](skills/ai-writing-content/SKILL.md) | Generate blog posts, product descriptions, newsletters |
| "My AI fails on hard problems that need planning" | [`/ai-reasoning`](skills/ai-reasoning/SKILL.md) | Add multi-step reasoning, Self-Discovery, chain-of-thought |
| "My task needs multiple AI steps" | [`/ai-building-pipelines`](skills/ai-building-pipelines/SKILL.md) | Chain classify, retrieve, generate, verify into one pipeline |
| "I need to verify AI output before users see it" | [`/ai-checking-outputs`](skills/ai-checking-outputs/SKILL.md) | Add guardrails, fact-checking, safety filters, and quality gates |
| "My AI makes stuff up / hallucinates" | [`/ai-stopping-hallucinations`](skills/ai-stopping-hallucinations/SKILL.md) | Ground AI in facts with citations, verification, and source checking |
| "My AI doesn't follow our rules" | [`/ai-following-rules`](skills/ai-following-rules/SKILL.md) | Enforce content policies, format rules, and business constraints |
| "My AI gives wrong answers" | [`/ai-improving-accuracy`](skills/ai-improving-accuracy/SKILL.md) | Measure quality, then systematically improve it |
| "My AI gives different answers every time" | [`/ai-making-consistent`](skills/ai-making-consistent/SKILL.md) | Lock down outputs so they're predictable and reliable |
| "My AI is too expensive" | [`/ai-cutting-costs`](skills/ai-cutting-costs/SKILL.md) | Reduce API costs with smart routing, caching, fine-tuning |
| "Let's fine-tune on our data" | [`/ai-fine-tuning`](skills/ai-fine-tuning/SKILL.md) | Train models on your data for max quality or cost savings |
| "Can we switch to a different model?" | [`/ai-switching-models`](skills/ai-switching-models/SKILL.md) | Switch providers, compare models, re-optimize automatically |
| "We don't have enough training data" | [`/ai-generating-data`](skills/ai-generating-data/SKILL.md) | Generate synthetic examples, fill data gaps, bootstrap from scratch |
| "How do I put my AI behind an API?" | [`/ai-serving-apis`](skills/ai-serving-apis/SKILL.md) | Wrap your AI in FastAPI endpoints for production serving |
| "Is our AI safe to launch?" | [`/ai-testing-safety`](skills/ai-testing-safety/SKILL.md) | Automatically find vulnerabilities with adversarial testing |
| "We need to moderate user content" | [`/ai-moderating-content`](skills/ai-moderating-content/SKILL.md) | Build AI content moderation with severity levels and routing |
| "Is our AI still working in production?" | [`/ai-monitoring`](skills/ai-monitoring/SKILL.md) | Monitor quality, safety, and cost — catch degradation early |
| "Why did my AI give that wrong answer?" | [`/ai-tracing-requests`](skills/ai-tracing-requests/SKILL.md) | Trace individual requests — see every LM call, retrieval, and step |
| "Which of our optimization experiments was best?" | [`/ai-tracking-experiments`](skills/ai-tracking-experiments/SKILL.md) | Log, compare, and promote optimization runs |
| "I need AI to score, grade, or evaluate things" | [`/ai-scoring`](skills/ai-scoring/SKILL.md) | Score essays, audit support quality, rate code reviews against rubrics |
| "My AI works on simple inputs but fails on complex ones" | [`/ai-decomposing-tasks`](skills/ai-decomposing-tasks/SKILL.md) | Break unreliable single-step tasks into reliable subtasks |
| "I need a conversational AI assistant" | [`/ai-building-chatbots`](skills/ai-building-chatbots/SKILL.md) | Build chatbots with memory, state, and doc-grounded responses |
| "I need multiple AI agents working together" | [`/ai-coordinating-agents`](skills/ai-coordinating-agents/SKILL.md) | Supervisor agents, specialist handoff, parallel research teams |
| "My AI is broken/erroring" | [`/ai-fixing-errors`](skills/ai-fixing-errors/SKILL.md) | Diagnose and fix crashes, wrong outputs, and weird behavior |

## Install

### Option 1: `npx skills` (recommended — works with any AI coding agent)

Install all 30 skills in one command. Works with Claude Code, Cursor, Codex, Cline, Windsurf, and [35+ other agents](https://agentskills.io).

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
/plugin install dspy-build-skills@dspy-skills     # Building AI features (15 skills)
/plugin install dspy-quality-skills@dspy-skills    # Quality and reliability (8 skills)
/plugin install dspy-ops-skills@dspy-skills        # Production operations (7 skills)
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

See [`docs/skills-spec.md`](docs/skills-spec.md) for the full skill format specification.

## Links

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Claude Code Skills](https://code.claude.com/docs/en/skills)
- [Agent Skills Standard](https://agentskills.io)
