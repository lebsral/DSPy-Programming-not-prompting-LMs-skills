# AI Skills for Claude Code

Build reliable AI features. Powered by [DSPy](https://dspy.ai/) — a framework that lets you program language models with composable modules instead of hand-writing prompts.

## What problem are you solving?

| Your problem | Skill | What it does |
|-------------|-------|--------------|
| "I'm starting a new AI feature" | [`/ai-kickoff`](skills/ai-kickoff/SKILL.md) | Scaffold a complete AI project with the right structure |
| "I need to auto-sort/tag/categorize content" | [`/ai-sorting`](skills/ai-sorting/SKILL.md) | Build AI that sorts tickets, tags emails, detects sentiment |
| "I need to search docs and answer questions" | [`/ai-searching-docs`](skills/ai-searching-docs/SKILL.md) | Build AI-powered knowledge base, help center, or doc Q&A |
| "I need to pull structured data from messy text" | [`/ai-parsing-data`](skills/ai-parsing-data/SKILL.md) | Parse invoices, extract entities, convert text to JSON |
| "I need AI to take actions and call APIs" | [`/ai-taking-actions`](skills/ai-taking-actions/SKILL.md) | Build AI that calls APIs, uses tools, and completes tasks |
| "My task needs multiple AI steps" | [`/ai-building-pipelines`](skills/ai-building-pipelines/SKILL.md) | Chain classify, retrieve, generate, verify into one pipeline |
| "I need to verify AI output before users see it" | [`/ai-checking-outputs`](skills/ai-checking-outputs/SKILL.md) | Add guardrails, fact-checking, safety filters, and quality gates |
| "My AI makes stuff up / hallucinates" | [`/ai-stopping-hallucinations`](skills/ai-stopping-hallucinations/SKILL.md) | Ground AI in facts with citations, verification, and source checking |
| "My AI doesn't follow our rules" | [`/ai-following-rules`](skills/ai-following-rules/SKILL.md) | Enforce content policies, format rules, and business constraints |
| "My AI gives wrong answers" | [`/ai-improving-accuracy`](skills/ai-improving-accuracy/SKILL.md) | Measure quality, then systematically improve it |
| "My AI gives different answers every time" | [`/ai-making-consistent`](skills/ai-making-consistent/SKILL.md) | Lock down outputs so they're predictable and reliable |
| "My AI is too expensive" | [`/ai-cutting-costs`](skills/ai-cutting-costs/SKILL.md) | Reduce API costs with smart routing, caching, fine-tuning |
| "Let's fine-tune on our data" | [`/ai-fine-tuning`](skills/ai-fine-tuning/SKILL.md) | Train models on your data for max quality or cost savings |
| "My AI is broken/erroring" | [`/ai-fixing-errors`](skills/ai-fixing-errors/SKILL.md) | Diagnose and fix crashes, wrong outputs, and weird behavior |

## Quick Start

### Install skills (available across all your projects)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/DSPy-Programming-not-prompting-LMs-skills.git

# Copy skills to your Claude Code personal skills directory
cp -r DSPy-Programming-not-prompting-LMs-skills/skills/* ~/.claude/skills/
```

Or symlink to stay in sync with updates:

```bash
ln -s "$(pwd)/DSPy-Programming-not-prompting-LMs-skills/skills/"* ~/.claude/skills/
```

### Use a skill

In Claude Code, either:
- **Invoke directly:** `/ai-sorting` or `/ai-kickoff my-project`
- **Ask naturally:** "Help me sort support tickets into categories" — Claude picks the right skill

## How It Works

Each skill is a directory under `skills/` containing:

- **`SKILL.md`** — Main instructions Claude follows (YAML frontmatter + markdown)
- **`examples.md`** — Worked examples (loaded on demand)
- **`reference.md`** — Detailed reference material (loaded on demand)

Under the hood, skills use [DSPy](https://dspy.ai/) — a framework for building AI features with composable modules that compile into optimized prompts. You don't need to know DSPy to use these skills; they guide you through everything.

Skills follow the [Claude Code skills format](https://code.claude.com/docs/en/skills) and the [Agent Skills](https://agentskills.io) open standard.

## Reference Docs

- [`docs/dspy-reference.md`](docs/dspy-reference.md) — DSPy API quick reference (modules, optimizers, patterns)
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
