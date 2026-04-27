# DSPy Skills Repository

This is a collection of Claude Code skills for building AI features with DSPy.

## Conventions

- **Dual naming convention**: Skills use two naming schemes:
  - `ai-` prefix (problem-first): named after the problem — `ai-sorting`, `ai-parsing-data`. For users who think "I need to sort tickets."
  - `dspy-` prefix (API-first): named after the DSPy concept or ecosystem tool — `dspy-signatures`, `dspy-chain-of-thought`, `dspy-vizpy`. For users who already know DSPy and think "I want to use ChainOfThought." Also used for third-party tools that integrate with DSPy.
- **Web developer language**: `ai-` skill descriptions use phrases web/SaaS developers actually say ("sort tickets", "search docs", "parse data"), not ML jargon. `dspy-` skill descriptions explicitly reference DSPy API names.
- **Skill format**: Each skill follows the Claude Code `SKILL.md` format with YAML frontmatter. See `docs/skills-spec.md` for the full spec.
- **Keep SKILL.md files under 500 lines**. Move detailed reference material to supporting files like `reference.md` or `examples.md`.
- **DSPy API reference**: See `docs/dspy-reference.md` for DSPy modules, optimizers, and patterns. Reference this when building or modifying skills.
- **Provider-agnostic**: Skills should work with any LM provider DSPy supports (OpenAI, Anthropic, local models, etc.). Don't hardcode specific providers.

## Workspace Directory

Skill eval workspaces (from the skill-creator plugin and similar tools) go in `.workspaces/` at the repo root — NOT as siblings to skill directories. When creating workspace output for a skill, use `.workspaces/<skill-name>/` instead of `skills/<skill-name>-workspace/`. This keeps the `skills/` directory clean.

## Repo Structure

```
skills/           # Claude Code skills (each has SKILL.md)
docs/             # Reference documentation
examples/         # Full working example projects
.workspaces/      # Skill eval workspaces (gitignored)
```

## Adding or Updating a Skill

1. Create a directory under `skills/` named `ai-<problem>` or `dspy-<concept>`
2. Write `SKILL.md` with frontmatter (`name`, `description`) and instructions
3. Add `examples.md` or `reference.md` for supporting content
4. Update the problem catalog table in `README.md`
5. Test with `/ai-<problem>` in Claude Code
6. Bump the patch version in `.claude-plugin/marketplace.json`
