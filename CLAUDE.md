# DSPy Skills Repository

This is a collection of Claude Code skills for building AI features with DSPy.

## Conventions

- **Problem-first naming**: Skills are named after the problem the user is solving, using the `ai-` prefix (e.g., `ai-sorting` not `dspy-classify`). The user thinks "I need to sort tickets", not "I need to use ChainOfThought".
- **Web developer language**: Descriptions use phrases web/SaaS developers actually say ("sort tickets", "search docs", "parse data"), not ML jargon.
- **Skill format**: Each skill follows the Claude Code `SKILL.md` format with YAML frontmatter. See `docs/skills-spec.md` for the full spec.
- **Keep SKILL.md files under 500 lines**. Move detailed reference material to supporting files like `reference.md` or `examples.md`.
- **DSPy API reference**: See `docs/dspy-reference.md` for DSPy modules, optimizers, and patterns. Reference this when building or modifying skills.
- **Provider-agnostic**: Skills should work with any LM provider DSPy supports (OpenAI, Anthropic, local models, etc.). Don't hardcode specific providers.

## Repo Structure

```
skills/           # Claude Code skills (each has SKILL.md)
docs/             # Reference documentation
examples/         # Full working example projects
```

## Adding a New Skill

1. Create a directory under `skills/` named `ai-<problem>`
2. Write `SKILL.md` with frontmatter (`name`, `description`) and instructions
3. Add `examples.md` or `reference.md` for supporting content
4. Update the problem catalog table in `README.md`
5. Test with `/ai-<problem>` in Claude Code
