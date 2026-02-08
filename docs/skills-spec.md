# Claude Code Skills Specification

> Reference for contributors writing new skills. Extracted from the [Claude Code skills documentation](https://code.claude.com/docs/en/skills).

## Skill Structure

Each skill is a directory with `SKILL.md` as the entrypoint:

```
my-skill/
├── SKILL.md           # Main instructions (required)
├── template.md        # Template for Claude to fill in
├── examples/
│   └── sample.md      # Example output showing expected format
└── scripts/
    └── validate.sh    # Script Claude can execute
```

## SKILL.md Format

Every `SKILL.md` has two parts:

1. **YAML frontmatter** (between `---` markers) — tells Claude when to use the skill
2. **Markdown content** — instructions Claude follows when the skill is invoked

```yaml
---
name: my-skill
description: What this skill does and when to use it
---

Your skill instructions here...
```

## Frontmatter Fields

| Field                      | Required    | Description                                                                                                      |
|:---------------------------|:------------|:-----------------------------------------------------------------------------------------------------------------|
| `name`                     | No          | Display name. If omitted, uses directory name. Lowercase, numbers, hyphens only (max 64 chars).                  |
| `description`              | Recommended | What the skill does. Claude uses this to decide when to apply the skill.                                          |
| `argument-hint`            | No          | Hint shown during autocomplete. Example: `[issue-number]` or `[filename]`.                                        |
| `disable-model-invocation` | No          | `true` = only user can invoke via `/name`. Default: `false`.                                                      |
| `user-invocable`           | No          | `false` = hidden from `/` menu, only Claude can invoke. Default: `true`.                                          |
| `allowed-tools`            | No          | Tools Claude can use without asking permission when skill is active.                                              |
| `model`                    | No          | Model to use when skill is active.                                                                                |
| `context`                  | No          | `fork` = run in a forked subagent context.                                                                        |
| `agent`                    | No          | Subagent type when `context: fork` is set. Options: `Explore`, `Plan`, `general-purpose`, or custom.              |

## String Substitutions

| Variable               | Description                                          |
|:-----------------------|:-----------------------------------------------------|
| `$ARGUMENTS`           | All arguments passed when invoking the skill.         |
| `$ARGUMENTS[N]`        | Specific argument by 0-based index.                   |
| `$N`                   | Shorthand for `$ARGUMENTS[N]`.                        |
| `${CLAUDE_SESSION_ID}` | Current session ID.                                   |

## Invocation Control

| Frontmatter                      | User can invoke | Claude can invoke | When loaded into context                                     |
|:---------------------------------|:----------------|:------------------|:-------------------------------------------------------------|
| (default)                        | Yes             | Yes               | Description always in context, full skill loads when invoked |
| `disable-model-invocation: true` | Yes             | No                | Description not in context, full skill loads when you invoke |
| `user-invocable: false`          | No              | Yes               | Description always in context, full skill loads when invoked |

## Where Skills Live

| Location   | Path                                     | Applies to                     |
|:-----------|:-----------------------------------------|:-------------------------------|
| Personal   | `~/.claude/skills/<skill-name>/SKILL.md` | All your projects              |
| Project    | `.claude/skills/<skill-name>/SKILL.md`   | This project only              |

## Supporting Files

Keep `SKILL.md` under 500 lines. Move detailed reference material to separate files and reference them:

```markdown
## Additional resources

- For complete API details, see [reference.md](reference.md)
- For usage examples, see [examples.md](examples.md)
```

## Dynamic Context

The `` !`command` `` syntax runs shell commands before skill content is sent to Claude:

```yaml
---
name: pr-summary
context: fork
agent: Explore
---

PR diff: !`gh pr diff`
```

## Best Practices

1. **Problem-first naming**: Name skills after what users are trying to do (e.g., `ai-sorting` not `dspy-classify`), not implementation details
2. **Clear descriptions**: Include keywords users would naturally say
3. **Keep SKILL.md focused**: Under 500 lines, use supporting files for details
4. **Test both invocation paths**: Direct `/skill-name` and automatic Claude invocation
5. **Use `disable-model-invocation: true`** for workflows with side effects

## Links

- [Skills documentation](https://code.claude.com/docs/en/skills)
- [Agent Skills standard](https://agentskills.io)
- [Subagents](https://code.claude.com/docs/en/sub-agents)
- [Plugins](https://code.claude.com/docs/en/plugins)
