# Skill Authoring Standards

Checklist and guidelines for writing DSPy skills, based on Anthropic's "Lessons from Building Claude Code" blog post and internal experience.

## Required structure

Every skill SKILL.md must include:

```yaml
---
name: <skill-name>
description: <WHAT it does>. Use when <trigger phrases>. Also: <more trigger phrases>.
---
```

### Naming

- `ai-` prefix for problem-first skills: named after the problem the user has — `ai-sorting`, `ai-parsing-data`. For users who think "I need to sort tickets."
- `dspy-` prefix for API/ecosystem skills: named after the DSPy concept or tool — `dspy-signatures`, `dspy-chain-of-thought`, `dspy-vizpy`. For users who already know DSPy.

### Description field

The description is scanned at session start to decide skill relevance. Write it as "when to trigger," not a summary.

**Structure**: `<WHAT it does>. Use when <trigger phrases the user would say>. Also: <more trigger phrases>.`

Good trigger phrases use words web/SaaS developers actually say:
- sort tickets — not "perform multi-class text classification"
- search docs — not "implement retrieval-augmented generation"
- AI gives wrong answers — not "improve model output fidelity"

`dspy-` skills should additionally include the DSPy API name (e.g., `dspy.ChainOfThought`, `BootstrapFewShot`).

**YAML quoting**: Use plain YAML scalars (no quotes) for descriptions. Do not use double quotes, single quotes, or escape sequences. Expand contractions to avoid apostrophes (write "do not" instead of "don't", "it is" instead of "it's"). Drop possessive apostrophes where the meaning stays clear ("the model weights" instead of "the model's weights"). If you avoid all quotes and special YAML characters (`:` at line start, `#` after space), plain scalars just work.

### Body pattern (progressive disclosure)

1. **Step 1 — Gather context**: Ask 2-4 clarifying questions before generating code
2. **Steps 2-4 — Core work**: Code examples, patterns, guidance — the meat of the skill
3. **Gotchas** (required): 3-5 DSPy/Claude-specific failure points (see below)
4. **Cross-references**: Point to related skills with `/skill-name`
5. **Additional resources**: Link to `examples.md`, `reference.md` if they exist

## Gotchas section (required)

The highest-signal content in any skill. Include 3-5 things Claude specifically gets wrong in this domain — not generic advice, but patterns observed from actual Claude behavior with DSPy.

Format each gotcha as:
- **Bold the failure pattern.** Then explain what Claude does wrong and what the correct behavior is.

Good gotchas are:
- Specific to Claude's defaults and biases (e.g., "Claude uses `Literal[list]` instead of `Literal[tuple(list)]`")
- Specific to DSPy API quirks (e.g., "Don't add `reasoning` to your signature — DSPy injects it automatically")
- Actionable (tell Claude what to do instead)
- Domain-specific (not generic like "ask clarifying questions")

Bad gotchas are:
- Generic advice ("make sure to be thorough")
- Things Claude already knows ("use proper grammar", "handle errors")
- User-facing tips disguised as Claude instructions
- Basic Python or DSPy concepts explained in the official docs

Update gotchas over time as you discover new failure patterns from real usage.

## Don't state the obvious

Focus skill content on information that pushes Claude out of its defaults. Don't repeat what Claude already knows about Python, coding patterns, or general ML concepts. If Claude would do the right thing without the instruction, leave it out.

Examples of what to skip:
- "Write clean, readable code" — Claude does this by default
- "Import the required modules" — Claude does this
- Standard Python patterns (try/except, list comprehensions, etc.)
- Generic programming advice ("test your code", "use version control")

Examples of what to include:
- DSPy API gotchas Claude wouldn't know (e.g., `with_inputs()` is required on Examples)
- Non-obvious patterns (e.g., categories > 15 degrade accuracy — use hierarchical classification)
- When to use one DSPy module vs another (ChainOfThought vs Predict tradeoffs)
- Optimizer selection guidance based on data size and quality requirements

## Avoid railroading

Give Claude the information it needs, but give it the flexibility to adapt to the user's situation. Rigid step-by-step scripts produce cookie-cutter outputs that don't fit the user's context.

**Instead of**: "You MUST use BootstrapFewShot with exactly 4 demos"
**Write**: "Start with BootstrapFewShot (typically 3-5 demos). Upgrade to MIPROv2 if accuracy plateaus."

**Instead of**: "Always use ChainOfThought"
**Write**: "Use ChainOfThought for tasks requiring multi-step reasoning. Use Predict for simple extraction or lookup tasks where reasoning adds cost without improving accuracy."

Use words like "typically," "adjust based on," "consider," and "depending on" instead of "always," "must," "exactly," and "never." The exception: actual DSPy API constraints (required method signatures, type restrictions) should remain definitive.

## Provider-agnostic code

All code examples must work with any LM provider DSPy supports. Use generic provider examples:

```python
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
```

Never hardcode a specific provider as the only option. Always include the `# or ...` comment showing alternatives.

## Skill file structure

```
skills/<skill-name>/
├── SKILL.md           # Main instructions (required, under 500 lines)
├── examples.md        # Worked examples and sample outputs
└── reference.md       # API details, field mappings, long reference material
```

Keep `SKILL.md` focused on instructions. Move anything over ~500 words of reference material to supporting files. Reference from SKILL.md with: "For worked examples, see [examples.md](examples.md)."

## Standalone completeness (required)

External users install individual skills via `npx skills add` — they get the skill directory only, **not** `docs/dspy-reference.md` or any other repo-level docs. Every skill must be fully self-contained.

### `reference.md` for `dspy-` skills

Every `dspy-` skill must have a `reference.md` containing condensed API docs for the DSPy classes it covers:

1. **Constructor signatures** with parameter tables (name, type, default, description)
2. **Key methods** with signatures and return types
3. **Upstream link** — start with `> Condensed from [dspy.ai/api/...](url). Verify against upstream for latest.`

`ai-` skills that reference specific DSPy APIs heavily should also include a `reference.md`.

### SKILL.md must work without `reference.md`

The SKILL.md itself should embed enough API context (constructor params, key usage patterns) that a user can write working code without reading `reference.md`. The reference file is a companion for deeper details, not the only place API docs live.

### No dangling references to repo docs

SKILL.md must not contain phrases like "see `docs/dspy-reference.md`" or "covered in the DSPy reference doc." All necessary information must live in the skill's own files.

## API documentation links (required for dspy- skills)

Every `dspy-` skill must link to the canonical API documentation for the DSPy concepts it covers. These go in the "Additional resources" section alongside `examples.md` and `reference.md` links.

**For `dspy-` skills covering DSPy core concepts**, link to the relevant `dspy.ai/api/` pages:

```markdown
## Additional resources

- [dspy.ChatAdapter API docs](https://dspy.ai/api/adapters/ChatAdapter/)
- For worked examples, see [examples.md](examples.md)
```

**For `dspy-` skills covering third-party integrations** (Langfuse, Phoenix, Weave, etc.), link to the integration's own documentation:

```markdown
## Additional resources

- [Langfuse DSPy integration docs](https://langfuse.com/docs/integrations/dspy)
- For worked examples, see [examples.md](examples.md)
```

**For `ai-` skills**, API doc links are optional but encouraged when the skill references specific DSPy modules or optimizers. Add them in the "Additional resources" section.

The URL pattern for DSPy core docs is `https://dspy.ai/api/<category>/<Name>/`:

| Category | URL pattern | Example |
|----------|------------|---------|
| Modules | `https://dspy.ai/api/modules/<Name>/` | `https://dspy.ai/api/modules/ChainOfThought/` |
| Optimizers | `https://dspy.ai/api/optimizers/<Name>/` | `https://dspy.ai/api/optimizers/MIPROv2/` |
| Adapters | `https://dspy.ai/api/adapters/<Name>/` | `https://dspy.ai/api/adapters/ChatAdapter/` |
| Evaluation | `https://dspy.ai/api/evaluation/<Name>/` | `https://dspy.ai/api/evaluation/Evaluate/` |
| Models | `https://dspy.ai/api/models/<Name>/` | `https://dspy.ai/api/models/LM/` |
| Signatures | `https://dspy.ai/api/signatures/` | |
| Primitives | `https://dspy.ai/api/primitives/` | |

## Cross-references between skills

Every skill should end with a cross-references section pointing users to related skills. Use the `/skill-name` format so users can invoke them directly.

### Install hint (required)

Skills are independently installable — users may not have other skills from this repo. The cross-references section must start with a blockquote install hint so users can install any referenced skill:

```markdown
## Cross-references

> Install any skill: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill <name>`

- **Predict** for simple calls without reasoning -- see `/dspy-predict`
- **Signatures** for defining input/output contracts -- see `/dspy-signatures`
```

For `ai-` skills, cross-reference both the related `ai-` skills and the underlying `dspy-` skills. For `dspy-` skills, link to related API-level skills and the problem-level `ai-` skills that use them.

### ai-do install back-link (required)

`ai-do` is the routing hub for the entire skill collection — the entry point for users who are not sure which skill they need. Every skill (except ai-do itself) must include this line as the **last entry** in its cross-references section:

```
- **Install `/ai-do` if you do not have it** — it routes any AI problem to the right skill and is the fastest way to work: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do`
```

This ensures users always have a path to install the router and makes `ai-do` the default workflow for future tasks.
