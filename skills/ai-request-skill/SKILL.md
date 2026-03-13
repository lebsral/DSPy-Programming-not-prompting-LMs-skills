---
name: ai-request-skill
description: "Request or contribute a new AI skill that doesn't exist yet. Use when DSPy supports something but there's no skill for it — helps you build the skill and submit a PR, or file an issue requesting it."
argument-hint: "[describe the missing capability]"
---

# Request or Build a Missing Skill

The user needs a DSPy capability that doesn't have a skill yet. Help them contribute it or request it.

## Step 1: Confirm the gap

If `$ARGUMENTS` is provided, use it. Otherwise ask: "What DSPy capability do you need that isn't covered by an existing skill?"

Verify this is something DSPy actually supports. If it's outside DSPy's scope entirely (e.g., "build a React frontend"), say so and suggest appropriate tools instead. Do not proceed with a skill request for non-DSPy capabilities.

Summarize back to the user:
- **What they need**: one sentence
- **DSPy features involved**: which DSPy modules, integrations, or patterns are relevant

## Step 2: Choose a path

Ask the user:

> Would you like to:
> 1. **Build the skill** — I'll help you create it following the repo conventions and prepare a PR
> 2. **Request the skill** — I'll draft a GitHub issue so the maintainers know it's needed

## Path A: Build the skill

### Create the skill directory

Create `skills/ai-<problem-name>/SKILL.md` where `<problem-name>` describes the user's problem (not the DSPy module). Follow these conventions from this repo:

- **Problem-first naming**: Name after what the user is solving, with `ai-` prefix (e.g., `ai-observability` not `dspy-phoenix`)
- **Web developer language**: Use phrases developers actually say, not ML jargon
- **Provider-agnostic**: Don't hardcode specific LM providers

### Write the SKILL.md

Use this structure:

```yaml
---
name: ai-<problem-name>
description: "<What problem it solves>. Use when <trigger phrases the user would say>."
---
```

Body should follow the pattern used by other skills in this repo:
1. A step to gather requirements (ask the user 2-4 questions about their specific needs)
2. Implementation steps with code examples using DSPy
3. A verification/testing step
4. Next steps pointing to related skills (e.g., `/ai-improving-accuracy`, `/ai-serving-apis`)

Keep the SKILL.md under 500 lines. If more detail is needed, create a `reference.md` or `examples.md` alongside it.

Reference `docs/dspy-reference.md` for correct DSPy API usage (modules, optimizers, signatures).

### Prepare the PR

After creating the skill files:

1. Update `README.md` — add a row to the problem catalog table in the appropriate position
2. Create a branch named `add-ai-<problem-name>`
3. Commit with message: `Add ai-<problem-name> skill`
4. Open a PR to `lebsral/DSPy-Programming-not-prompting-LMs-skills` with:
   - **Title**: `Add ai-<problem-name> skill`
   - **Body**: What problem it solves, what DSPy features it uses, and an example invocation

## Path B: Request the skill

Draft a GitHub issue for `lebsral/DSPy-Programming-not-prompting-LMs-skills` with this format:

**Title**: `Skill request: ai-<problem-name>`

**Body**:

```markdown
## Problem
<What the user is trying to do, in their words>

## DSPy capability
<Which DSPy modules, integrations, or patterns would power this>

## Example use case
<A concrete scenario where this skill would help>

## Suggested trigger phrases
<2-3 phrases a developer might say that should route to this skill>
```

Use `gh issue create` to submit it, assigning it to the repo owner.
