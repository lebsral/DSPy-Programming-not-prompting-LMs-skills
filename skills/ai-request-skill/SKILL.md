---
name: ai-request-skill
description: Request or contribute a new AI skill that doesn't exist yet. Use when DSPy supports something but there's no skill for it — helps you build the skill and submit a PR, or file an issue requesting it. Also use when the user says 'there should be a skill for this', 'can we make a skill', 'I want to contribute a skill', or 'none of the ai- skills cover my use case'.
argument-hint: "[describe the missing capability]"
---

# Request or Build a Missing Skill

The user needs a DSPy capability that doesn't have a skill yet. Help them contribute it or request it.

**This skill always ends with a concrete action on GitHub:**
- **Path A (Build)**: Create the skill files, commit, push, and open a **pull request** to `lebsral/DSPy-Programming-not-prompting-LMs-skills`
- **Path B (Request)**: File a **GitHub issue** on `lebsral/DSPy-Programming-not-prompting-LMs-skills` describing what's needed

Do not stop at "here's what the PR/issue would look like" — actually create it using `gh pr create` or `gh issue create`.

## Step 1: Confirm the gap

If `$ARGUMENTS` is provided, use it. Otherwise ask: "What DSPy capability do you need that isn't covered by an existing skill?"

Verify this is something DSPy actually supports. If it's outside DSPy's scope entirely (e.g., "build a React frontend"), say so and suggest appropriate tools instead.

Check the existing skills in `skills/` or [ai-*](https://github.com/lebsral/dspy-programming-not-prompting-lms-skills) to make sure there isn't already a skill that covers this. If there's a close match, suggest it instead.

Summarize back to the user:
- **What they need**: one sentence
- **DSPy features involved**: which DSPy modules, integrations, or patterns are relevant
- **Closest existing skill**: what's close but doesn't quite fit

## Step 2: Choose a path

Ask the user:

> Would you like to:
> 1. **Build the skill** — I'll help you create it with proper testing and prepare a PR
> 2. **Request the skill** — I'll draft a GitHub issue so the maintainers know it's needed

## Path A: Build the skill

### Use skill-creator if available

Check whether the `/skill-creator` skill is available (it's from the [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/skill-creator) repo). If available, delegate to it — it handles the full create-test-iterate workflow including evaluation, benchmarking, and description optimization.

When delegating to `/skill-creator`, provide this DSPy-specific context:

> **Repo conventions for this skill:**
> - Dual naming: `ai-<problem>` prefix for problem-first skills (e.g., `ai-observability`), `dspy-<concept>` prefix for API-first skills (e.g., `dspy-signatures`)
> - Web developer language in descriptions — use phrases developers actually say ("monitor AI quality", "search docs"), not ML jargon
> - Provider-agnostic: don't hardcode specific LM providers, use `dspy.LM("openai/gpt-4o-mini")` as default examples
> - SKILL.md under 500 lines; overflow goes to `examples.md` or `reference.md`
> - Reference `docs/dspy-reference.md` for correct DSPy API usage
> - Test prompts should be realistic developer requests, not abstract ML tasks
>
> **DSPy patterns to include (based on what the skill covers):**
> - Data loading: CSV, JSON, transcripts (VTT, LiveKit, Recall), Langfuse traces
> - Signatures: class-based with typed fields, Pydantic models for structured output
> - Modules: dspy.Predict, ChainOfThought, ReAct — with guidance on when to use each
> - Validation: dspy.Assert (hard) vs dspy.Suggest (soft) integrated in forward()
> - Evaluation: dspy.Evaluate with custom metrics
> - Optimization: BootstrapFewShot for quick start, MIPROv2 for production
> - Save/load: `program.save()` / `program.load()` for deployment
>
> **Skill structure:**
> ```
> skills/ai-<problem>/
> ├── SKILL.md        # Main instructions (required)
> ├── examples.md     # 2-3 worked examples (recommended)
> └── reference.md    # Deep reference material (if needed)
> ```
>
> **SKILL.md body pattern** (follow what other skills in this repo do):
> 1. Step to gather requirements (ask 2-4 questions about the user's specific needs)
> 2. Implementation steps with DSPy code examples
> 3. Data loading patterns relevant to the problem domain
> 4. Evaluation and optimization step
> 5. Save/load and deployment
> 6. Next steps pointing to related skills (e.g., `/ai-improving-accuracy`, `/ai-serving-apis`)

Then let skill-creator run its workflow: draft → test → review → iterate → package.

### If skill-creator is NOT available

Build the skill manually following the conventions above. Install skill-creator for future use:

```bash
npx skills add anthropics/skills/skill-creator
```

Or from GitHub:
```bash
# Clone and copy
git clone https://github.com/anthropics/skills.git /tmp/anthropic-skills
cp -r /tmp/anthropic-skills/skills/skill-creator ~/.claude/skills/
```

#### Write the SKILL.md

```yaml
---
name: ai-<problem-name>
description: "<What problem it solves>. Use when <trigger phrases the user would say>."
---
```

Read 2-3 existing skills in `skills/` to match the tone and structure. Key things to get right:

**Description field** — This is how Claude decides whether to use the skill. Be specific about trigger phrases. Include both what the skill does AND when to use it. Err on the side of being "pushy" — Claude tends to under-trigger skills, so include edge cases:

```yaml
# Bad: too vague
description: "Help with AI observability"

# Good: specific triggers, covers edge cases
description: "Monitor your AI's quality, latency, and cost in production. Use when you need to track AI accuracy over time, detect model degradation, set up alerts for quality drops, log predictions, measure production performance, or answer 'is our AI still working well?'"
```

**Code examples** — Every code block should be copy-pasteable. Include imports, LM configuration, and a usage example at the bottom. Use `docs/dspy-reference.md` for correct API patterns.

**examples.md** — Create 2-3 realistic end-to-end examples. Each should show a complete working program, not just fragments. Include data loading, the DSPy module, evaluation, and a usage section.

#### Test the skill

Create 2-3 test prompts — realistic requests a developer would make. Run them with the skill active and verify the outputs make sense. If you have access to subagents, run with-skill and without-skill baselines to measure impact.

### Submit the PR

After creating (and optionally testing) the skill files, submit a pull request. Do all of these steps — don't stop at "here's what to do":

1. Update `README.md` — add a row to the problem catalog table in the appropriate position
2. Create a branch: `git checkout -b add-ai-<problem-name>`
3. Stage and commit: `git add skills/ai-<problem-name>/ README.md && git commit -m "Add ai-<problem-name> skill"`
4. Push: `git push -u origin add-ai-<problem-name>`
5. Open the PR:

```bash
gh pr create \
  --repo lebsral/DSPy-Programming-not-prompting-LMs-skills \
  --title "Add ai-<problem-name> skill" \
  --body "$(cat <<'EOF'
## Summary
- **Problem**: <what the user is solving>
- **DSPy features**: <modules, patterns used>
- **Example invocation**: `/ai-<problem-name> <example prompt>`

## Files
- `skills/ai-<problem-name>/SKILL.md` — main instructions
- `skills/ai-<problem-name>/examples.md` — worked examples
- `README.md` — catalog table updated
EOF
)"
```

Return the PR URL to the user when done.

## Path B: Request the skill

File a GitHub issue on the repo. Do not just draft it — actually submit it:

```bash
gh issue create \
  --repo lebsral/DSPy-Programming-not-prompting-LMs-skills \
  --title "Skill request: ai-<problem-name>" \
  --assignee lebsral \
  --body "$(cat <<'EOF'
## Problem
<What the user is trying to do, in their words>

## DSPy capability
<Which DSPy modules, integrations, or patterns would power this>

## Example use case
<A concrete scenario where this skill would help>

## Suggested trigger phrases
<2-3 phrases a developer might say that should route to this skill>
EOF
)"
```

Return the issue URL to the user when done.

## Quality checklist

Before submitting a new skill (via PR or skill-creator), verify:

- [ ] Name follows `ai-<problem>` convention (problem-first, not DSPy-concept-first)
- [ ] Description includes specific trigger phrases a developer would actually say
- [ ] SKILL.md is under 500 lines
- [ ] Code examples are provider-agnostic (no hardcoded API keys or specific providers)
- [ ] Code examples include imports, LM config, and are copy-pasteable
- [ ] Includes data loading patterns relevant to the problem domain
- [ ] Has evaluation/optimization step (not just "build it and hope")
- [ ] Points to related skills in "next steps" (e.g., `/ai-improving-accuracy`)
- [ ] README.md catalog table updated with new row
- [ ] ai-do https://github.com/lebsral/DSPy-Programming-not-prompting-LMs-skills/blob/main/skills/ai-do/SKILL.md updated with new row in PR or gh issue
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
