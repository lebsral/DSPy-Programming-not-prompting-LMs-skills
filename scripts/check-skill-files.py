#!/usr/bin/env python3
"""Enforce the standard three-file skill layout.

CLAUDE.md and docs/skill-standards.md describe each skill as a directory with
SKILL.md plus supporting examples.md and reference.md. In practice that was
aspirational — many skills shipped without one or both. This check makes the
standard real: every skill must have SKILL.md, examples.md, and reference.md,
unless it is on an explicit exemption list below.

Exemptions are principled, not "we didn't get to it":

  ai-do            Pure router. Its reference is catalog.md, not a DSPy API
                   surface; it teaches no code, so examples.md/reference.md
                   would be empty ceremony.
  ai-request-skill Meta/process skill (how to contribute a skill). No DSPy API
                   to document and no code pattern to exemplify.
  dspy-assertions  Legacy-only. dspy.Assert/Suggest were REMOVED in DSPy 3.x;
                   the skill exists to redirect, so it gets no fresh examples.

Run: python3 scripts/check-skill-files.py
Exit non-zero if any non-exempt skill is missing a required file.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO / "skills"

EXEMPT_EXAMPLES = {"ai-do", "ai-request-skill", "dspy-assertions"}
EXEMPT_REFERENCE = {"ai-do", "ai-request-skill"}


def skill_dirs():
    return sorted(
        p for p in SKILLS_DIR.iterdir()
        if p.is_dir() and (p / "SKILL.md").exists()
    )


def main():
    problems = []
    skills = skill_dirs()
    for d in skills:
        name = d.name
        if not (d / "examples.md").exists() and name not in EXEMPT_EXAMPLES:
            problems.append(f"{name}: missing examples.md")
        if not (d / "reference.md").exists() and name not in EXEMPT_REFERENCE:
            problems.append(f"{name}: missing reference.md")

    if problems:
        print(f"Skill files INCOMPLETE ({len(problems)} gaps across {len(skills)} skills):\n")
        for p in problems:
            print("  " + p)
        print("\nFix: add the file, or add the skill to an EXEMPT_* set with a "
              "one-line rationale if it genuinely needs no such file.")
        return 1

    print(f"Skill files OK — all {len(skills)} skills have SKILL.md + examples.md "
          f"+ reference.md (exemptions: examples {sorted(EXEMPT_EXAMPLES)}, "
          f"reference {sorted(EXEMPT_REFERENCE)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
