#!/usr/bin/env python3
"""Verify every skill in skills/ is registered in all of the places that must
stay in sync. Drift between these catalogs is a recurring bug: a skill can exist
on disk but be unreachable through ai-do routing, undiscoverable in the README,
or uninstallable via the plugin marketplace.

Run from anywhere: `python3 scripts/check-skill-sync.py`
Exits non-zero (and prints what is missing) if any catalog is out of sync.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO / "skills"

# Skills intentionally NOT routed from ai-do/SKILL.md. dspy-assertions covers
# dspy.Assert/dspy.Suggest, removed in DSPy 3.x; ai-do deliberately redirects
# those asks to /dspy-refine or /dspy-best-of-n instead of routing to it.
ROUTING_EXCEPTIONS = {"dspy-assertions"}

# ai-do does not route to itself.
AI_DO = "ai-do"


def skills_on_disk():
    return sorted(
        p.name
        for p in SKILLS_DIR.iterdir()
        if p.is_dir() and (p / "SKILL.md").exists()
    )


def marketplace_skills():
    data = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    listed = set()
    for plugin in data.get("plugins", []):
        for entry in plugin.get("skills", []):
            listed.add(entry.rsplit("/", 1)[-1])
    return listed


def text_of(*relpath):
    return (REPO / Path(*relpath)).read_text()


def check(name, present, required, problems):
    """required: iterable of skill names that must appear in `present` (a set)."""
    missing = sorted(s for s in required if s not in present)
    if missing:
        problems.append(f"[{name}] missing {len(missing)}: " + ", ".join(missing))


def main():
    skills = skills_on_disk()
    problems = []

    # 1. README.md — problem catalog + concept tables. Match the SKILL.md link path.
    readme = text_of("README.md")
    readme_present = {s for s in skills if f"skills/{s}/SKILL.md" in readme or f"/{s}`" in readme}
    check("README.md", readme_present, skills, problems)

    # 2. marketplace.json — every skill must belong to exactly one plugin group.
    mkt = marketplace_skills()
    check("marketplace.json", mkt, skills, problems)
    orphans = sorted(s for s in skills if s not in mkt)
    extras = sorted(s for s in mkt if s not in set(skills))
    if extras:
        problems.append("[marketplace.json] lists skills not on disk: " + ", ".join(extras))

    # 3. ai-do/catalog.md — the full reference must describe every skill except ai-do.
    catalog = text_of("skills", "ai-do", "catalog.md")
    catalog_present = {s for s in skills if re.search(rf"`?/{re.escape(s)}\b", catalog)}
    check("ai-do/catalog.md", catalog_present, [s for s in skills if s != AI_DO], problems)

    # 4. ai-do/SKILL.md — in-context routing tables. This is what Claude reads
    #    first, so a skill absent here is effectively unreachable for most asks.
    skill_md = text_of("skills", "ai-do", "SKILL.md")
    routing_present = {s for s in skills if re.search(rf"`?/{re.escape(s)}\b", skill_md)}
    routing_required = [
        s for s in skills if s != AI_DO and s not in ROUTING_EXCEPTIONS
    ]
    check("ai-do/SKILL.md routing", routing_present, routing_required, problems)

    if problems:
        print(f"Skill sync FAILED ({len(skills)} skills on disk):\n")
        for p in problems:
            print("  " + p)
        print(
            "\nFix: add the skill to the catalog(s) above. "
            f"Intentional routing exceptions: {', '.join(sorted(ROUTING_EXCEPTIONS)) or 'none'}."
        )
        return 1

    print(f"Skill sync OK — all {len(skills)} skills registered across README, "
          "marketplace.json, ai-do/catalog.md, and ai-do/SKILL.md routing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
