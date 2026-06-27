#!/usr/bin/env python3
"""Verify every skill in skills/ is registered in all of the places that must
stay in sync. Drift between these catalogs is a recurring bug, and it cuts both
ways:

  * Missing — a skill exists on disk but is unreachable through ai-do routing,
    undiscoverable in the README, or uninstallable via the plugin marketplace.
  * Dangling — ai-do advertises a skill that has NO directory on disk. The
    router then emits `npx skills add ... --skill <name>`, which fails because
    the name is unknown. Worse, a bundled `--skill a,b,c` install fails
    entirely if any one name is unknown, so a single dead reference can break
    the install of valid skills alongside it.

This script guards both directions.

Run from anywhere: `python3 scripts/check-skill-sync.py`
Exits non-zero (and prints what is wrong) if any catalog is out of sync.
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


def marketplace_groups():
    """{plugin_name: [skill_name, ...]} for each plugin group."""
    data = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    return {
        plugin["name"]: [e.rsplit("/", 1)[-1] for e in plugin.get("skills", [])]
        for plugin in data.get("plugins", [])
    }


def marketplace_skills():
    listed = set()
    for names in marketplace_groups().values():
        listed.update(names)
    return listed


def text_of(*relpath):
    return (REPO / Path(*relpath)).read_text()


# A skill-name token: ai-... or dspy-... (lowercase letters, digits, hyphens).
# Placeholders like /ai-<problem>, /dspy-*, or /skill-name do not match.
_NAME = r"(?:ai|dspy)-[a-z0-9]+(?:-[a-z0-9]+)*"


def referenced_skills(text):
    """Every skill name the text points at — as `/<name>` or inside a
    `--skill <a,b,c>` install bundle. Used to catch dangling references to
    skills that do not exist on disk."""
    found = set(re.findall(r"/(" + _NAME + r")\b", text))
    for bundle in re.findall(r"--skill\s+([a-z0-9,\-]+)", text):
        for tok in bundle.split(","):
            if re.fullmatch(_NAME, tok):
                found.add(tok)
    return found


def check(name, present, required, problems):
    """required: iterable of skill names that must appear in `present` (a set)."""
    missing = sorted(s for s in required if s not in present)
    if missing:
        problems.append(f"[{name}] missing {len(missing)}: " + ", ".join(missing))


def main():
    skills = skills_on_disk()
    real = set(skills)
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
    catalog_dangling = sorted(referenced_skills(catalog) - real)
    if catalog_dangling:
        problems.append(
            "[ai-do/catalog.md] references skills not on disk: " + ", ".join(catalog_dangling)
        )

    # 4. ai-do/SKILL.md — in-context routing tables. This is what Claude reads
    #    first, so a skill absent here is effectively unreachable for most asks.
    skill_md = text_of("skills", "ai-do", "SKILL.md")
    routing_present = {s for s in skills if re.search(rf"`?/{re.escape(s)}\b", skill_md)}
    routing_required = [
        s for s in skills if s != AI_DO and s not in ROUTING_EXCEPTIONS
    ]
    check("ai-do/SKILL.md routing", routing_present, routing_required, problems)
    routing_dangling = sorted(referenced_skills(skill_md) - real)
    if routing_dangling:
        problems.append(
            "[ai-do/SKILL.md] references skills not on disk: " + ", ".join(routing_dangling)
        )

    # 5. README.md advertised counts — the install copy quotes a total skill
    #    count and a per-plugin-group count. Both drift silently as skills are
    #    added. Assert them against reality (disk + marketplace group sizes).
    readme = text_of("README.md")
    m = re.search(r"Install all (\d+) skills", readme)
    if not m:
        problems.append("[README.md] missing the 'Install all N skills' line")
    elif int(m.group(1)) != len(skills):
        problems.append(
            f"[README.md] says 'Install all {m.group(1)} skills' but {len(skills)} on disk"
        )
    for group, names in marketplace_groups().items():
        gm = re.search(rf"{re.escape(group)}@dspy-skills.*?\((\d+) skills\)", readme)
        if not gm:
            problems.append(f"[README.md] plugin group {group} not listed with a count")
        elif int(gm.group(1)) != len(names):
            problems.append(
                f"[README.md] {group} says ({gm.group(1)} skills) but marketplace "
                f"has {len(names)}"
            )

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
          "marketplace.json, ai-do/catalog.md, and ai-do/SKILL.md routing; "
          "ai-do advertises no skill missing from disk; README counts match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
