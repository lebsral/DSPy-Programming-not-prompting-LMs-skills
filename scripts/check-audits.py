#!/usr/bin/env python3
"""Report audit coverage and freshness for every skill.

Each skill carries `skills/<name>/audit.yaml` recording the last quality audit:

    last_audit:
      date: 2026-05-04
      score: 43/43
      versions:
        dspy: 3.2.1

A skill that has never been audited carries an explicit pending marker
(`last_audit: null`) instead, so the gap is visible rather than silent. This
script classifies every skill as:

    ok       audited within the freshness window
    stale    audited, but longer ago than --max-age-days
    pending  audit.yaml present but never audited (last_audit: null)
    missing  no audit.yaml at all (structural problem)

Structural problems (missing / unparseable) fail the run. Staleness and pending
status are informational by default (so CI does not turn red as weeks pass);
pass --strict to fail on those too.

  python3 scripts/check-audits.py                 # report
  python3 scripts/check-audits.py --strict        # also fail on stale/pending
  python3 scripts/check-audits.py --init-missing  # write pending stubs

Stdlib only — no yaml dependency, safe for CI.
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO / "skills"

PENDING_STUB = """\
# No formal audit has been recorded for this skill yet. Run the skill-quality
# audit against the current DSPy version, then replace last_audit with a real
# block: {date: YYYY-MM-DD, score: N/N, versions: {dspy: X.Y.Z}}.
last_audit: null
"""


def skill_dirs():
    return sorted(
        p for p in SKILLS_DIR.iterdir()
        if p.is_dir() and (p / "SKILL.md").exists()
    )


def classify(path, today, max_age_days):
    """Return (status, detail). status in ok|stale|pending|missing|malformed."""
    if not path.exists():
        return "missing", "no audit.yaml"
    text = path.read_text()
    # Pending marker: last_audit explicitly null, or no date recorded anywhere.
    if re.search(r"^\s*last_audit:\s*null\s*$", text, re.MULTILINE):
        return "pending", "never audited"
    m = re.search(r"date:\s*(\d{4})-(\d{2})-(\d{2})", text)
    if not m:
        return "malformed", "no 'date:' and not marked pending"
    audited = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    age = (today - audited).days
    ver = re.search(r"dspy:\s*([0-9][\w.\-]*)", text)
    vtxt = f", dspy {ver.group(1)}" if ver else ""
    if age > max_age_days:
        return "stale", f"audited {audited} ({age}d ago{vtxt})"
    return "ok", f"audited {audited} ({age}d ago{vtxt})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-age-days", type=int, default=120,
                    help="audits older than this are 'stale' (default 120)")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on stale/pending, not just structural problems")
    ap.add_argument("--init-missing", action="store_true",
                    help="write a pending stub for any skill lacking audit.yaml")
    args = ap.parse_args()

    today = date.today()
    skills = skill_dirs()

    if args.init_missing:
        created = 0
        for d in skills:
            p = d / "audit.yaml"
            if not p.exists():
                p.write_text(PENDING_STUB)
                created += 1
        print(f"Wrote {created} pending audit.yaml stub(s).")
        return 0

    buckets = {"ok": [], "stale": [], "pending": [], "missing": [], "malformed": []}
    for d in skills:
        status, detail = classify(d / "audit.yaml", today, args.max_age_days)
        buckets[status].append((d.name, detail))

    n = len(skills)
    print(f"Audit coverage ({n} skills): "
          f"{len(buckets['ok'])} ok, {len(buckets['stale'])} stale, "
          f"{len(buckets['pending'])} pending, {len(buckets['missing'])} missing, "
          f"{len(buckets['malformed'])} malformed "
          f"(freshness window {args.max_age_days}d)")

    for label in ("missing", "malformed", "pending", "stale"):
        rows = buckets[label]
        if rows:
            print(f"\n  {label.upper()} ({len(rows)}):")
            for name, detail in sorted(rows):
                print(f"    {name:34} {detail}")

    structural = buckets["missing"] + buckets["malformed"]
    if structural:
        print("\nFix: run `python3 scripts/check-audits.py --init-missing` to "
              "create pending stubs, or repair the malformed file(s).")
        return 1
    if args.strict and (buckets["stale"] or buckets["pending"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
