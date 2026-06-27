#!/usr/bin/env python3
"""Run the eval suites that ship with each skill.

Every skill may carry `skills/<name>/evals/evals.json`: a set of realistic
prompts, an expected_output sketch, and named assertions describing what a
correct response must contain. Until now nothing executed them. This runner
turns that latent data into an actual test suite.

Two modes:

  --validate (default)   Pure stdlib. Validate every evals.json against the
                         schema and print a coverage report. No API key, no
                         network — safe to run in CI on every PR.

  --grade                Opt-in. For each eval, run the prompt through the
                         skill (its SKILL.md body becomes the model's
                         instructions) and grade the response against each
                         assertion with an LLM judge. Requires DSPy + a
                         configured model (so it needs an API key and costs
                         money). Lazily imports dspy so --validate never pays
                         that cost.

Usage:
  python3 scripts/run_evals.py                 # validate all skills
  python3 scripts/run_evals.py --validate      # same, explicit
  python3 scripts/run_evals.py --skill ai-sorting --grade
  python3 scripts/run_evals.py --grade --model openai/gpt-4o-mini

Exit codes: 0 ok, 1 schema/grading failure, 2 environment problem.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO / "skills"


# ---------------------------------------------------------------------------
# Discovery + schema validation (stdlib only)
# ---------------------------------------------------------------------------

def skill_dirs():
    return sorted(
        p for p in SKILLS_DIR.iterdir()
        if p.is_dir() and (p / "SKILL.md").exists()
    )


def load_evals(skill_dir):
    """Return (cases, error). cases is None when the file is missing or
    unparseable; otherwise it is the normalized list (see normalize)."""
    path = skill_dir / "evals" / "evals.json"
    if not path.exists():
        return None, None  # missing is reported separately from malformed
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    return data, None


def normalize(data):
    """Collapse the several eval formats this repo uses into one shape.

    Accepted on disk:
      wrapped:    {"skill_name": str, "evals": [ {id, prompt, expected_output,
                  files, assertions: [{name, description}, ...]} ]}
      bare:       [ {prompt, expected_output, assertions: ["text", ...]} ]
      contains:   [ {id, prompt, should_contain: [...], should_not_contain: [...]} ]
      io:         [ {id, description, input: {...}, expected: {...}} ]

    Returns (cases, problems). Each case is one of:
      prompt-style: {id, kind:"prompt", prompt, expected_output,
                     assertions: [{name, description}, ...]}
      io-style:     {id, kind:"io", input, expected}   # not prompt-graded

    expected_output is informational and optional; an assertion (or a
    should_contain / should_not_contain entry) is what makes a prompt case
    gradeable.
    """
    if isinstance(data, dict):
        raw = data.get("evals")
        if not isinstance(raw, list):
            return [], ["object form is missing an 'evals' list"]
    elif isinstance(data, list):
        raw = data
    else:
        return [], ["top level must be an object or a list"]
    if not raw:
        return [], ["no eval cases"]

    cases, problems, seen = [], [], set()
    for i, ev in enumerate(raw):
        tag = f"eval[{i}]"
        if not isinstance(ev, dict):
            problems.append(f"{tag} is not an object")
            continue
        cid = ev.get("id", i)
        if cid in seen:
            problems.append(f"{tag} duplicate id {cid}")
        seen.add(cid)

        # I/O data-eval (functional stimulus -> expected), e.g. data-cleaning /
        # rewriting skills. Keys vary: input|inputs and expected|evaluation.
        io_stim = ev.get("input", ev.get("inputs"))
        io_exp = ev.get("expected", ev.get("evaluation"))
        if io_stim is not None and io_exp is not None and not ev.get("prompt"):
            cases.append({"id": cid, "kind": "io", "input": io_stim, "expected": io_exp})
            continue

        # Otherwise a prompt-style case. Gather grading signals across the
        # several key names used in this repo's eval files.
        norm = []
        for a in ev.get("assertions") or []:
            if isinstance(a, str) and a.strip():
                norm.append({"name": a.strip(), "description": a.strip()})
            elif isinstance(a, dict) and (a.get("description") or "").strip():
                norm.append({"name": a.get("name") or a["description"],
                             "description": a["description"]})
            else:
                problems.append(f"{tag} has an empty/invalid assertion")
        for key in ("should_contain", "expected_patterns"):
            for s in ev.get(key) or []:
                if str(s).strip():
                    norm.append({"name": f"contains:{s}",
                                 "description": f"The response covers or includes: {s}"})
        for key in ("should_not_contain", "must_not_contain"):
            for s in ev.get(key) or []:
                if str(s).strip():
                    norm.append({"name": f"excludes:{s}",
                                 "description": f"The response avoids: {s}"})

        has_prompt = bool((ev.get("prompt") or "").strip())
        if not has_prompt and not norm:
            problems.append(f"{tag} has no prompt/input and no checkable signal")
        elif not has_prompt:
            problems.append(f"{tag} has assertions but no 'prompt' (and is not an I/O case)")
        elif not norm and not (ev.get("expected_output") or "").strip():
            problems.append(f"{tag} has a prompt but no assertion / expected_output")

        cases.append({"id": cid, "kind": "prompt", "prompt": ev.get("prompt", ""),
                      "expected_output": ev.get("expected_output", ""),
                      "assertions": norm})
    return cases, problems


def cmd_validate(skills):
    have, missing, malformed = [], [], []
    total_cases = total_prompt = total_io = total_assertions = 0

    for d in skills:
        data, err = load_evals(d)
        if data is None and err is None:
            missing.append(d.name)
            continue
        if err:
            malformed.append((d.name, err))
            continue
        cases, problems = normalize(data)
        if problems:
            malformed.append((d.name, "; ".join(problems)))
            continue
        have.append(d.name)
        total_cases += len(cases)
        total_prompt += sum(1 for c in cases if c["kind"] == "prompt")
        total_io += sum(1 for c in cases if c["kind"] == "io")
        total_assertions += sum(len(c.get("assertions", [])) for c in cases)

    n = len(skills)
    print(f"Eval coverage: {len(have)}/{n} skills have a valid evals.json "
          f"({total_cases} cases - {total_prompt} prompt-graded, {total_io} I/O - "
          f"{total_assertions} assertions)")
    if missing:
        print(f"\n  No evals.json ({len(missing)}): " + ", ".join(sorted(missing)))
    if malformed:
        print(f"\n  MALFORMED ({len(malformed)}):")
        for name, why in sorted(malformed):
            print(f"    {name}: {why}")
        return 1
    print("\nAll present eval suites are schema-valid.")
    return 0


# ---------------------------------------------------------------------------
# Graded mode (opt-in; needs dspy + a model)
# ---------------------------------------------------------------------------

def skill_instructions(skill_dir):
    """The SKILL.md body with YAML frontmatter stripped."""
    text = (skill_dir / "SKILL.md").read_text()
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    return text.strip()


def cmd_grade(skills, model):
    try:
        import dspy
    except ImportError:
        print("--grade needs DSPy. Install it: pip install dspy", file=sys.stderr)
        return 2

    try:
        dspy.configure(lm=dspy.LM(model))
    except Exception as e:  # noqa: BLE001 - surface any config/auth error clearly
        print(f"Could not configure model {model!r}: {e}", file=sys.stderr)
        print("Set the provider API key (e.g. OPENAI_API_KEY) and pass --model.",
              file=sys.stderr)
        return 2

    class Solve(dspy.Signature):
        """Follow the skill instructions to answer the user's request."""
        skill_instructions: str = dspy.InputField()
        request: str = dspy.InputField()
        response: str = dspy.OutputField()

    class Judge(dspy.Signature):
        """Decide whether the response satisfies the assertion. Be strict."""
        request: str = dspy.InputField()
        response: str = dspy.InputField()
        assertion: str = dspy.InputField(desc="what the response must satisfy")
        passes: bool = dspy.OutputField()
        reason: str = dspy.OutputField()

    solver = dspy.Predict(Solve)
    judge = dspy.Predict(Judge)

    results = []
    grand_pass = grand_total = 0
    for d in skills:
        data, err = load_evals(d)
        if data is None or err:
            continue
        cases, problems = normalize(data)
        if problems:
            print(f"  {d.name:34} skipped (malformed evals.json)")
            continue
        instructions = skill_instructions(d)
        skill_pass = skill_total = 0
        for ev in cases:
            if ev["kind"] != "prompt":
                continue  # I/O data-evals are not LLM-graded here
            answer = solver(skill_instructions=instructions, request=ev["prompt"]).response
            checks = []
            for a in ev["assertions"]:
                v = judge(request=ev["prompt"], response=answer, assertion=a["description"])
                ok = bool(v.passes)
                checks.append({"assertion": a["name"], "passes": ok, "reason": v.reason})
                skill_total += 1
                skill_pass += int(ok)
            results.append({"skill": d.name, "eval_id": ev["id"], "checks": checks})
        grand_pass += skill_pass
        grand_total += skill_total
        if skill_total:
            print(f"  {d.name:34} {skill_pass}/{skill_total} assertions passed")

    out = REPO / "eval-results.json"
    out.write_text(json.dumps(results, indent=2))
    pct = (100 * grand_pass / grand_total) if grand_total else 0
    print(f"\nGraded {grand_total} assertions across {len(skills)} skills: "
          f"{grand_pass} passed ({pct:.0f}%). Details -> {out.relative_to(REPO)}")
    # Grading is a quality signal, not a hard gate; never fail the process on a
    # low score (a flaky model run should not block a PR). Only env errors do.
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grade", action="store_true",
                    help="run + grade with an LLM judge (needs dspy + API key)")
    ap.add_argument("--validate", action="store_true",
                    help="schema + coverage only (default)")
    ap.add_argument("--skill", help="restrict to one skill by name")
    ap.add_argument("--model", default="openai/gpt-4o-mini",
                    help="LM for --grade (default openai/gpt-4o-mini)")
    args = ap.parse_args()

    skills = skill_dirs()
    if args.skill:
        skills = [d for d in skills if d.name == args.skill]
        if not skills:
            print(f"No such skill: {args.skill}", file=sys.stderr)
            return 2

    if args.grade:
        return cmd_grade(skills, args.model)
    return cmd_validate(skills)


if __name__ == "__main__":
    sys.exit(main())
