# AI Planning Reference

## The Four-Phase Model

Every AI feature follows the same sequence. Skipping phases creates technical debt — optimizing before a baseline exists is guesswork; deploying before hardening creates production incidents.

| Phase | Objective | Done when |
|---|---|---|
| **Build** | A working baseline exists | Feature runs end-to-end on real inputs |
| **Measure** | Quality is quantified | Metric score and dev set are established |
| **Harden** | Reliability meets the bar | Target accuracy hit; edge cases covered |
| **Deploy** | Running in production | Serving, monitoring, and tracing in place |

---

## Skills by Phase

### Build

| Need | Skill |
|---|---|
| Scaffold the project structure | `/ai-kickoff` |
| Pick the right DSPy module or pattern | `/ai-choosing-architecture` |
| Classify, sort, label, or route | `/ai-sorting` |
| Extract structured data from text | `/ai-parsing-data` |
| Search documents or build RAG | `/ai-searching-docs` |
| Query a database in natural language | `/ai-querying-databases` |
| Condense long content into summaries | `/ai-summarizing` |
| Generate text, copy, or reports | `/ai-writing-content` |
| Call APIs and take autonomous actions | `/ai-taking-actions` |
| Multi-step reasoning | `/ai-reasoning` |
| Chain multiple AI steps into one pipeline | `/ai-building-pipelines` |
| Conversational AI with memory | `/ai-building-chatbots` |
| Coordinate multiple agents | `/ai-coordinating-agents` |

### Measure

| Need | Skill |
|---|---|
| Evaluate quality and run optimizers | `/ai-improving-accuracy` |
| Score or grade outputs against a rubric | `/ai-scoring` |
| Generate synthetic training or eval data | `/ai-generating-data` |
| Track and compare experiment runs | `/ai-tracking-experiments` |

### Harden

| Need | Skill |
|---|---|
| Validate outputs before users see them | `/ai-checking-outputs` |
| Prevent hallucination and fabricated facts | `/ai-stopping-hallucinations` |
| Enforce format, policy, or business rules | `/ai-following-rules` |
| Reduce output variance across runs | `/ai-making-consistent` |
| Moderate user-generated content | `/ai-moderating-content` |
| Red-team and adversarial testing | `/ai-testing-safety` |
| Push accuracy past prompting limits | `/ai-fine-tuning` |
| Fix broken AI features | `/ai-fixing-errors` |
| Audit AI code for correctness | `/ai-auditing-code` |

### Deploy

| Need | Skill |
|---|---|
| Serve AI behind a REST API | `/ai-serving-apis` |
| Monitor production quality and drift | `/ai-monitoring` |
| Trace and debug individual requests | `/ai-tracing-requests` |
| Cut API costs | `/ai-cutting-costs` |
| Switch LM providers without regressions | `/ai-switching-models` |

---

## Phase Gate Checklist

Use these as signals that a phase is done enough to advance. Treat them as guardrails, not rigid requirements.

**Build → Measure**
- Feature runs end-to-end on real (not toy) inputs
- Output format is stable enough to score
- You can describe what a correct output looks like

**Measure → Harden**
- Metric function returns a numeric score
- Dev set has 20+ examples (50+ preferred for optimizer runs)
- Baseline accuracy is written down

**Harden → Deploy**
- Accuracy meets the stated target threshold
- Adversarial or edge-case inputs are tested
- Output validation guards the user-facing path

**Deploy → stable**
- API is served and health-checked
- Errors surface in monitoring
- At least one trace viewer lets you inspect a live request

---

## Phase Dependencies

| Transition | What you need from the previous phase |
|---|---|
| Build → Measure | A running module and 20+ real examples |
| Measure → Harden | A metric function and a written baseline score |
| Harden → Deploy | Target accuracy met; critical failure modes handled |
| Deploy → stable | Serving confirmed; monitoring and tracing wired up |

Missing a dependency does not block progress — it creates debt. The earlier a dependency is skipped, the more it costs to recover later.
