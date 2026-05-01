# ai-planning: Worked Examples

Four complete planning walkthroughs for common AI project types.

---

## Example 1: Support Ticket Triage System

**Context:** A SaaS company has 10,000 support tickets in Postgres. Agents currently spend 20% of their time reading and manually routing tickets to the right team. They want automatic routing plus a one-line summary for each ticket when it arrives.

### Step 1: The 3 Questions

**End goal:** Incoming tickets are automatically routed to the correct team (billing, technical, account) and agents see a one-sentence summary at the top of each ticket. Routing accuracy target: 90%+.

**What exists today:** 10,000 historical tickets in Postgres, each with a `team` label applied by agents. No existing AI code.

**Hard constraints:** Must use OpenAI-compatible API (company policy). No PII can leave the internal network for the summary step (ticket bodies contain customer data) — use a self-hosted model for summarization.

### Step 2: Capability Mapping

| Capability needed | Skill |
|---|---|
| Route tickets to teams | `/ai-sorting` |
| Summarize ticket body | `/ai-summarizing` |
| Connect both steps | `/ai-building-pipelines` |
| Measure and improve accuracy | `/ai-improving-accuracy` |

### Phase Plan

**Phase 1: Classify and Route (start here)**
- Goal: Tickets are automatically assigned to the correct team with 90%+ accuracy
- Skill: `/ai-sorting` — the 10k labeled tickets are a ready-made training set; use them as few-shot examples or compile a classifier
- Deliverable: A DSPy classifier that accepts ticket text and returns a team label, evaluated against a held-out test set

**Phase 2: Summarize for Agents**
- Goal: Each routed ticket has a one-sentence summary agents can read before opening it
- Skill: `/ai-summarizing` — use the self-hosted model here to satisfy the PII constraint
- Deliverable: A summarizer that runs on each incoming ticket and stores the summary in a new Postgres column

**Phase 3: Pipeline and Production**
- Goal: Both steps run automatically on every new ticket via a webhook
- Skill: `/ai-building-pipelines` to chain classify + summarize; `/ai-serving-apis` to expose as an internal endpoint
- Deliverable: Deployed API that accepts a ticket ID and returns route + summary

**Phase 4: Optimize**
- Goal: Push routing accuracy from baseline toward 95%+
- Skill: `/ai-improving-accuracy` — compile the classifier using `BootstrapFewShot` or `MIPROv2` against the labeled dataset
- Deliverable: Compiled program with measurable accuracy improvement over Phase 1 baseline

### Dependencies
- Phase 2 needs Phase 1 output because the summary is displayed alongside the route — both fields are written together
- Phase 3 needs both Phase 1 and Phase 2 working independently before connecting them
- Phase 4 needs the Phase 1 test set and baseline accuracy number to measure against

### What to Skip for Now
- Agent-facing chatbot — not requested, adds scope
- Automatic ticket resolution — too risky without human review, Phase 5+ if ever
- Monitoring / tracing — add after the pipeline is stable in production

---

## Example 2: Internal Knowledge Base Q&A

**Context:** A company has 500 help articles in Confluence. The support team wants a chatbot that answers internal questions with a citation to the source article, so agents stop asking each other the same questions repeatedly.

### Step 1: The 3 Questions

**End goal:** An agent types a question in Slack and gets a direct answer plus a link to the article it came from. Answers should not be hallucinated — if the answer is not in the knowledge base, the bot should say so.

**What exists today:** 500 Confluence articles exported as Markdown files. No existing AI code. No labeled question-answer pairs.

**Hard constraints:** 6-week timeline to first demo. Must cite sources. Must not hallucinate.

### Step 2: Capability Mapping

| Capability needed | Skill |
|---|---|
| Search and answer from articles | `/ai-searching-docs` |
| Prevent hallucinations | `/ai-stopping-hallucinations` |
| Improve answer quality over time | `/ai-improving-accuracy` |
| Expose as Slack integration | `/ai-serving-apis` |

### Phase Plan

**Phase 1: Search and Answer with Citations (start here)**
- Goal: Given a question, retrieve the most relevant articles and generate a grounded answer with a link
- Skill: `/ai-searching-docs` — index the 500 articles, build a RAG pipeline that returns answer + source URL
- Deliverable: A working Q&A script; team members can test it manually. No Slack integration yet.

**Phase 2: Add Hallucination Guardrails**
- Goal: The bot refuses to answer when the knowledge base does not contain relevant information, rather than making something up
- Skill: `/ai-stopping-hallucinations` — add a faithfulness check that verifies the answer is grounded in the retrieved context
- Deliverable: Updated pipeline with a confidence threshold; low-confidence answers return "I don't know, here are the closest articles"

**Phase 3: Evaluate and Improve**
- Goal: Measure answer quality against a test set of 50 real questions the support team wrote down
- Skill: `/ai-improving-accuracy` — create a DSPy metric for answer correctness + citation presence, run optimizer
- Deliverable: Compiled program with documented accuracy on the 50-question test set

**Phase 4: Slack Integration**
- Goal: Agents can ask questions directly in Slack without leaving their workflow
- Skill: `/ai-serving-apis` — wrap the compiled pipeline in a Slack bot endpoint
- Deliverable: Deployed Slack integration, announced to the support team

### Dependencies
- Phase 2 requires Phase 1 to be working because guardrails wrap the existing RAG pipeline
- Phase 3 requires the support team to write 50 test questions — coordinate this during Phase 2
- Phase 4 requires Phase 3 to be done so the Slack integration ships with the optimized, evaluated version

### What to Skip for Now
- Conversation memory / multi-turn chat — single-turn Q&A covers 90% of the use case
- Automatic article ingestion from Confluence API — start with the static export, automate later
- Per-user personalization — not needed for an internal tool at this scale

---

## Example 3: Invoice Processing Pipeline

**Context:** A finance team receives 200+ PDF invoices per week by email. They currently have two people manually entering vendor name, invoice number, date, line items, and total into their ERP. They want this automated.

### Step 1: The 3 Questions

**End goal:** PDF invoices arrive by email, fields are extracted automatically, validated against business rules (totals match line items, vendor exists in the system), and pushed into the ERP. Human review only for exceptions.

**What exists today:** 3 months of historical invoices as PDFs (about 2,400 files). No labeled extraction data. An ERP with a REST API for creating invoice records.

**Hard constraints:** Extraction accuracy must be high enough that humans only review flagged exceptions, not every invoice. No invoice can be submitted to the ERP with a mismatched total.

### Step 2: Capability Mapping

| Capability needed | Skill |
|---|---|
| Extract fields from PDFs | `/ai-parsing-data` |
| Validate extracted fields | `/ai-checking-outputs` |
| Connect extract + validate + submit | `/ai-building-pipelines` |
| Measure and improve extraction accuracy | `/ai-improving-accuracy` |

### Phase Plan

**Phase 1: Extract Invoice Fields (start here)**
- Goal: Given a PDF invoice, extract vendor name, invoice number, date, line items, and total as structured JSON
- Skill: `/ai-parsing-data` — define a DSPy `TypedPredictor` with a Pydantic model for the invoice schema; test on 20 real PDFs manually
- Deliverable: An extraction script that processes a PDF and outputs structured JSON. Accuracy measured by hand on 20 samples.

**Phase 2: Validate Before Submission**
- Goal: Catch mismatches (line items do not sum to total, vendor not in ERP, duplicate invoice number) before they reach the ERP
- Skill: `/ai-checking-outputs` — add assertion-based validation using `dspy.Assert` for math checks and ERP API lookups for vendor/duplicate checks
- Deliverable: Pipeline that extracts, validates, and flags exceptions for human review with a reason

**Phase 3: Batch Processing and ERP Integration**
- Goal: Process all incoming invoices automatically; push clean ones to the ERP, queue exceptions for review
- Skill: `/ai-building-pipelines` — orchestrate extract + validate + route (submit or queue); `/ai-serving-apis` for the email webhook
- Deliverable: Automated pipeline triggered by incoming email; finance team reviews only flagged exceptions

**Phase 4: Improve Extraction Accuracy**
- Goal: Label 100 invoices as a gold test set and push extraction F1 above the agreed threshold
- Skill: `/ai-improving-accuracy` — compile the extractor using labeled examples; track field-level accuracy
- Deliverable: Compiled extraction program with documented per-field accuracy

### Dependencies
- Phase 2 wraps Phase 1 output — extraction must be working before validation can be layered on
- Phase 3 requires Phase 2 because submitting unvalidated data to the ERP is the explicit non-goal
- Phase 4 requires a labeled test set — finance team needs to label 100 invoices during Phase 2 or 3

### What to Skip for Now
- Auto-approval without human review — start with human-in-the-loop for exceptions, remove later
- Learning from corrections — save corrected invoices to retrain, but do not build the feedback loop in Phase 1
- Support for non-PDF formats (Excel, EDI) — add after PDF is solid

---

## Example 4: AI Content Generation Platform

**Context:** A marketing team generates product descriptions for 5,000 SKUs per quarter. Each description must match brand voice, include key features, and be SEO-optimized. Currently written by contractors at $8/description — they want to reduce this cost by 80% while maintaining quality.

### Step 1: The 3 Questions

**End goal:** A marketer uploads a product spec (name, features, category, target audience) and gets a publication-ready product description in the brand voice within 30 seconds. Descriptions that pass quality review go live without edits.

**What exists today:** 5,000 approved product descriptions from the past two years. Brand voice guidelines document. No existing AI code.

**Hard constraints:** Output must match brand voice — the team will reject anything that sounds generic. 80% cost reduction target. Marketer must be able to trigger generation from an existing internal tool (has a webhook interface).

### Step 2: Capability Mapping

| Capability needed | Skill |
|---|---|
| Generate on-brand descriptions | `/ai-writing-content` |
| Score quality before publishing | `/ai-scoring` |
| Optimize for brand voice | `/ai-improving-accuracy` |
| Expose via webhook | `/ai-serving-apis` |

### Phase Plan

**Phase 1: Generate On-Brand Descriptions (start here)**
- Goal: Given a product spec, generate a description that matches the brand voice guidelines and includes required features
- Skill: `/ai-writing-content` — use the 5,000 existing descriptions as few-shot examples; build a DSPy `ChainOfThought` generator with brand voice in the signature
- Deliverable: A generation script; the marketing team reviews 20 outputs and gives a thumbs up / thumbs down with notes

**Phase 2: Score Quality Before Publishing**
- Goal: Automate the thumbs up / thumbs down decision so humans only review borderline outputs
- Skill: `/ai-scoring` — build a DSPy scorer that grades on brand voice match, feature coverage, and SEO keyword presence; calibrate against the Phase 1 human reviews
- Deliverable: A scorer that labels outputs as publish / review / reject with a confidence score

**Phase 3: Optimize Generator for Brand Voice**
- Goal: Push the fraction of outputs scored "publish" above 85% without human edits
- Skill: `/ai-improving-accuracy` — use the Phase 2 scorer as the DSPy metric; compile the generator with `MIPROv2`
- Deliverable: Compiled generator + scorer with documented publish rate on a held-out test set of 100 SKUs

**Phase 4: Production Webhook**
- Goal: Marketers trigger generation from the existing internal tool; output is stored and flagged for review if scorer says review or reject
- Skill: `/ai-serving-apis` — wrap Phase 3 pipeline in a webhook endpoint; store results with score in the internal tool
- Deliverable: Live integration; cost tracking shows reduction vs. contractor baseline

### Dependencies
- Phase 2 requires Phase 1 outputs and human reviews as calibration data — collect these during Phase 1 testing
- Phase 3 requires the Phase 2 scorer to be accurate before using it as the optimization metric; an inaccurate scorer will optimize in the wrong direction
- Phase 4 requires Phase 3 because the webhook should ship with the optimized, scored version

### What to Skip for Now
- Multilingual generation — tackle after English is solid
- SEO keyword injection — the scorer checks for keywords, but a separate keyword-stuffing step adds complexity and brand-voice risk
- Auto-publishing without human review — even 15% review rate is a 5x productivity gain over today; remove the review step in a later phase once trust is established
