# Scoring Examples

## Essay Grading

```python
import dspy
from pydantic import BaseModel, Field

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define the rubric
ESSAY_CRITERIA = [
    "clarity: Is the writing clear and easy to follow? (1=confusing, 5=crystal clear)",
    "argument: Is the argument well-structured with a clear thesis? (1=no structure, 5=compelling)",
    "evidence: Does the essay cite relevant evidence and examples? (1=none, 5=strong support)",
]

ESSAY_WEIGHTS = [0.3, 0.4, 0.3]  # Argument weighted highest

ESSAY_ANCHORS = {
    "clarity": """Score 2: "The thing about climate change is that it affects things and there are many reasons for it happening."
→ Vague referents, no specifics, reads like filler.
Score 4: "Rising sea levels threaten coastal cities like Miami, where 2.7 million residents face increased flooding risk by 2050."
→ Specific claims, concrete examples, easy to follow.""",

    "argument": """Score 2: "Climate change is bad. Also, polar bears are dying. In conclusion, we should do something."
→ No logical progression, disconnected claims, no thesis.
Score 4: "While renewable energy adoption is accelerating, three structural barriers — grid infrastructure, storage costs, and regulatory fragmentation — prevent the transition speed needed to meet 2030 targets."
→ Clear thesis, enumerated supporting points, acknowledges complexity.""",

    "evidence": """Score 2: "Many scientists agree that this is a problem."
→ Vague appeal to authority, no specific sources.
Score 4: "According to the IPCC's 2023 Synthesis Report, global temperatures have risen 1.1°C since pre-industrial levels, with the rate of increase accelerating since 1970."
→ Specific source, concrete data points, verifiable claim.""",
}


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=5)
    justification: str


class ScoreCriterion(dspy.Signature):
    """Score the essay on a single criterion. Use the anchor examples to calibrate."""
    submission: str = dspy.InputField(desc="The essay being graded")
    criterion: str = dspy.InputField(desc="The criterion to score")
    anchors: str = dspy.InputField(desc="Reference examples for calibration")
    score: int = dspy.OutputField(desc="Score from 1 to 5")
    justification: str = dspy.OutputField(desc="Evidence from the essay supporting this score")


class EssayGrader(dspy.Module):
    def __init__(self):
        self.score_criterion = dspy.ChainOfThought(ScoreCriterion)

    def forward(self, submission: str):
        scores = []
        for criterion, weight in zip(ESSAY_CRITERIA, ESSAY_WEIGHTS):
            name = criterion.split(":")[0]
            result = self.score_criterion(
                submission=submission,
                criterion=criterion,
                anchors=ESSAY_ANCHORS.get(name, ""),
            )
            scores.append(CriterionScore(
                criterion=name, score=result.score, justification=result.justification
            ))

        overall = sum(cs.score * w for cs, w in zip(scores, ESSAY_WEIGHTS))
        return dspy.Prediction(criterion_scores=scores, overall_score=round(overall, 2))


grader = EssayGrader()
result = grader(submission="Climate change poses an existential threat to coastal communities...")

for cs in result.criterion_scores:
    print(f"  {cs.criterion}: {cs.score}/5 — {cs.justification[:80]}...")
print(f"Overall: {result.overall_score}/5")
```

## Code Review Scoring

```python
CODE_CRITERIA = [
    "correctness: Does the code produce correct results for all cases? (1=broken, 5=handles all edge cases)",
    "readability: Is the code easy to understand and well-named? (1=cryptic, 5=self-documenting)",
    "security: Is the code free from vulnerabilities? (1=critical issues, 5=follows security best practices)",
]

CODE_WEIGHTS = [0.5, 0.25, 0.25]  # Correctness weighted highest

CODE_ANCHORS = {
    "correctness": """Score 2: Function returns wrong result for empty list input, no error handling for None values.
→ Fails basic edge cases that are easy to predict.
Score 4: Handles empty input, validates types, but doesn't account for concurrent access.
→ Covers common cases, misses only advanced scenarios.""",

    "readability": """Score 2: Variables named `x`, `tmp`, `d2`. No comments. 50-line function doing 4 things.
→ Reader has to reverse-engineer intent from implementation.
Score 4: Descriptive names like `user_email`, `retry_count`. Functions under 20 lines. One comment explaining a non-obvious business rule.
→ Intent is clear from reading, minimal mental overhead.""",

    "security": """Score 2: SQL query built with string concatenation. User input passed directly to shell command.
→ Classic injection vulnerabilities.
Score 4: Parameterized queries, input validation, no hardcoded secrets. Uses established auth library.
→ Follows OWASP basics, uses safe defaults.""",
}


class ScoreCodeCriterion(dspy.Signature):
    """Score the code submission on a single review criterion."""
    code: str = dspy.InputField(desc="The code being reviewed")
    criterion: str = dspy.InputField(desc="The review criterion")
    anchors: str = dspy.InputField(desc="Reference examples for calibration")
    score: int = dspy.OutputField(desc="Score from 1 to 5")
    justification: str = dspy.OutputField(desc="Specific code elements supporting this score")


class CodeReviewScorer(dspy.Module):
    def __init__(self):
        self.score_criterion = dspy.ChainOfThought(ScoreCodeCriterion)

    def forward(self, code: str):
        scores = []
        for criterion, weight in zip(CODE_CRITERIA, CODE_WEIGHTS):
            name = criterion.split(":")[0]
            result = self.score_criterion(
                code=code,
                criterion=criterion,
                anchors=CODE_ANCHORS.get(name, ""),
            )
            scores.append(CriterionScore(
                criterion=name, score=result.score, justification=result.justification
            ))

        overall = sum(cs.score * w for cs, w in zip(scores, CODE_WEIGHTS))
        return dspy.Prediction(criterion_scores=scores, overall_score=round(overall, 2))


reviewer = CodeReviewScorer()
result = reviewer(code="""
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
""")
for cs in result.criterion_scores:
    print(f"  {cs.criterion}: {cs.score}/5")
```

## Customer Support Quality Audit

```python
SUPPORT_CRITERIA = [
    "helpfulness: Did the agent resolve the customer's issue? (1=ignored the problem, 5=fully resolved)",
    "accuracy: Was the information provided correct? (1=wrong info, 5=completely accurate)",
    "tone: Was the agent professional and empathetic? (1=rude/dismissive, 5=warm and professional)",
]

SUPPORT_WEIGHTS = [0.4, 0.4, 0.2]


class ScoreSupportCriterion(dspy.Signature):
    """Score a customer support interaction on a single quality criterion."""
    conversation: str = dspy.InputField(desc="The support conversation transcript")
    criterion: str = dspy.InputField(desc="The quality criterion to score")
    score: int = dspy.OutputField(desc="Score from 1 to 5")
    justification: str = dspy.OutputField(desc="Evidence from the conversation")


class SupportAuditor(dspy.Module):
    def __init__(self):
        self.score_criterion = dspy.ChainOfThought(ScoreSupportCriterion)

    def forward(self, conversation: str):
        scores = []
        for criterion, weight in zip(SUPPORT_CRITERIA, SUPPORT_WEIGHTS):
            name = criterion.split(":")[0]
            result = self.score_criterion(
                conversation=conversation,
                criterion=criterion,
            )
            scores.append(CriterionScore(
                criterion=name, score=result.score, justification=result.justification
            ))

        overall = sum(cs.score * w for cs, w in zip(scores, SUPPORT_WEIGHTS))
        decision = "pass" if overall >= 3.5 else "needs_coaching"

        return dspy.Prediction(
            criterion_scores=scores,
            overall_score=round(overall, 2),
            decision=decision,
        )


auditor = SupportAuditor()
result = auditor(conversation="""
Customer: I've been waiting 3 weeks for my refund and nobody is helping me.
Agent: I understand your frustration. Let me look into this right now.
Agent: I can see your refund was stuck in processing. I've escalated it — you'll receive it within 2 business days. I'll email you the confirmation.
Customer: Thank you so much!
""")
print(f"Overall: {result.overall_score}/5 — {result.decision}")
```

## Evaluating Scorer Quality

```python
# Gold-standard scored examples (human-scored)
scored_trainset = [
    dspy.Example(
        submission="Climate change is a big problem...",
        gold_scores={"clarity": 3, "argument": 2, "evidence": 2},
    ).with_inputs("submission"),
    # Add 20-50+ examples
]

def scoring_metric(example, prediction, trace=None):
    errors = []
    for cs in prediction.criterion_scores:
        gold = example.gold_scores.get(cs.criterion)
        if gold is not None:
            errors.append(abs(cs.score - gold))
    if not errors:
        return 0.0
    mae = sum(errors) / len(errors)
    return max(0.0, 1.0 - mae / 4.0)

# Optimize
optimizer = dspy.MIPROv2(metric=scoring_metric, auto="medium")
optimized_grader = optimizer.compile(grader, trainset=scored_trainset)
```
