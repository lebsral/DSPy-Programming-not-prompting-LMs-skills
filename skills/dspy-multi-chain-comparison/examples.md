# dspy-multi-chain-comparison -- Worked Examples

## Example 1: Complex analysis with multi-chain comparison

An architecture review module that uses multiple reasoning chains to evaluate a proposed system design. Each chain considers the design independently, and the comparison step picks the most thorough analysis.

```python
import dspy
from typing import Literal
from pydantic import BaseModel, Field


# --- Signatures ---

class ArchitectureReview(dspy.Signature):
    """Review a proposed system architecture and identify strengths, weaknesses, and recommendations."""
    design_description: str = dspy.InputField(desc="The proposed architecture or design")
    requirements: str = dspy.InputField(desc="Key requirements the design must satisfy")
    strengths: str = dspy.OutputField(desc="What the design does well")
    weaknesses: str = dspy.OutputField(desc="Gaps, risks, or areas of concern")
    recommendation: str = dspy.OutputField(desc="Concrete next steps to improve the design")
    overall_rating: Literal["strong", "adequate", "needs_work", "risky"] = dspy.OutputField()


# --- Module ---

class ArchitectureReviewer(dspy.Module):
    """Review a system design using multiple reasoning chains for thorough analysis."""

    def __init__(self, num_chains=3):
        self.extract_requirements = dspy.ChainOfThought(
            "design_description -> key_requirements: str"
        )
        self.review = dspy.MultiChainComparison(ArchitectureReview, M=num_chains)

    def forward(self, design_description, requirements=""):
        # If no explicit requirements, extract them from the design
        if not requirements.strip():
            extracted = self.extract_requirements(design_description=design_description)
            requirements = extracted.key_requirements

        result = self.review(
            design_description=design_description,
            requirements=requirements,
        )

        return dspy.Prediction(
            strengths=result.strengths,
            weaknesses=result.weaknesses,
            recommendation=result.recommendation,
            overall_rating=result.overall_rating,
        )


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

reviewer = ArchitectureReviewer(num_chains=3)

result = reviewer(
    design_description="""
    We're building a real-time analytics dashboard. The plan:
    - React frontend with WebSocket connections to a Node.js API
    - Node.js API queries PostgreSQL directly for all data
    - PostgreSQL stores raw events (500M rows, growing 2M/day)
    - Cron job runs nightly to compute aggregate tables
    - Single server deployment behind Nginx
    """,
    requirements="""
    - Dashboard must load in under 2 seconds
    - Support 200 concurrent users
    - Show data no more than 5 minutes old
    - Budget: $500/month infrastructure
    """,
)

print(f"Rating: {result.overall_rating}")
print(f"\nStrengths:\n{result.strengths}")
print(f"\nWeaknesses:\n{result.weaknesses}")
print(f"\nRecommendation:\n{result.recommendation}")


# --- Optimization ---

def review_metric(example, prediction, trace=None):
    """Score based on whether the review catches known issues."""
    # Check that known weaknesses are mentioned
    known_issues = example.known_issues  # list of strings
    weaknesses_text = prediction.weaknesses.lower()

    issues_found = sum(
        1 for issue in known_issues
        if issue.lower() in weaknesses_text
    )
    coverage = issues_found / len(known_issues) if known_issues else 0

    # Check rating matches expected
    rating_correct = prediction.overall_rating == example.expected_rating

    return 0.6 * coverage + 0.4 * rating_correct


# trainset = [
#     dspy.Example(
#         design_description="...",
#         requirements="...",
#         known_issues=["single point of failure", "no caching layer"],
#         expected_rating="needs_work",
#     ).with_inputs("design_description", "requirements"),
# ]
# optimizer = dspy.BootstrapFewShot(metric=review_metric, max_bootstrapped_demos=4)
# optimized = optimizer.compile(reviewer, trainset=trainset)
# optimized.save("architecture_reviewer.json")
```

Key points:
- `MultiChainComparison` shines here because architecture reviews benefit from multiple perspectives -- one chain might catch scalability issues while another notices security gaps
- The comparison step picks the analysis that covers the most ground, not just the first one generated
- Pairing with `ChainOfThought` for requirement extraction shows how to compose `MultiChainComparison` inside a larger module
- The metric checks whether the review catches known issues, not just whether it produces any output


## Example 2: Decision-making with multiple perspectives

A hiring decision module that evaluates a candidate from multiple angles. Each chain reasons independently about the candidate's fit, and the comparison step synthesizes the most balanced assessment.

```python
import dspy
from typing import Literal
from pydantic import BaseModel, Field


# --- Signatures ---

class CandidateEvaluation(dspy.Signature):
    """Evaluate a job candidate based on their profile and the role requirements."""
    candidate_profile: str = dspy.InputField(desc="Candidate's experience, skills, and background")
    role_requirements: str = dspy.InputField(desc="What the role needs")
    team_context: str = dspy.InputField(desc="Current team composition and gaps")
    strengths_for_role: str = dspy.OutputField(desc="How the candidate's strengths match the role")
    concerns: str = dspy.OutputField(desc="Potential gaps or risks")
    growth_areas: str = dspy.OutputField(desc="Where the candidate would need to develop")
    hiring_recommendation: Literal["strong_yes", "yes", "maybe", "no"] = dspy.OutputField()


class InterviewQuestions(dspy.Signature):
    """Generate targeted interview questions based on the evaluation."""
    evaluation_summary: str = dspy.InputField(desc="Summary of the candidate evaluation")
    concerns: str = dspy.InputField(desc="Areas of concern to probe")
    questions: list[str] = dspy.OutputField(
        desc="3-5 targeted interview questions to address the concerns"
    )


# --- Module ---

class HiringAdvisor(dspy.Module):
    """Evaluate a candidate from multiple perspectives and generate interview questions."""

    def __init__(self):
        self.evaluate = dspy.MultiChainComparison(CandidateEvaluation, M=4)
        self.generate_questions = dspy.ChainOfThought(InterviewQuestions)

    def forward(self, candidate_profile, role_requirements, team_context):
        # Stage 1: Multi-perspective evaluation
        # Each chain may weigh different factors -- technical depth, culture fit,
        # growth potential, risk tolerance. The comparison picks the most balanced view.
        evaluation = self.evaluate(
            candidate_profile=candidate_profile,
            role_requirements=role_requirements,
            team_context=team_context,
        )

        # Stage 2: Generate targeted questions based on concerns
        questions_result = self.generate_questions(
            evaluation_summary=f"Strengths: {evaluation.strengths_for_role}",
            concerns=evaluation.concerns,
        )

        return dspy.Prediction(
            strengths=evaluation.strengths_for_role,
            concerns=evaluation.concerns,
            growth_areas=evaluation.growth_areas,
            recommendation=evaluation.hiring_recommendation,
            interview_questions=questions_result.questions,
        )


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

advisor = HiringAdvisor()

result = advisor(
    candidate_profile="""
    5 years at a Series B startup as senior backend engineer.
    Strong in Python, PostgreSQL, Redis. Built their event processing pipeline
    handling 10k events/sec. Led a team of 3 for the last year.
    No experience with Go or Kubernetes. Has given conference talks on
    distributed systems. Left because the company was acquired.
    """,
    role_requirements="""
    Staff engineer role at a Series C company. Need someone who can:
    - Design and lead implementation of a new microservices platform
    - Mentor a team of 6 engineers (mix of mid and senior)
    - Work primarily in Go with Kubernetes on AWS
    - Handle ambiguity and drive technical decisions
    """,
    team_context="""
    Current team: 6 engineers, mostly mid-level, strong in Go but weak on
    system design. No staff engineer currently. Tech lead left 2 months ago.
    Team morale is low and they need a technical leader.
    """,
)

print(f"Recommendation: {result.recommendation}")
print(f"\nStrengths:\n{result.strengths}")
print(f"\nConcerns:\n{result.concerns}")
print(f"\nGrowth areas:\n{result.growth_areas}")
print(f"\nInterview questions:")
for i, q in enumerate(result.interview_questions, 1):
    print(f"  {i}. {q}")


# --- Optimization ---

def hiring_metric(example, prediction, trace=None):
    """Score based on recommendation accuracy and concern coverage."""
    # Check recommendation matches
    rec_correct = prediction.recommendation == example.expected_recommendation

    # Check that key concerns are raised
    concerns_text = prediction.concerns.lower()
    expected_concerns = example.expected_concerns  # list of strings
    concerns_found = sum(
        1 for c in expected_concerns
        if c.lower() in concerns_text
    )
    concern_coverage = concerns_found / len(expected_concerns) if expected_concerns else 1.0

    # Check interview questions are relevant (at least 3 generated)
    has_questions = len(prediction.interview_questions) >= 3

    return 0.4 * rec_correct + 0.4 * concern_coverage + 0.2 * has_questions


# trainset = [
#     dspy.Example(
#         candidate_profile="...",
#         role_requirements="...",
#         team_context="...",
#         expected_recommendation="yes",
#         expected_concerns=["no go experience", "no kubernetes experience"],
#     ).with_inputs("candidate_profile", "role_requirements", "team_context"),
# ]
# optimizer = dspy.MIPROv2(metric=hiring_metric, auto="medium")
# optimized = optimizer.compile(advisor, trainset=trainset)
# optimized.save("hiring_advisor.json")
```

Key points:
- Using M=4 chains gives more perspective diversity for a nuanced people-decision -- each chain may emphasize different aspects (technical fit, leadership readiness, culture alignment, risk)
- The comparison step picks the most balanced assessment rather than the most optimistic or pessimistic one
- Chaining `MultiChainComparison` into `ChainOfThought` for follow-up questions shows the natural composition pattern -- use the expensive multi-chain step where it matters most, then use cheaper single-chain steps for downstream tasks
- The metric checks both the recommendation and whether specific concerns are raised, ensuring the evaluation is thorough, not just confident
