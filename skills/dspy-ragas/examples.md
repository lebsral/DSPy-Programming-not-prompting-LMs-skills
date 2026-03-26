# Ragas Examples

## Evaluate a support bot RAG pipeline

```python
import dspy
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Your support bot RAG pipeline
class SupportBot(dspy.Module):
    def __init__(self, retriever):
        self.retrieve = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        passages = self.retrieve(question).passages
        result = self.answer(context=passages, question=question)
        return dspy.Prediction(answer=result.answer, context=passages)

# Set up retriever (using FAISS for this example)
embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
corpus = [
    "Refunds are processed within 5-7 business days.",
    "To reset your password, go to Settings > Security > Reset Password.",
    "Enterprise plans include SSO, SAML, and dedicated support.",
    "Free trial lasts 14 days with full feature access.",
    "Billing is monthly. Annual plans get 20% discount.",
]
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=3)
bot = SupportBot(retriever=search)

# Test questions with ground truth
test_data = [
    {"question": "How long do refunds take?", "answer": "5-7 business days"},
    {"question": "How do I reset my password?", "answer": "Go to Settings > Security > Reset Password"},
    {"question": "What's included in enterprise?", "answer": "SSO, SAML, and dedicated support"},
    {"question": "How long is the free trial?", "answer": "14 days with full features"},
]

# Collect predictions
samples = []
for item in test_data:
    pred = bot(question=item["question"])
    samples.append(SingleTurnSample(
        user_input=item["question"],
        response=pred.answer,
        retrieved_contexts=pred.context,
        reference=item["answer"],
    ))

dataset = EvaluationDataset(samples=samples)

# Run Ragas evaluation
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
)

print("Ragas Evaluation Results:")
print(f"  Faithfulness:      {result['faithfulness']:.2f}")
print(f"  Answer Relevancy:  {result['answer_relevancy']:.2f}")
print(f"  Context Precision: {result['context_precision']:.2f}")
print(f"  Context Recall:    {result['context_recall']:.2f}")

# Interpret: if context_recall is low, the retriever is missing relevant docs
# If faithfulness is low, the generator is hallucinating beyond the context
```

## Diagnose retriever vs generator issues

```python
import pandas as pd
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision

# After collecting samples from your pipeline...
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), ContextRecall(), ContextPrecision()],
)

df = result.to_pandas()

# Find samples where retriever fails (low context recall)
retriever_failures = df[df["context_recall"] < 0.5]
print(f"\nRetriever failures ({len(retriever_failures)} samples):")
for _, row in retriever_failures.iterrows():
    print(f"  Q: {row['user_input']}")
    print(f"     Context Recall: {row['context_recall']:.2f}")

# Find samples where generator fails (low faithfulness despite good retrieval)
generator_failures = df[(df["faithfulness"] < 0.5) & (df["context_recall"] >= 0.7)]
print(f"\nGenerator failures ({len(generator_failures)} samples):")
for _, row in generator_failures.iterrows():
    print(f"  Q: {row['user_input']}")
    print(f"     Faithfulness: {row['faithfulness']:.2f}")
    print(f"     Context Recall: {row['context_recall']:.2f}")

# Summary diagnosis
avg_ctx_recall = df["context_recall"].mean()
avg_faith = df["faithfulness"].mean()

if avg_ctx_recall < 0.7:
    print("\n→ RETRIEVER is the bottleneck. Improve chunking, embeddings, or k.")
    print("  Try: /ai-searching-docs or /dspy-qdrant")
elif avg_faith < 0.8:
    print("\n→ GENERATOR is the bottleneck. Improve grounding or optimize prompts.")
    print("  Try: /ai-stopping-hallucinations or /dspy-miprov2")
else:
    print("\n→ Pipeline looks healthy. Focus on edge cases.")
```

## Compare before and after optimization

```python
import dspy
from dspy.evaluate import Evaluate
from ragas import evaluate as ragas_evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# --- Baseline ---
baseline_rag = RAG(retriever=search)

# DSPy metric (fast, used for optimization)
from dspy.evaluate import SemanticF1
dspy_metric = SemanticF1()
evaluator = Evaluate(devset=devset, metric=dspy_metric, num_threads=4)
baseline_dspy_score = evaluator(baseline_rag)

# Ragas metrics (slow, used for analysis)
baseline_samples = collect_ragas_samples(baseline_rag, devset)
baseline_ragas = ragas_evaluate(
    dataset=EvaluationDataset(samples=baseline_samples),
    metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
)

# --- Optimize ---
optimizer = dspy.MIPROv2(metric=dspy_metric, auto="medium")
optimized_rag = optimizer.compile(baseline_rag, trainset=trainset)

# --- Optimized ---
optimized_dspy_score = evaluator(optimized_rag)

optimized_samples = collect_ragas_samples(optimized_rag, devset)
optimized_ragas = ragas_evaluate(
    dataset=EvaluationDataset(samples=optimized_samples),
    metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
)

# --- Compare ---
print("DSPy SemanticF1:")
print(f"  Baseline:  {baseline_dspy_score:.1f}%")
print(f"  Optimized: {optimized_dspy_score:.1f}%")

print("\nRagas Decomposed:")
for metric_name in ["faithfulness", "answer_relevancy", "context_recall"]:
    before = baseline_ragas[metric_name]
    after = optimized_ragas[metric_name]
    print(f"  {metric_name}: {before:.2f} → {after:.2f} ({after - before:+.2f})")
```
