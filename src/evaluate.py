import json
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer
from summarizer import summarize

root = Path(__file__).parent.parent
samples = json.loads((root / "data" / "sample_articles.json").read_text())["samples"]

# ROUGE counts word overlap with the reference. It cannot detect factual errors.
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

rows = []
for sample in samples:
    prediction = summarize(sample["article"])  # pretrained model only, no training
    scores = scorer.score(sample["summary"], prediction)
    rows.append({"prediction": prediction, **{k: v.fmeasure for k, v in scores.items()}})

results = pd.DataFrame(rows)
results.to_csv(root / "results" / "rouge_scores.csv", index=False)
print(results[["rouge1", "rouge2", "rougeL"]].round(3))
print("\nAverage ROUGE F1:\n", results[["rouge1", "rouge2", "rougeL"]].mean().round(3))
print("\nNote: only 3 demo samples, so these scores are not a benchmark.")
