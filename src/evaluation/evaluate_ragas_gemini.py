"""Gemini-judged RAGAS on the 30-sample set (thin wrapper over the resilient evaluator).

python -m src.evaluation.evaluate_ragas_gemini
"""

from src.evaluation.evaluate_ragas import main

if __name__ == "__main__":
    main(
        [
            "--provider",
            "gemini",
            "--dataset",
            "data/evaluation/ragas_dataset.json",
            "--results",
            "data/evaluation/ragas_gemini_results.json",
            "--delay",
            "1",
        ]
    )
