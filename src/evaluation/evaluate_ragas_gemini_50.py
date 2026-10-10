"""Gemini-judged RAGAS on the 50-question set (the >=50-question monitor of the handbook).

python -m src.evaluation.evaluate_ragas_gemini_50
"""

import sys

from src.evaluation.evaluate_ragas import main

if __name__ == "__main__":
    main(
        [
            "--provider",
            "gemini",
            "--dataset",
            "data/evaluation/ragas_dataset_gemini_50.json",
            "--results",
            "data/evaluation/ragas_results_gemini_50.json",
            "--delay",
            "1",
            *sys.argv[1:],
        ]
    )
