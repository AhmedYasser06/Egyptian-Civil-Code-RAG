# Evaluation sets (committed so CI and the notebooks can use them)

Copy your question sets here from `data/evaluation/` (which is git-ignored):

    mkdir -p eval && cp data/evaluation/rag_eval.json data/evaluation/retrieval_eval.json eval/

* `rag_eval.json`       - >= 54 questions with `question`, `expected_articles`, `reference_answer`
                           (RAGAS monitor needs >= 50; the CI gate uses an even 20-question subset).
* `retrieval_eval.json` - retrieval questions with gold articles (reranker distillation/evaluation).

Check the files are understood:  `python -m src.evaluation.dataset data/evaluation/rag_eval.json data/evaluation/retrieval_eval.json`
