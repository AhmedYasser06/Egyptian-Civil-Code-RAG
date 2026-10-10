"""Build the drift reference from the evaluation questions and demonstrate real drift.

    python -m src.monitoring.drift_demo --questions data/evaluation/rag_eval.json

Writes ``reports/drift_reference.npz`` (used by the API through DRIFT_REFERENCE_PATH) and
``reports/drift_demo.json`` comparing three streams of queries:

* in-domain   : held-out evaluation questions (civil-code topics)      -> drift ~ 0
* out-of-domain: family / personal-status law questions (not in corpus) -> drift clearly higher
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.dataset import load_eval
from src.monitoring.drift import CosineDriftMonitor

FAMILY_LAW_QUERIES = [
    "ما هي شروط الطلاق في القانون المصري؟",
    "ما هو سن الحضانة للأم بعد الطلاق؟",
    "كيف تحسب نفقة الزوجة والأولاد؟",
    "ما هي شروط عقد الزواج الصحيح؟",
    "ما حكم الخلع وما إجراءاته؟",
    "من له الولاية على نفس القاصر في الأحوال الشخصية؟",
    "ما هي حقوق المطلقة بعد انتهاء العدة؟",
    "كيف يثبت النسب للطفل؟",
    "ما هو نصيب الزوجة والأبناء في الميراث؟",
    "ما شروط الوصية الواجبة للأحفاد؟",
    "What are the conditions for divorce under Egyptian family law?",
    "Who gets child custody after divorce in Egypt?",
    "How is alimony for a wife and children calculated?",
    "What is the legal marriage age and what are the conditions of marriage?",
    "How is an estate divided among the wife, children and parents?",
    "What are the rules on khula and judicial separation?",
    "How is paternity of a child established?",
    "What are the guardian's rights over a minor's person?",
    "ما هي إجراءات تغيير الاسم في شهادة الميلاد؟",
    "What is the waiting period (iddah) after divorce?",
]


def embed(texts: list[str], model_id: str, device: str) -> np.ndarray:
    from src.llm.providers.HuggingFaceEmbedding import HuggingFaceEmbedding

    provider = HuggingFaceEmbedding(model_id=model_id, device=device)
    return np.asarray(provider.embed_documents(texts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/evaluation/rag_eval.json")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-ref", default="reports/drift_reference.npz")
    ap.add_argument("--out-report", default="reports/drift_demo.json")
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()

    questions = [item["question"] for item in load_eval(args.questions)]
    # even indices build the reference, odd ones play "live in-domain traffic"
    ref_q, live_q = questions[::2], questions[1::2]

    ref_emb = embed(ref_q, args.model, args.device)
    monitor = CosineDriftMonitor.from_embeddings(ref_emb, min_samples=5, threshold=args.threshold)
    monitor.save(args.out_ref)

    report = {"reference_questions": len(ref_q), "threshold": args.threshold}
    for name, qs in (("in_domain", live_q), ("family_law", FAMILY_LAW_QUERIES)):
        fresh = CosineDriftMonitor.from_embeddings(ref_emb, min_samples=5, threshold=args.threshold)
        stats = None
        for vec in embed(qs, args.model, args.device):
            stats = fresh.observe(vec) or stats
        report[name] = stats
        print(name, json.dumps(stats, indent=2))

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {args.out_ref} and {args.out_report}")


if __name__ == "__main__":
    main()
