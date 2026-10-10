"""Re-ranker distillation: BAAI/bge-reranker-v2-m3 (teacher, 568M) -> small multilingual student.

    python -m src.optimization.distill_reranker build-data --device cuda
    python -m src.optimization.distill_reranker train      --device cuda
    python -m src.optimization.distill_reranker evaluate   --device cuda

* build-data: synthetic queries from every article (topic, first sentence in Arabic and in
  English), the gold article plus its dense-retrieval neighbours as candidates, scored by the
  teacher (raw logits). Articles that are gold answers of the evaluation questions are held out.
* train: student = intfloat/multilingual-e5-small with a 1-logit head, trained on
  MSE(student, teacher logits) + listwise KL over each query's candidate list.
* evaluate: recall@1/3/5, MRR, rerank latency and size for no-rerank / teacher / student on the
  evaluation questions -> reports/reranker_distillation.{json,md} and MLflow.

The student is a drop-in replacement: ``RERANKER_MODEL_ID=models/reranker-student``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from src.evaluation.dataset import load_eval

WORK = Path("eval_out/distill")
TEACHER = "BAAI/bge-reranker-v2-m3"
STUDENT = "intfloat/multilingual-e5-small"
GROUP = 16  # candidates per query


def passage(article: dict) -> str:
    """Same layout as LegalReranker.rerank so the student sees what it will see in production."""
    return (
        f"Article {article['article_number']}\n"
        f"Arabic:\n{article.get('text_ar') or ''}\n"
        f"English:\n{article.get('text_en') or ''}"
    )


def first_sentence(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    cut = re.split(r"(?<=[.!?؟۔])\s", text, maxsplit=1)[0]
    return cut[:limit]


def synth_queries(article: dict) -> list[str]:
    queries = [
        article.get("topic"),
        first_sentence(article.get("text_en", "")),
        first_sentence(article.get("text_ar", "")),
    ]
    out = []
    for q in queries:
        if q and len(q) >= 8 and q not in out:
            out.append(q)
    return out


def load_teacher_scorer(model_id: str, device: str, fp16: bool):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    if fp16 and device.startswith("cuda"):
        model = model.half()

    @torch.inference_mode()
    def score(pairs: list[tuple[str, str]], batch: int = 32, max_len: int = 384) -> list[float]:
        out: list[float] = []
        for i in range(0, len(pairs), batch):
            chunk = pairs[i : i + batch]
            enc = tok(
                [q for q, _ in chunk],
                [p for _, p in chunk],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            out += model(**enc).logits.view(-1).float().cpu().tolist()
        return out

    return score


# ----------------------------------------------------------------------------- build-data
def cmd_build_data(args) -> None:
    from sentence_transformers import SentenceTransformer

    from src.mlflow.chunking_experiment import EMBEDDING_MODEL, INPUT_FILE
    from src.retrieval.memory_retriever import InMemoryRetriever, load_articles

    random.seed(0)
    articles = load_articles(INPUT_FILE)
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    held_out = {a for q in load_eval(args.questions) for a in q["expected_articles"]}

    embed = SentenceTransformer(EMBEDDING_MODEL, device=args.device)
    retriever = InMemoryRetriever(chunks, articles, embed, reranker=None, candidate_k=GROUP)
    score = load_teacher_scorer(TEACHER, args.device, fp16=True)

    groups = []
    numbers = [n for n in articles if n not in held_out and not articles[n].get("is_repealed")]
    random.shuffle(numbers)
    for n in numbers[: args.max_articles]:
        for q in synth_queries(articles[n]):
            cands = retriever.retrieve(q, top_k=GROUP)
            nums = [int(c["article_number"]) for c in cands]
            if n not in nums:
                nums = [n] + nums[: GROUP - 1]
            while len(nums) < GROUP:  # pad with random negatives
                nums.append(random.choice(list(articles)))
            nums = list(dict.fromkeys(nums))[:GROUP]
            groups.append({"query": q, "gold": n, "articles": nums})
        if len(groups) % 200 == 0:
            print(f"[build-data] {len(groups)} queries")

    pairs = [(g["query"], passage(articles[a])) for g in groups for a in g["articles"]]
    print(f"[build-data] teacher scoring {len(pairs)} pairs")
    logits = score(pairs)
    i = 0
    for g in groups:
        g["teacher_logits"] = logits[i : i + len(g["articles"])]
        i += len(g["articles"])

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "groups.json").write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")
    print(f"[build-data] {len(groups)} query groups -> {WORK / 'groups.json'}")


# ----------------------------------------------------------------------------- train
def cmd_train(args) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.mlflow.chunking_experiment import INPUT_FILE
    from src.retrieval.memory_retriever import load_articles

    random.seed(0)
    torch.manual_seed(0)
    articles = load_articles(INPUT_FILE)
    groups = json.loads((WORK / "groups.json").read_text(encoding="utf-8"))
    random.shuffle(groups)
    val, train = groups[: args.val], groups[args.val :]

    tok = AutoTokenizer.from_pretrained(args.student)
    model = AutoModelForSequenceClassification.from_pretrained(args.student, num_labels=1)
    model.to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train) // args.queries_per_step)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1
    )
    scaler = torch.amp.GradScaler(enabled=args.device.startswith("cuda"))

    def forward(batch):
        qs, ps = [], []
        for g in batch:
            for a in g["articles"]:
                qs.append(g["query"])
                ps.append(passage(articles[a]))
        enc = tok(
            qs, ps, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt"
        ).to(args.device)
        with torch.autocast(args.device.split(":")[0], enabled=args.device.startswith("cuda")):
            logits = model(**enc).logits.view(-1)
        return logits.float()

    def losses(batch):
        student = forward(batch)
        teacher = torch.tensor([x for g in batch for x in g["teacher_logits"]], device=args.device)
        mse = F.mse_loss(student, teacher)
        kl = 0.0
        pos = 0
        for g in batch:
            n = len(g["articles"])
            s, t = student[pos : pos + n], teacher[pos : pos + n]
            pos += n
            kl = kl + F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1), reduction="sum")
        return mse + args.kl_weight * kl / len(batch)

    step = 0
    for epoch in range(args.epochs):
        random.shuffle(train)
        for i in range(0, len(train) - args.queries_per_step + 1, args.queries_per_step):
            loss = losses(train[i : i + args.queries_per_step])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if step < steps:
                sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[train] epoch {epoch} step {step}/{steps} loss {loss.item():.4f}")
        model.eval()
        with torch.no_grad():
            vloss = sum(
                losses(val[i : i + args.queries_per_step]).item()
                for i in range(0, len(val), args.queries_per_step)
            )
        model.train()
        print(f"[train] epoch {epoch} val loss {vloss:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[train] student saved to {out}")


# ----------------------------------------------------------------------------- evaluate
def cmd_evaluate(args) -> None:
    from sentence_transformers import SentenceTransformer

    import mlflow
    from src.mlflow.chunking_experiment import EMBEDDING_MODEL, INPUT_FILE
    from src.retrieval.memory_retriever import InMemoryRetriever, load_articles
    from src.retrieval.reranker import LegalReranker

    articles = load_articles(INPUT_FILE)
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    questions = [q for q in load_eval(args.questions) if q["expected_articles"]]
    embed = SentenceTransformer(EMBEDDING_MODEL, device=args.device)
    base = InMemoryRetriever(chunks, articles, embed, reranker=None, candidate_k=15)
    candidates = {q["id"]: base.retrieve(q["question"], top_k=15) for q in questions}

    def size_mb(r):
        return sum(p.numel() for p in r.model.parameters()) * 4 / 1e6

    def run(name, reranker):
        r1 = r3 = r5 = mrr = 0.0
        lat = []
        for q in questions:
            cands = [dict(c) for c in candidates[q["id"]]]
            t0 = time.perf_counter()
            ranked = reranker.rerank(q["question"], cands, 5) if reranker else cands[:5]
            lat.append((time.perf_counter() - t0) * 1000)
            got = [int(a["article_number"]) for a in ranked]
            gold = set(q["expected_articles"])
            r1 += bool(gold & set(got[:1]))
            r3 += bool(gold & set(got[:3]))
            r5 += bool(gold & set(got[:5]))
            mrr += next((1 / (i + 1) for i, a in enumerate(got) if a in gold), 0.0)
        n = len(questions)
        lat.sort()
        return {
            "name": name,
            "recall@1": r1 / n,
            "recall@3": r3 / n,
            "recall@5": r5 / n,
            "mrr": mrr / n,
            "rerank_ms_mean": sum(lat) / n,
            "rerank_ms_p95": lat[int(0.95 * (n - 1))],
            "params_m": sum(p.numel() for p in reranker.model.parameters()) / 1e6
            if reranker
            else 0,
            "size_mb_fp32": size_mb(reranker) if reranker else 0,
        }

    results = [run("no_rerank", None)]
    teacher = LegalReranker(TEACHER, device=args.device, max_length=384)
    results.append(run("teacher", teacher))
    del teacher
    student = LegalReranker(args.student_path, device=args.device, max_length=384)
    results.append(run("student", student))

    t, s = results[1], results[2]
    summary = {
        "n_questions": len(questions),
        "results": results,
        "speedup": t["rerank_ms_mean"] / s["rerank_ms_mean"],
        "size_ratio": t["params_m"] / s["params_m"],
        "recall@5_retained": s["recall@5"] / t["recall@5"] if t["recall@5"] else None,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/reranker_distillation.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "| reranker | R@1 | R@3 | R@5 | MRR | ms/query (mean) | ms p95 | params (M) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | "
            f"{r['recall@5']:.3f} | {r['mrr']:.3f} | {r['rerank_ms_mean']:.0f} | "
            f"{r['rerank_ms_p95']:.0f} | {r['params_m']:.0f} |"
        )
    lines.append(
        f"\nstudent is {summary['speedup']:.1f}x faster and "
        f"{summary['size_ratio']:.1f}x smaller; keeps "
        f"{(summary['recall@5_retained'] or 0) * 100:.0f}% of teacher recall@5 "
        f"on {len(questions)} questions."
    )
    Path("reports/reranker_distillation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("civil-code-rag-optimization")
    for r in results:
        with mlflow.start_run(run_name=f"reranker_{r['name']}"):
            mlflow.log_metrics({k: v for k, v in r.items() if k != "name"})
            mlflow.log_param("reranker", r["name"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--questions", default="data/evaluation/retrieval_eval.json")
    ap.add_argument("--chunks", default="data/processed/final-chunks-bgem3-v2.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data")
    b.add_argument("--max-articles", type=int, default=1200)
    b.set_defaults(fn=cmd_build_data)

    t = sub.add_parser("train")
    t.add_argument("--student", default=STUDENT)
    t.add_argument("--out", default="models/reranker-student")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--queries-per-step", type=int, default=2)
    t.add_argument("--max-len", type=int, default=384)
    t.add_argument("--kl-weight", type=float, default=1.0)
    t.add_argument("--val", type=int, default=100)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate")
    e.add_argument("--student-path", default="models/reranker-student")
    e.set_defaults(fn=cmd_evaluate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
