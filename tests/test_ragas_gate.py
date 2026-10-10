import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ragas_gate", Path(__file__).parent.parent / "scripts" / "ragas_gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def report(faith, n=20, corpus="c1", questions="q1", scored=None):
    return {
        "meta": {"n_questions": n, "corpus_md5": corpus, "questions_md5": questions},
        "aggregate": {"faithfulness": faith, "faithfulness_n": scored if scored is not None else n},
    }


def test_passes_at_threshold():
    assert gate.evaluate(report(0.75), 0.75, 20, "c1", "q1") == []


def test_fails_below_threshold():
    assert "faithfulness 0.740" in gate.evaluate(report(0.74), 0.75, 20, "c1", "q1")[0]


def test_fails_when_too_few_questions():
    assert any("need >= 20" in p for p in gate.evaluate(report(0.9, n=10), 0.75, 20, None, None))


def test_fails_when_report_is_stale():
    problems = gate.evaluate(report(0.9), 0.75, 20, "DIFFERENT", "q1")
    assert any("corpus changed" in p for p in problems)
    problems = gate.evaluate(report(0.9), 0.75, 20, "c1", "DIFFERENT")
    assert any("rag_eval.json changed" in p for p in problems)


def test_fails_when_judge_failed_on_many_questions():
    assert any("only" in p for p in gate.evaluate(report(0.9, scored=10), 0.75, 20, None, None))


def test_reads_corpus_md5_from_dvc_lock(tmp_path):
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "schema: '2.0'\nstages:\n  extract_corpus:\n    outs:\n"
        "    - path: data/processed/final-articles-v2.json\n      md5: abc123\n"
    )
    assert gate.corpus_md5_from_lock(lock) == "abc123"
