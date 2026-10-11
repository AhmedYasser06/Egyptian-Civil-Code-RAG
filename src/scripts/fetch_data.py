"""Get the corpus + embeddings onto any machine (CI, Kaggle, Docker build, a reviewer's laptop).

The DVC remote of this project used to be a MinIO on ``localhost:9000`` -- reachable from the
author's laptop only, so ``dvc pull`` failed on GitHub Actions ("Unable to locate credentials")
and on Kaggle ("Could not connect to the endpoint URL"). This script removes that dependency:

1. if ``data/processed`` already holds everything -> nothing to do;
2. else restore it from the git-tracked snapshot ``data/release/`` (~11 MB, always available after
   ``git clone`` / ``actions/checkout``): the corpus + chunks as JSON and the embeddings as a
   float16 ``.npz`` that is unpacked back to ``embedded-chunks-v2.json``;
3. else (``--dvc`` or DVC credentials present) try ``dvc pull``.

Author, once (after ``dvc repro`` / after any corpus change)::

    python -m src.scripts.fetch_data --make-release     # data/processed -> data/release
    git add data/release && git commit -m "Refresh data snapshot"

Everyone else::

    python -m src.scripts.fetch_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.scripts import compact_embeddings

PROCESSED = Path("data/processed")
RELEASE = Path("data/release")
JSON_FILES = (
    "final-articles-v2.json",
    "final-chunks-bgem3-v2.json",
    "validation_report-v2.json",
    "chunk_stats-bgem3-v2.json",
)
REQUIRED = ("final-articles-v2.json", "final-chunks-bgem3-v2.json", "embedded-chunks-v2.json")
NPZ = "embedded-chunks-v2.npz"


def _have_all(processed: Path) -> bool:
    return all((processed / name).exists() for name in REQUIRED)


def make_release(processed: Path = PROCESSED, release: Path = RELEASE) -> None:
    missing = [n for n in REQUIRED if not (processed / n).exists()]
    if missing:
        raise SystemExit(f"cannot build the snapshot, missing in {processed}: {missing}")
    release.mkdir(parents=True, exist_ok=True)
    for name in JSON_FILES:
        if (processed / name).exists():
            shutil.copy2(processed / name, release / name)
    compact_embeddings.pack(processed / "embedded-chunks-v2.json", release / NPZ)
    total = sum(p.stat().st_size for p in release.iterdir()) / 1e6
    print(f"[fetch_data] snapshot written to {release} ({total:.1f} MB) -> git add {release}")


def from_release(processed: Path = PROCESSED, release: Path = RELEASE) -> bool:
    if not (release / NPZ).exists():
        return False
    processed.mkdir(parents=True, exist_ok=True)
    for name in JSON_FILES:
        if (release / name).exists() and not (processed / name).exists():
            shutil.copy2(release / name, processed / name)
    if not (processed / "embedded-chunks-v2.json").exists():
        compact_embeddings.unpack(release / NPZ, processed / "embedded-chunks-v2.json")
    return _have_all(processed)


def from_dvc() -> bool:
    try:
        r = subprocess.run(["dvc", "pull"], check=False)
    except FileNotFoundError:
        print("[fetch_data] dvc is not installed")
        return False
    return r.returncode == 0


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def verify(processed: Path = PROCESSED) -> None:
    """Sanity checks: contiguous-ish corpus and chunk/embedding counts agree."""
    articles = json.loads((processed / "final-articles-v2.json").read_text(encoding="utf-8"))
    chunks = json.loads((processed / "final-chunks-bgem3-v2.json").read_text(encoding="utf-8"))
    emb = json.loads((processed / "embedded-chunks-v2.json").read_text(encoding="utf-8"))
    ids_c, ids_e = {c["chunk_id"] for c in chunks}, {c["chunk_id"] for c in emb}
    print(f"[fetch_data] articles={len(articles)} chunks={len(chunks)} embedded={len(emb)}")
    if ids_c != ids_e:
        raise SystemExit("[fetch_data] chunk ids differ between final-chunks and embedded-chunks")
    lock = Path("dvc.lock")
    if lock.exists():
        import yaml

        stages = yaml.safe_load(lock.read_text(encoding="utf-8")).get("stages", {})
        for stage in stages.values():
            for out in stage.get("outs", []):
                p = Path(out["path"])
                if p.name == "final-articles-v2.json" and p.exists() and out.get("md5"):
                    same = _md5(p) == out["md5"]
                    print(f"[fetch_data] final-articles md5 matches dvc.lock: {same}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--make-release", action="store_true", help="data/processed -> data/release")
    ap.add_argument("--dvc", action="store_true", help="try `dvc pull` if the snapshot is missing")
    ap.add_argument("--processed", default=str(PROCESSED))
    ap.add_argument("--release", default=str(RELEASE))
    a = ap.parse_args()
    processed, release = Path(a.processed), Path(a.release)

    if a.make_release:
        make_release(processed, release)
        return 0
    if _have_all(processed):
        print("[fetch_data] data/processed already complete")
    elif from_release(processed, release):
        print(f"[fetch_data] restored data/processed from {release}")
    elif a.dvc and from_dvc() and _have_all(processed):
        print("[fetch_data] restored data/processed with dvc pull")
    else:
        print(
            "[fetch_data] FAILED: no snapshot in data/release and DVC unavailable.\n"
            "  On the machine that has the data run:\n"
            "    python -m src.scripts.fetch_data --make-release && git add data/release && "
            "git commit && git push",
            file=sys.stderr,
        )
        return 1
    verify(processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
