"""Pack / unpack ``embedded-chunks-v2.json`` into one small file (upload, Kaggle dataset).

    python -m src.scripts.compact_embeddings pack     # JSON (huge) -> embedded-chunks-v2.npz
    python -m src.scripts.compact_embeddings unpack   # .npz -> JSON (text/meta identical,
                                                      #         vectors rounded to float16)

float16 keeps cosine similarity within ~1e-3, which does not change the top-k.  The .npz is
roughly 8x smaller than the JSON.  Nothing here is needed if you use DVC remotes (the normal route).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

JSON_PATH = Path("data/processed/embedded-chunks-v2.json")
NPZ_PATH = Path("data/processed/embedded-chunks-v2.npz")


def pack(src: Path = JSON_PATH, dst: Path = NPZ_PATH) -> None:
    chunks = json.loads(src.read_text(encoding="utf-8"))
    vectors = np.asarray([c["embedding"] for c in chunks], dtype=np.float16)
    meta = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]
    np.savez_compressed(dst, vectors=vectors, meta=json.dumps(meta, ensure_ascii=False))
    print(f"packed {len(chunks)} chunks -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def unpack(src: Path = NPZ_PATH, dst: Path = JSON_PATH) -> None:
    data = np.load(src, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    vectors = data["vectors"].astype(np.float32)
    for chunk, vec in zip(meta, vectors):
        chunk["embedding"] = [round(float(x), 6) for x in vec]
    dst.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"unpacked {len(meta)} chunks -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["pack", "unpack"])
    ap.add_argument("--json", default=str(JSON_PATH))
    ap.add_argument("--npz", default=str(NPZ_PATH))
    a = ap.parse_args()
    if a.action == "pack":
        pack(Path(a.json), Path(a.npz))
    else:
        unpack(Path(a.npz), Path(a.json))


if __name__ == "__main__":
    main()
