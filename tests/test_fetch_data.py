import json

from src.scripts import compact_embeddings, fetch_data


def _chunks(n=3, dim=8):
    return [
        {
            "chunk_id": f"art{i:04d}-ar-00",
            "article_number": i,
            "text": "x",
            "embedding": [0.1 * i] * dim,
        }
        for i in range(1, n + 1)
    ]


def test_snapshot_roundtrip(tmp_path):
    processed, release, restored = tmp_path / "p", tmp_path / "r", tmp_path / "q"
    processed.mkdir()
    (processed / "final-articles-v2.json").write_text(json.dumps([{"article_number": 1}]))
    chunks = [{k: v for k, v in c.items() if k != "embedding"} for c in _chunks()]
    (processed / "final-chunks-bgem3-v2.json").write_text(json.dumps(chunks))
    (processed / "embedded-chunks-v2.json").write_text(json.dumps(_chunks()))

    fetch_data.make_release(processed, release)
    assert (release / "embedded-chunks-v2.npz").exists()

    assert fetch_data.from_release(restored, release)
    again = json.loads((restored / "embedded-chunks-v2.json").read_text())
    assert [c["chunk_id"] for c in again] == [c["chunk_id"] for c in _chunks()]
    assert abs(again[1]["embedding"][0] - 0.2) < 1e-3  # float16 rounding only


def test_missing_snapshot_is_reported(tmp_path):
    assert fetch_data.from_release(tmp_path / "p", tmp_path / "none") is False


def test_pack_preserves_metadata(tmp_path):
    src, npz, out = tmp_path / "e.json", tmp_path / "e.npz", tmp_path / "o.json"
    src.write_text(json.dumps(_chunks()))
    compact_embeddings.pack(src, npz)
    compact_embeddings.unpack(npz, out)
    assert json.loads(out.read_text())[0]["article_number"] == 1
