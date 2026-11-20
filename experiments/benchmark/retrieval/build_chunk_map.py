"""Map every chunk of swebok.duckdb (demo 2's vector index) to the leaves of swebok-v4/root it contains.

The chunks come from the flat swebok-v4.md and carry no path, so the map is built by text overlap: a chunk
maps to a leaf when they share an 80-character window of normalized text. Keys are chunk_id(text), the
same id the notebook records when search_swebok returns a chunk.

    python experiments/benchmark/retrieval/build_chunk_map.py
"""
import hashlib
import json
import re
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO / "swebok-v4" / "root"
WINDOW, STRIDE = 80, 40


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def chunk_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main():
    with duckdb.connect(str(REPO / "swebok.duckdb"), read_only=True) as con:
        chunks = [row[0] for row in con.execute("SELECT chunk FROM swebok_chunks").fetchall()]
    normed = [norm(c) for c in chunks]
    leaves = {str(p.relative_to(ROOT)): norm(p.read_text(encoding="utf-8"))
              for p in ROOT.rglob("*.md") if p.name not in ("INDEX.md", "README.md")}
    mapping = {chunk_id(c): [] for c in chunks}
    for path, text in leaves.items():
        windows = [text[i:i + WINDOW] for i in range(0, max(len(text) - WINDOW, 0) + 1, STRIDE)]
        windows = [w for w in windows if len(w) == WINDOW and not w.startswith("###")]
        for c, n in zip(chunks, normed):
            if any(w in n for w in windows):
                mapping[chunk_id(c)].append(path)
    (HERE / "chunk_to_leaves.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = [len(v) for v in mapping.values()]
    print(f"{len(chunks)} chunks; {sum(s == 0 for s in sizes)} map to no leaf; "
          f"mean {sum(sizes) / len(sizes):.1f} leaves per chunk; max {max(sizes)}")


if __name__ == "__main__":
    main()
