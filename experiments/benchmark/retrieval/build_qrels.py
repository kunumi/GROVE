"""questions_src.json -> questions.json: adds the expected leaves of each question.

A leaf is expected when its text contains one of the question's evidence phrases (whitespace-normalized,
case-insensitive). Paths are relative to swebok-v4/root. Fails if an evidence phrase matches no leaf.

    python experiments/benchmark/retrieval/build_qrels.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "swebok-v4" / "root"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def leaves():
    return {str(p.relative_to(ROOT)): norm(p.read_text(encoding="utf-8"))
            for p in ROOT.rglob("*.md") if p.name not in ("INDEX.md", "README.md")}


def main():
    src = json.loads((HERE / "questions_src.json").read_text(encoding="utf-8"))
    texts = leaves()
    for q in src["questions"]:
        expected = set()
        for phrase in q["evidence"]:
            hits = {path for path, text in texts.items() if norm(phrase) in text}
            if not hits:
                raise SystemExit(f"{q['id']}: evidence matches no leaf: {phrase!r}")
            expected |= hits
        q["expected_leaves"] = sorted(expected)
    (HERE / "questions.json").write_text(json.dumps(src, ensure_ascii=False, indent=2), encoding="utf-8")
    for q in src["questions"]:
        print(q["id"], len(q["expected_leaves"]), [p.split("/")[-1] for p in q["expected_leaves"]])


if __name__ == "__main__":
    main()
