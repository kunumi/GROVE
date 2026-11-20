"""Recover the frozen evaluator inputs in a separate directory using a local Guide extraction."""
import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import re

MARK = re.compile(r"\[SWEBOK excerpt ([a-f0-9]{20})\]")
STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def transform_json(text, transform):
    """Change string values while retaining the archived JSON layout."""
    def replace(match):
        value = json.loads(match[0])
        changed = transform(value)
        return match[0] if changed == value else json.dumps(changed, ensure_ascii=False)
    return STRING.sub(replace, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guide", type=Path, help="Local swebok-v4 directory")
    parser.add_argument("destination", type=Path, help="New or empty working directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "results/excerpt_locations.json").read_text())
    destination = args.destination.resolve()
    if destination == root or root in destination.parents or (destination.exists() and any(destination.iterdir())):
        parser.error("Use a separate empty directory")

    @lru_cache(maxsize=None)
    def source(path):
        return (args.guide / path).read_text(encoding="utf-8")

    @lru_cache(maxsize=None)
    def excerpt(key):
        path, start, end, edits = manifest["excerpts"][key]
        text = source(path)[start:end]
        for offset, size, replacement in reversed(edits):
            text = text[:offset] + replacement + text[offset + size:]
        if sha256(text.encode()).hexdigest()[:20] != key:
            raise ValueError(f"Source extraction differs for excerpt {key}")
        return text

    snapshot = json.loads((root / "results/round-1/questions_used.json").read_text())
    archived_notes = {
        (question["id"], item["leaf"]): item["why"]
        for question in snapshot["questions"] for item in question["also_relevant"]
    }

    # The frozen evaluator reads the original run paths in the scoring copy.
    archived_runs = Path("results/demos/retrieval_v2")
    public_rounds = {"r1": "round-1", "r2": "round-2", "r3": "round-3"}
    # Checksums use the filenames recorded during the experiment.
    benchmark_dir = Path("experiments/benchmark/retrieval")
    public_inputs = {
        benchmark_dir / "questions_v2.json": benchmark_dir / "questions.json",
        benchmark_dir / "questions_src_v2.json": benchmark_dir / "questions_src.json",
    }
    verified = {}
    for name, checksum in manifest["files"].items():
        archived_path = Path(name)
        path = public_inputs.get(archived_path, archived_path)
        if path.is_relative_to(archived_runs):
            run, *rest = path.relative_to(archived_runs).parts
            path = Path("results", public_rounds[run], *rest)
        text = (root / path).read_text(encoding="utf-8")
        if archived_path in public_inputs:
            benchmark = json.loads(text)
            for question in benchmark["questions"]:
                for item in question["also_relevant"]:
                    item["why"] = archived_notes[question["id"], item["leaf"]]
            text = json.dumps(benchmark, ensure_ascii=False, indent=2)
        restored = transform_json(text, lambda v: MARK.sub(lambda m: excerpt(m[1]), v))
        if sha256(restored.encode()).hexdigest() != checksum:
            raise ValueError(f"Archived file checksum mismatch: {name}")
        verified[public_inputs.get(archived_path, archived_path)] = restored

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        copy = Path(temp) / "experiment"
        shutil.copytree(root / "experiments", copy / "experiments")
        for name in ("Dockerfile", "docker-compose.yml", "requirements.txt", ".dockerignore", ".gitignore"):
            shutil.copyfile(root / name, copy / name)
        (copy / "results").mkdir()
        for suffix in ("md", "csv"):
            report = Path("results") / f"retrieval_evaluation.{suffix}"
            shutil.copyfile(root / report, copy / report)
        for name, restored in verified.items():
            out = copy / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(restored, encoding="utf-8")
        # Keep the historical command examples usable in the private scoring copy.
        for archived, public in public_inputs.items():
            (copy / archived).symlink_to(public.name)
        evaluator = copy / "experiments/benchmark/evaluate_retrieval.py"
        evaluator.with_name("evaluate_retrieval_v2.py").symlink_to(evaluator.name)
        copy.rename(destination)
    print(f"Verified and restored {len(manifest['files'])} JSON files.")


if __name__ == "__main__":
    main()
