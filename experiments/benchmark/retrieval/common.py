"""Shared by the evaluators: SWEBOK leaves, path resolution, the chunk map, the leaves a turn retrieved and the
citations an answer makes."""
import json
import re
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BOK = REPO / "swebok-v4"
ROOT = BOK / "root"


PAGE_HEADER = re.compile(r"#{6}[^\n]*")  # "###### SOFTWARE TESTING 5-15": PDF page headers inside sentences
PUNCT = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "•": " "})


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(PUNCT)).strip().lower()


@lru_cache(maxsize=1)
def leaf_texts() -> dict:
    """{path relative to root: normalized text, without page headers} for every content leaf."""
    return {str(p.relative_to(ROOT)): norm(PAGE_HEADER.sub(" ", p.read_text(encoding="utf-8")))
            for p in ROOT.rglob("*.md") if p.name not in ("INDEX.md", "README.md")}


@lru_cache(maxsize=1)
def chunk_map() -> dict:
    return json.loads((HERE / "chunk_to_leaves.json").read_text(encoding="utf-8"))


def to_leaf(path: str):
    """A path as an agent wrote it (absolute, relative to swebok-v4/ or to root/, with or without 'root/')
    -> the leaf path relative to root, or None if it is not a content leaf."""
    raw = path.strip().strip("`'\"*").strip()
    if not raw.endswith(".md"):
        return None
    bare = raw.lstrip("/")
    tail = bare[len("root/"):] if bare.startswith("root/") else bare
    for candidate in (Path(raw), ROOT / tail, BOK / raw, ROOT / raw, ROOT / raw.lstrip("/")):
        try:
            if candidate.is_file():
                rel = candidate.resolve().relative_to(ROOT.resolve())
                return str(rel) if rel.name not in ("INDEX.md", "README.md") else None
        except (ValueError, OSError):
            continue
    if "/" not in raw or "..." in raw or "…" in raw:  # a bare or elided name: accept it when one leaf has it
        same = by_basename().get(raw.replace("…", "/").split("/")[-1].strip(), [])
        if len(same) == 1:
            return same[0]
    return None


@lru_cache(maxsize=1)
def by_basename() -> dict:
    out = {}
    for path in leaf_texts():
        out.setdefault(Path(path).name, []).append(path)
    return out


def retrieved_leaves(turn: dict) -> set:
    """Leaves a turn brought into context: files read, documents JEV approved, chunks the vector search
    returned (mapped to leaves)."""
    found = {to_leaf(p) for p in turn.get("read", []) + turn.get("jev_approved", [])}
    for cid in turn.get("chunks", []):
        found |= set(chunk_map().get(cid, []))
    return {leaf for leaf in found if leaf}


SOURCE = re.compile(r"\(?\s*\**\s*sources?\s*\**\s*:\s*(.+)$", re.I)
QUOTE = re.compile(r"[\"“”]([^\"“”]{10,}?)[\"“”]")
SECTION = re.compile(r"§\s*\d+(?:\.\d+)*\.?\s+([A-Z][^,;()|—–\"“”*]{10,90})")
SAME = re.compile(r"\bsame (?:file|source|document|section)\b", re.I)


def quote_leaves(quote: str) -> set:
    """Leaves whose text contains the quote; an elided quote ('...', '…') is matched piece by piece."""
    pieces = [norm(p).strip(" .,;:'") for p in re.split(r"\.\.\.|…|\[\.\.\.\]", quote)]
    pieces = [p for p in pieces if len(p) >= 15] or [norm(quote).strip(" .,;:'")]
    hits = None
    for piece in pieces:
        found = set()
        for probe in (piece, piece[:60], piece[-60:]):
            found = {path for path, text in leaf_texts().items() if probe and probe in text}
            if found:
                break
        if not found:  # list items quoted without their numbering: match sentence by sentence
            for sentence in (s.strip(" .,;:") for s in piece.split(". ")):
                if len(sentence) >= 10:
                    found |= {path for path, text in leaf_texts().items() if sentence in text}
        hits = found if hits is None else (hits & found or hits | found)
    return hits or set()


def paths_in(text: str) -> set:
    """Leaves named in a free-text source. SWEBOK paths contain spaces and commas, so for every '.md' the
    longest preceding span that resolves to a real leaf wins."""
    found = set()
    for end in (m.end() for m in re.finditer(r"\.md\b", text)):
        starts = [0] + [i + 1 for i, ch in enumerate(text[:end]) if ch in " ;,(\"'`*["]
        for start in starts:  # earliest (longest) first
            leaf = to_leaf(text[start:end].strip())
            if leaf:
                found.add(leaf)
                break
    return found


BULLET = " >-•*0123456789.)\t"


def parse_citations(answer: str) -> list:
    """[{guideline, source, leaves}] for every guideline of the answer that names a source. The source may be on
    the same line as the guideline or on its own line right after the guideline's paragraph; several sources of
    the same guideline are merged. `leaves` are the content leaves the sources resolve to (by path, bare file
    name or verbatim quote); empty when they resolve to none."""
    out = []
    last_paragraph = ""
    for line in (answer or "").splitlines():
        stripped = line.strip()
        m = SOURCE.search(line)
        if not m:
            if stripped.startswith("#"):
                last_paragraph = ""
            elif stripped and not stripped.startswith(("---", ">", "|")):
                last_paragraph = stripped.strip(BULLET).strip()
            continue
        before = line[:m.start()].strip(BULLET).strip()
        guideline = before if len(before) >= 15 else last_paragraph
        if len(guideline) < 15:
            continue
        source = re.sub(r"\[JEV [0-9.]+\]\s*$", "", m.group(1)).strip().rstrip(")").strip()
        leaves = paths_in(source)
        if not leaves:
            for quote in QUOTE.findall(source.translate(PUNCT)):
                leaves |= quote_leaves(quote)
        if not leaves:  # "§4.7 State-Based and Table-Driven Construction Techniques": the leaf with that heading
            for title in SECTION.findall(source):
                leaves |= quote_leaves(title.strip())
        if not leaves and SAME.search(source) and out:  # "same file — ..." refers to the previous citation
            leaves = set(out[-1]["leaves"])
        if out and out[-1]["guideline"] == guideline:  # another source for the same guideline
            out[-1]["source"] += " | " + source
            out[-1]["leaves"] = sorted(set(out[-1]["leaves"]) | leaves)
        else:
            out.append({"guideline": guideline, "source": source, "leaves": sorted(leaves)})
    return out
