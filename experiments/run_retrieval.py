"""Retrieval experiment v2: every question to every architecture, logging what each one brings back from the SWEBOK.

Port of the retrieval part of the experiments notebook (version of 18/11), as a headless script, with the v2 changes marked "v2:".
The architectures' logic (prompts, Jev questions, policies, JevSearcher) is the notebook's.

    docker compose run --rm app python experiments/run_retrieval.py --run smoke --questions Q01
    docker compose run --rm app python experiments/run_retrieval.py --run r1 --questions-file questions_v2.json

Output: results/demos/retrieval_v2/<run>/<demo>/<question>.md|.json (one turn each: answer, tool calls, usage, sources).
"""
import argparse
import asyncio
import base64
import contextvars
import datetime
import hashlib
import json
import mimetypes
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aisuite
import duckdb
import numpy as np
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulCriteria, TypeSafeClient

from fast_track_lib import Clause, Policy, create_agent
from fast_track_lib.core.drivers.aisuite_driver import AisuiteDriver
from fast_track_lib.core.embeddings.models import AisuiteEmbeddingModel
from fast_track_lib.tools.basic import read_file, write_file
from fast_track_lib.webdemo import AgentDemo

load_dotenv()
client = aisuite.Client()  # read_images and jev_explore use it

LLM_MODEL = "anthropic:claude-sonnet-4-6"
AGENT_KWARGS = dict(model=LLM_MODEL, memory="dynamic",
                    memory_params={"window_size": 100000, "token_limit": 100000})

# ── per-turn recorder (notebook cell 1) ──────────────────────────────────────────────────────────────────────
class TurnRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.llm = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        self.jev_calls = 0
        self.embedding_calls = 0
        self.embedding_texts = 0
        self.jev_chars = 0        # v2: characters sent to Jev (state + questions)
        self.jev_input_tokens = 0   # v2: Jev usage as reported by the API
        self.jev_output_tokens = 0
        self.jev_models = set()
        self.embedding_chars = 0  # v2: characters embedded
        self.read = []
        self.jev_approved = []
        self.chunks = []
        self.admission = []

    def to_dict(self):
        return {"llm": dict(self.llm), "jev_calls": self.jev_calls, "embedding_calls": self.embedding_calls,
                "embedding_texts": self.embedding_texts, "jev_chars": self.jev_chars,
                "jev_input_tokens": self.jev_input_tokens, "jev_output_tokens": self.jev_output_tokens,
                "jev_models": sorted(self.jev_models),
                "embedding_chars": self.embedding_chars, "read": self.read, "jev_approved": self.jev_approved,
                "chunks": self.chunks, "admission": self.admission}


RECORDER = contextvars.ContextVar("turn_recorder", default=None)


def record(update):
    rec = RECORDER.get()
    if rec is not None:
        with rec.lock:
            update(rec)


def chunk_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _tokens(usage):
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, dict) else (lambda key: getattr(usage, key, None))
    return (int(get("prompt_tokens") or get("input_tokens") or 0),
            int(get("completion_tokens") or get("output_tokens") or 0))


def note_llm(model, response):
    inp, out = _tokens(getattr(response, "usage", None))

    def update(rec):
        d = rec.llm[model]
        d["calls"] += 1
        d["input_tokens"] += inp
        d["output_tokens"] += out
    record(update)


def _wrap_once(cls, name, make):
    if not getattr(cls, f"_metered_{name}", False):
        setattr(cls, name, make(getattr(cls, name)))
        setattr(cls, f"_metered_{name}", True)


def _sync_llm(orig):
    def complete(self, model, messages, **params):
        response = orig(self, model, messages, **params)
        note_llm(model, response)
        return response
    return complete


def _async_llm(orig):
    async def acomplete(self, model, messages, **params):
        response = await orig(self, model, messages, **params)
        note_llm(model, response)
        return response
    return acomplete


def _jev_size(kwargs) -> int:
    size = 0
    for key in ("state", "questions"):
        value = kwargs.get(key)
        if value is not None:
            size += len(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str))
    return size


def _note_jev(kwargs):
    size = _jev_size(kwargs)

    def update(rec):
        rec.jev_calls += 1
        rec.jev_chars += size
    record(update)


def _note_jev_usage(result):
    usage, model = getattr(result, "usage", None), getattr(result, "model", None)
    inp, out = _tokens(usage)

    def update(rec):
        rec.jev_input_tokens += inp
        rec.jev_output_tokens += out
        if model:
            rec.jev_models.add(str(model))
    record(update)


def _sync_jev(orig):
    def system_one(self, *args, **kwargs):
        _note_jev(kwargs)
        result = orig(self, *args, **kwargs)
        _note_jev_usage(result)
        return result
    return system_one


def _async_jev(orig):
    async def system_one(self, *args, **kwargs):
        _note_jev(kwargs)
        result = await orig(self, *args, **kwargs)
        _note_jev_usage(result)
        return result
    return system_one


def _embed(orig):
    def encode(self, sentences, *args, **kwargs):
        texts = list(sentences) if isinstance(sentences, (list, tuple)) else [sentences]

        def update(rec):
            rec.embedding_calls += 1
            rec.embedding_texts += len(texts)
            rec.embedding_chars += sum(len(str(s)) for s in texts)
        record(update)
        return orig(self, sentences, *args, **kwargs)
    return encode


_wrap_once(AisuiteDriver, "complete", _sync_llm)
_wrap_once(AisuiteDriver, "acomplete", _async_llm)
_wrap_once(TypeSafeClient, "system_one", _sync_jev)
_wrap_once(AsyncTypeSafeClient, "system_one", _async_jev)
_wrap_once(AisuiteEmbeddingModel, "encode", _embed)

_notebook_create = client.chat.completions.create


def _metered_create(*args, **kwargs):
    response = _notebook_create(*args, **kwargs)
    note_llm(kwargs.get("model", "?"), response)
    return response


client.chat.completions.create = _metered_create

# ── BoK layout, read tools and explorer policy (cell 2) ──────────────────────────────────────────────────────
REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "swebok-v4").exists())
BOK_PATH = REPO_ROOT / "swebok-v4"
BOK_ROOT = BOK_PATH / "root"
READ_LIMIT = 4


def check_valid_file(file_path: str, start_line: int = None, end_line: int = None):
    if not os.path.isfile(file_path):
        if os.path.isdir(file_path):
            raise PermissionError(f"You need to read a file, not a path! "
                                  f"Available files/folders in '{file_path}': {os.listdir(file_path)}")
        raise PermissionError("You need to read a file, not a path!")


def check_multiple_files(paths):
    errors = []
    for path in paths:
        try:
            check_valid_file(path)
        except Exception as e:  # v2: report the error itself (the notebook formatted the flag, not the message)
            errors.append(f"Found an error for the file: {path} - {e}")
    if errors:
        raise PermissionError("\n".join(errors))


def inside_bok(paths):
    """v2: read_many reads only files of the BoK folder, so no architecture can open the benchmark's gold
    standard, other runs' logs or anything else in the repo."""
    outside = [p for p in paths if not Path(os.path.normpath(p)).resolve().is_relative_to(BOK_PATH.resolve())]
    if outside:
        raise PermissionError(f"Only files of the BoK can be read ({BOK_PATH}). Not allowed: {', '.join(outside)}")


def read_many(paths: list[str]) -> str:
    """Read multiple files at a time and returns the full text

    Args:
        paths (List[str]): list of paths to be read

    Returns:
        content_text (str): full content text with references
    """
    record(lambda rec: rec.read.extend(paths))
    content = []
    for path in paths:
        text = read_file(path)
        content.append(f'Here is the content for the file: {path}:\n{text}\n')
    return '\n'.join(content)


def plan_tasks(plan: str):
    """
    Use this tool to plan you approach
    plan: str. Your plan to solve the problem
    """
    return plan


class ReadBudget:
    def __init__(self, limit=4):
        self.limit = limit
        self.count = 0

    def check_read(self, paths):
        if not all(re.search(r'INDEX\.md|README\.md|STYLES\.md', file_path) is not None for file_path in paths):
            self.count += 1
        if self.count > self.limit:
            raise PermissionError(f"read_file is locked after {self.limit} reads. "
                                  "Now you must finish your task at once!")


class ReadOnce:
    def __init__(self):
        self.visited = set()

    def __call__(self, paths):
        visited_text = "\n".join(self.visited)
        if set(paths).issubset(self.visited):
            file_path = "\n".join(paths)
            raise PermissionError(f"You already read the files:\n{file_path}\n"
                                  f"Either finish the task or read other files outside this list:\n{visited_text}")
        self.visited.update(paths)


class ReadFirst:
    def __init__(self, files):
        self.files = set(files)
        self.visited = set()

    def check_first(self, paths):
        self.visited.update(paths)
        condition = self.files.issubset(self.visited)
        if len(self.visited) <= len(self.files):
            condition = self.visited.issubset(self.files)
        if not condition:
            raise PermissionError(f"You need to read the following files first: {', '.join(self.files)}!")


def get_policy(final_tool="write_file"):
    budget = ReadBudget(limit=READ_LIMIT)
    read_first = ReadFirst(files=[str(BOK_ROOT / "README.md"), str(BOK_ROOT / "INDEX.md")])
    policy = Policy().add_graph({
        "plan_tasks": ["read_many"],
        "read_many": ["read_many", "read_images", final_tool],
        "read_images": ["read_many", "read_images", final_tool],
        final_tool: [],
    })
    policy = policy.add_chain(start=["plan_tasks"], end=[final_tool])
    return policy.add({
        "read_many": [Clause(inside_bok), Clause(check_multiple_files), Clause(read_first.check_first),
                      Clause(ReadOnce), Clause(budget.check_read)],
    })


# ── images (cells 3-4) ───────────────────────────────────────────────────────────────────────────────────────
def local_image_as_b64(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        data = f.read()
    return f"data:{mime or 'image/png'};base64," + base64.b64encode(data).decode()


def read_images(filepath: str):
    """Read images from a given file path"""
    inside_bok([filepath])  # v2: same confinement as read_many
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image giving as much detail as possible."},
        {"type": "image_url", "image_url": {"url": local_image_as_b64(filepath)}},
    ]}]
    response = client.chat.completions.create(model=LLM_MODEL, messages=messages, max_tokens=20000)
    return response.choices[0].message.content.strip()


# ── vector search (cell 5) ───────────────────────────────────────────────────────────────────────────────────
SWEBOK_DB = REPO_ROOT / "swebok.duckdb"
SWEBOK_TABLE = "swebok_chunks"
SWEBOK_MODEL = "openrouter:qwen/qwen3-embedding-8b"
SWEBOK_K = 10
with duckdb.connect(str(SWEBOK_DB), read_only=True) as _con:
    SWEBOK_DIM = _con.execute(f"SELECT len(embedding) FROM {SWEBOK_TABLE} LIMIT 1").fetchone()[0]
swebok_embedder = AisuiteEmbeddingModel(SWEBOK_MODEL, batch_size=32, dimensions=SWEBOK_DIM)


def search_swebok(query: str) -> str:
    """Semantic search over the SWEBOK v4 (Guide to the Software Engineering Body of Knowledge).

    Use it to find the SWEBOK passages most relevant to a software engineering question.
    Write the query in English and be specific (e.g. "how to manage technical debt during maintenance").

    Args:
        query (str): natural-language question or topic to search for
    Returns:
        str: the top passages, ranked by relevance, each with its cosine distance (lower is closer)
    """
    q = np.asarray(swebok_embedder.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
    with duckdb.connect(str(SWEBOK_DB), read_only=True) as con:
        con.execute("INSTALL vss; LOAD vss;")
        rows = con.execute(
            f"""
            SELECT array_cosine_distance(embedding, ?::FLOAT[{SWEBOK_DIM}]) AS distance, chunk
            FROM {SWEBOK_TABLE}
            ORDER BY distance
            LIMIT ?
            """,
            [q.tolist(), SWEBOK_K],
        ).fetchall()
    record(lambda rec: rec.chunks.extend(chunk_id(chunk) for _, chunk in rows))
    return "\n\n".join(f"[{i}] (distance={d:.4f})\n{chunk}" for i, (d, chunk) in enumerate(rows, 1))


# ── explorer (cells 6-7) ─────────────────────────────────────────────────────────────────────────────────────
EXPLORER_PROMPT = """
You are a research assistant that consults a Body of Knowledge (BoK) on behalf of a requester. You answer the requester's question using ONLY what you actually read in the BoK.

## The BoK

Name: {bok_name}
Location: {bok_path}

Layout:
- {bok_root}/README.md: overview of the whole BoK.
- {bok_root}/INDEX.md: list of the top-level topics, with links to each topic folder.
- Every topic folder has:
  - INDEX.md: links to the items one level below (sub-folders or content files).
  - README.md: summary of what that folder covers.
- Content files are the other .md files. They are the leaves of the tree and the only real evidence.
- Tables are in {bok_path}/tables and images are in {bok_path}/images. Content files point to them as
  [Table N](tables/<file>.md) and ![image](images/<file>.png). These links are relative to {bok_path}, not to the content file.
  These folders have no INDEX.md or README.md.
- Links inside INDEX.md are relative to the folder of that INDEX.md. Always turn them into full paths
  starting with {bok_root} before reading.

Top-level topics:
{topics}

## Tools

- plan_tasks: write your search plan. Always call it first.
- read_many: read several files in one call. Group the reads of each tree level into one call.
- read_images: describe one image file.
- write_file: write your full answer to {result_path}.

## Rules

1. Closed source. Every claim in your answer must come from a BoK passage you actually read in this run.
   Your own knowledge may only help you pick search terms and branches. It may not fill in content.
2. Read budget. Batches that contain only INDEX.md and README.md files are free.
   Batches that contain content files are limited to {read_limit}. Use the indexes to choose before you read content.
3. Do not map the whole BoK. Go down only the branches that are relevant to the question.
4. Titles, index entries and file names are for navigation only. They are not evidence.
5. If a content file you read points to a relevant table or image, read it. Do not guess its content from its name.
6. Never invent paths. Only use paths that appear in an INDEX.md you read.
7. If the BoK does not cover a point, say so plainly ("Not found in the BoK: <point>"). Do not fill the gap.
8. If two passages disagree, show both with their sources.

## Procedure

1. Plan: call plan_tasks with a short plan. List the topics to look for, not the answer.
2. Discover: in one read_many, read {bok_root}/README.md and {bok_root}/INDEX.md.
3. Navigate: for each relevant topic, read its INDEX.md and README.md (one read_many per level) and go down
   until you reach the content files that answer the question.
4. Read: read only the content files (and their tables or images) you need.
5. Check: for each statement you plan to write, confirm that a passage you read supports it in this context.
   Remove anything that fails the check.
6. Write: save the full answer with write_file to {result_path}. It must contain:
   - Answer: the guidance, organized by the points of the question.
   - Sources: for each statement, the file path (relative to {bok_path}) and the section, plus a short quote when it helps.
   - Gaps: the points the BoK does not cover.

If a tool fails or a path does not exist, report it. Do not pretend you read or wrote something.

## Final reply to the requester

After writing the file, reply ONLY in this format:

# Summary
2 to 4 sentences answering the question.

# Guidelines
- Guideline supported by the BoK (source: relative/path.md, section)
- ...

# Gaps
- Points the BoK does not cover, or "None".

Full answer: {result_path}
"""
topics_block = "\n".join(f"- {d.name}" for d in sorted(BOK_ROOT.iterdir()) if d.is_dir())
# v2: the explorer's scratch answer goes to results/ (outside the BoK); the evaluation reads the turn's answer,
# not this file.
result_path = str(REPO_ROOT / "results" / "runs" / "explorer" / "BOK_RESULT.md")
Path(result_path).parent.mkdir(parents=True, exist_ok=True)
EXPLORER_PROMPT = EXPLORER_PROMPT.format(
    bok_name="SWEBOK v4 (Guide to the Software Engineering Body of Knowledge)",
    bok_path=BOK_PATH, bok_root=BOK_ROOT, topics=topics_block, read_limit=READ_LIMIT, result_path=result_path,
)


def make_explorer():
    return create_agent(role=EXPLORER_PROMPT, policy=get_policy(),
                        skills={"read_many": read_many, "read_images": read_images,
                                "plan_tasks": plan_tasks, "write_file": write_file},
                        **AGENT_KWARGS)


# ── Jev (cells 10-14) ────────────────────────────────────────────────────────────────────────────────────────
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.14")  # v2: pinned (the notebook used the moving alias jev-latest)
JEV_BASE_URL = "https://openrouter.ai/api"


def resolve_ref(path: str):
    for candidate in (Path(path), BOK_PATH / path, BOK_ROOT / path):
        if candidate.is_file():
            return str(candidate)
    return None


def check_refs(recommendations_json: str):
    try:
        recs = json.loads(recommendations_json)["recommendations"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PermissionError(f"recommendations_json does not follow the schema: {exc}")
    missing = sorted({ref for rec in recs for ref in rec.get("refs", []) if resolve_ref(ref) is None})
    if missing:
        raise PermissionError("These refs are not BoK files: " + "; ".join(missing)
                              + f". Use the full paths of the content files you read (they start with {BOK_ROOT}).")
    inside_bok([resolve_ref(ref) for rec in recs for ref in rec.get("refs", [])])  # v2


FILTER_MARK = "[JEV filter]"
ADMISSION_CRITERIA = {
    "lenient": {  # the repo's original wording (01_tests.ipynb, commit 4055694)
        "instructions": "Can the guideline be stated from what the references say?",
        "true": "The references support the statement, directly or indirectly",
        "false": "The statement has nothing to do with the references, it goes off topic",
    },
}
# v2: back to the repo's original filter (keep when P > 0.5). The 18/11 "strict" wording and 0.4 threshold were
# calibrated on guidelines that cite leaves of the benchmark's own gold standard.
ADMISSION_CRITERION = "lenient"
ADMISSION_THRESHOLD = 0.5


def admitted(score: float) -> bool:
    return score > ADMISSION_THRESHOLD  # the original: result.nouls["supported"].noul > 0.5


ADMISSION_LOG = REPO_ROOT / "results" / "demos" / "retrieval_v2" / "admission_log.md"


def score_guidelines(recs, criterion=None):
    spec = ADMISSION_CRITERIA[criterion or ADMISSION_CRITERION]
    question = Noul(instructions=spec["instructions"], criteria=NoulCriteria(true=spec["true"], false=spec["false"]))
    scores = []
    with TypeSafeClient(api_key=os.environ["OPENROUTER_API_KEY"], base_url=JEV_BASE_URL) as ts:
        for rec in recs:
            state = {"recommendation": rec["recommendation"],
                     "refs": read_many([resolve_ref(r) or r for r in rec["refs"]])}
            result = ts.system_one(model=JEV_MODEL, state=state, questions={"supported": question})
            scores.append(result.nouls["supported"].noul)
    return scores


def _sources(rec) -> str:
    out = []
    for ref in rec["refs"]:
        full = resolve_ref(ref)
        out.append(str(Path(full).relative_to(BOK_PATH)) if full else ref)
    return "; ".join(out)


def log_admission(recs, scores):
    ADMISSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    cell = lambda text: str(text).replace("|", "\\|").replace("\n", " ")
    kept = sum(admitted(s) for s in scores)
    rows = [f"| {i} | {s:.2f} | {'admitted' if admitted(s) else 'REJECTED'} | {cell(rec['recommendation'])} "
            f"| {cell(_sources(rec))} |" for i, (rec, s) in enumerate(zip(recs, scores), 1)]
    block = (f"\n## {datetime.datetime.now():%Y-%m-%d %H:%M:%S}: kept {kept} of {len(recs)} "
             f"(criterion '{ADMISSION_CRITERION}', score > {ADMISSION_THRESHOLD})\n\n"
             "| # | score | verdict | guideline | sources |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    with open(ADMISSION_LOG, "a", encoding="utf-8") as f:
        f.write(block)


def generate_final_recommendation(recommendations_json: str):
    """
    Generate the final guidelines.
    recommendations_json: the guidelines JSON with the schema:

    {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "title": "RecommendationsResponse",
      "type": "object",
      "properties": {
        "recommendations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "recommendation": {"type": "string", "description": "The guideline"},
              "refs": {"type": "array", "description": "Full paths of the BoK content files that support it",
                       "items": {"type": "string"}}
            },
            "required": ["recommendation", "refs"],
            "additionalProperties": false
          }
        }
      },
      "required": ["recommendations"],
      "additionalProperties": false
    }
    """
    recs = json.loads(recommendations_json)["recommendations"]
    scores = score_guidelines(recs)
    log_admission(recs, scores)
    record(lambda rec: rec.admission.extend(
        {"guideline": r["recommendation"], "refs": r["refs"], "score": s, "admitted": admitted(s)}
        for r, s in zip(recs, scores)))
    kept = [(rec, s) for rec, s in zip(recs, scores) if admitted(s)]
    lines = [f"{FILTER_MARK} kept {len(kept)} of {len(recs)} guidelines "
             f"(JEV '{ADMISSION_CRITERION}' score > {ADMISSION_THRESHOLD})", "", "# Guidelines"]
    for rec, s in kept:
        lines.append(f"- {rec['recommendation']} (source: {_sources(rec)}) [JEV {s:.2f}]")
    if not kept:
        lines.append("- None: no guideline is supported by the references it cites.")
    return "\n".join(lines)


JEV_FILTER_ROLE = EXPLORER_PROMPT + """
## Final step (replaces write_file and the final reply)

Do not call write_file. Finish by calling generate_final_recommendation with every guideline and the
full paths (starting with """ + str(BOK_ROOT) + """) of the BoK content files that support it.
It drops the guidelines the references do not support, and its output is delivered as your final
answer exactly as it is: you will not write a reply after it.
"""


def get_jev_filter_explorer():
    policy = get_policy(final_tool="generate_final_recommendation")
    policy.add({"generate_final_recommendation": [Clause(check_refs)]})
    policy.add_breaker("generate_final_recommendation")
    return create_agent(role=JEV_FILTER_ROLE, policy=policy,
                        skills={"read_many": read_many, "read_images": read_images, "plan_tasks": plan_tasks,
                                "generate_final_recommendation": generate_final_recommendation},
                        **AGENT_KWARGS)


class JevSearcher:
    """JEV's walk over the body of knowledge, as a tool: one call runs the whole pipeline."""

    LINK = re.compile(r'^- \[(.+?)\]\(<(.+?)>\)', re.M)
    TABLE = re.compile(r'\]\((tables/[^)]+\.md)\)')
    MAX_LABELS = 255

    def __init__(self, bok_path, k=10, min_p=0.0, threshold=0.3, concurrency=16, max_chars=60000, model=None):
        self.bok_path = Path(bok_path)
        self.k, self.min_p, self.threshold = k, min_p, threshold
        self.concurrency, self.max_chars, self.model = concurrency, max_chars, model or JEV_MODEL

    async def __call__(self, case: str, question: str, instructions: str,
                       true_criteria: str, false_criteria: str) -> str:
        """
        Search the body of knowledge for the documents that serve a task, without reading the tree yourself.

        One call runs the whole search. At every index of the body of knowledge, `question` picks the
        most promising branches; every document reached is judged by `instructions` with the true and
        false criteria; the approved documents come back, ranked by the judge's score, with their
        content. Be specific in both parts: a generic question prunes the wrong branches, and a generic
        criterion approves documents that change nothing in the task. For different aspects of the
        same task (design, testing, maintenance), run one search for each.

        Args:
            case: The full task, as JEV will see it at every decision.
            question: The question that picks the branches at every index, asked over the titles, e.g.:
                "Which of these branches most likely contain guidance on unit test design for this module?"
            instructions: The question the judge answers for every document, e.g.: "Does the document
                give practices or criteria for designing unit tests applicable to this task?"
            true_criteria: When the answer is yes: what the document must bring.
            false_criteria: When the answer is no: what disqualifies the document.
        """
        fields = {"case": case, "question": question, "instructions": instructions,
                  "true_criteria": true_criteria, "false_criteria": false_criteria}
        empty = [name for name, value in fields.items() if not value.strip()]
        if empty:
            return f"Error: {', '.join(empty)} empty. The five fields are required."
        judge = {"type": "noul", "instructions": instructions,
                 "criteria": {"true": true_criteria, "false": false_criteria}}
        async with AsyncTypeSafeClient(model=self.model, api_key=os.environ["OPENROUTER_API_KEY"],
                                       base_url=JEV_BASE_URL) as ts:
            judged = await self._walk(ts, case, question, judge)
        return self._report(judged)

    def _list_children(self, index_path: Path) -> dict:
        text = index_path.read_text(encoding="utf-8")
        return {t: index_path.parent / p for t, p in self.LINK.findall(text) if (index_path.parent / p).exists()}

    def _describe(self, target: Path):
        return "\n".join(self._list_children(target)) if target.name == "INDEX.md" else None

    def _load_leaf(self, path: Path) -> str:
        topic_dir = next(p for p in path.parents if p.name == "root").parent
        text = path.read_text(encoding="utf-8")
        tables = [topic_dir / t for t in dict.fromkeys(self.TABLE.findall(text))]
        return text + "".join(f"\n\n## {t.stem}\n{t.read_text(encoding='utf-8')}" for t in tables if t.exists())

    def _start(self):
        if (self.bok_path / "root" / "INDEX.md").exists():
            start = self._list_children(self.bok_path / "root" / "INDEX.md")
        else:
            start = {d.name: d / "root" / "INDEX.md" for d in self.bok_path.iterdir()
                     if (d / "root" / "INDEX.md").exists()}
        if not start:
            raise FileNotFoundError(f"No root/INDEX.md under {self.bok_path}")
        return start

    async def _top_k(self, ts, sem, case, question, criteria):
        k, min_p = self.k, self.min_p
        if len(criteria) <= k and not min_p:
            return list(criteria)
        if len(criteria) > self.MAX_LABELS:
            labels = list(criteria)
            chunks = [{l: criteria[l] for l in labels[i:i + self.MAX_LABELS]}
                      for i in range(0, len(labels), self.MAX_LABELS)]
            winners = await asyncio.gather(*(self._top_k(ts, sem, case, question, c) for c in chunks))
            return await self._top_k(ts, sem, case, question, {l: criteria[l] for w in winners for l in w})
        async with sem:
            res = await ts.system_one(state=case,
                                      questions={"pick": {"type": "choice", "instructions": question,
                                                          "criteria": criteria}})
        probs = res.choices["pick"].probabilities
        return [l for l in sorted(probs, key=probs.get, reverse=True)[:k] if probs[l] >= min_p]

    async def _walk(self, ts, case, question, judge):
        sem = asyncio.Semaphore(self.concurrency)
        judged = []

        async def judge_leaf(path):
            content = self._load_leaf(path)
            async with sem:
                res = await ts.system_one(state={"task": case, "document": content}, questions={"useful": judge})
            return path, content, res.nouls["useful"].noul

        async def descend(children):
            criteria = {t: self._describe(p) for t, p in children.items()}
            picked = [children[t] for t in await self._top_k(ts, sem, case, question, criteria)]
            judged.extend(asyncio.create_task(judge_leaf(p)) for p in picked if p.name != "INDEX.md")
            await asyncio.gather(*(descend(self._list_children(p)) for p in picked if p.name == "INDEX.md"))

        await descend(self._start())
        return await asyncio.gather(*judged)

    def _report(self, judged):
        judged = sorted(judged, key=lambda d: d[2], reverse=True)
        passed = [d for d in judged if d[2] >= self.threshold]
        head = f"{len(judged)} documents judged, {len(passed)} approved (score >= {self.threshold})."
        if not passed:
            near = "\n".join(f"- [{s:.2f}] {p}" for p, _, s in judged[:5])
            return (f"{head} No document passed. The closest ones:\n{near}\n"
                    "Search again with a question or criterion closer to what these documents cover.")
        blocks, used, cut = [], 0, 0
        for path, content, score in passed:
            block = f"## [{score:.2f}] {path}\n{content}"
            if used + len(block) > self.max_chars and blocks:
                cut += 1
                continue
            blocks.append(block)
            used += len(block)
        delivered = [block.split("\n", 1)[0].split("] ", 1)[1] for block in blocks]
        record(lambda rec: rec.jev_approved.extend(delivered))
        tail = f" {cut} approved documents left out by the {self.max_chars}-character cap." if cut else ""
        return head + tail + "\n\n" + "\n\n".join(blocks)


CHOOSE = "Which of these branches of the body of knowledge most likely contain guidance for this task?"
USEFUL = {
    "instructions": "Does the document contain information useful to carry out this task?",
    "true_criteria": "The document brings practices, criteria or data directly applicable to the task",
    "false_criteria": "The document covers another subject or changes nothing in how the task is done",
}
RECOMMEND_PROMPT = """
You are a exploration tool. Based ONLY on the references given, write the guidelines for the task
in this format:

# Summary

Up to 500 characters.

# Guidelines

- guideline (source: relative/path.md)
- ...

Use only the text you receive. Never add guidelines from outside it.
"""


async def jev_explore(task: str):
    """JEV walks the tree with the fixed question and judge; the approved documents go to the LLM."""
    references = await JevSearcher(BOK_PATH)(task, CHOOSE, **USEFUL)
    messages = [{"role": "system", "content": RECOMMEND_PROMPT},
                {"role": "user", "content": f"Task:\n{task}\n\nReferences:\n{references}"}]
    response = client.chat.completions.create(model=LLM_MODEL, messages=messages, max_tokens=10000)
    return response.choices[0].message.content.strip()


class CallBudget:
    """At most `limit` calls of the tool it guards, per question (pass the class to Clause, not an instance)."""
    def __init__(self, limit=1):
        self.limit = limit
        self.count = 0

    def __call__(self, **kwargs):
        self.count += 1
        if self.count > self.limit:
            raise PermissionError("Call limit reached! Now you must finish your task at once!")


# v2: the same closed-source instruction for every architecture. The explorer's prompt (rule 1) and the JEV summary
# (RECOMMEND_PROMPT) already had it; the vector-search agent and the JEV-free agent did not.
CLOSED_SOURCE = (" Use only what you read in the SWEBOK in this run; do not add guidance from your own knowledge. "
                 "If the SWEBOK passages you read do not cover a point, say so.")

JEV_CODER_ROLE = ("You are a searcher agent that writes practices following the SWEBOK. To do so, you use the "
                  "`jev_search` tool. To use it properly, ask it broad questions and define mutually exclusive and "
                  "non-ambiguous acceptance criteria." + CLOSED_SOURCE)
JEV_FREE_LIMIT = 4  # v2: the same budget as JEV (READ_LIMIT calls of jev_explore); the notebook had 1


def get_jev_explorer_free(limit=JEV_FREE_LIMIT):
    return create_agent(role=JEV_CODER_ROLE,
                        policy=Policy().add({"jev_search": [Clause(CallBudget, config={"limit": limit})]}),
                        skills={"jev_search": JevSearcher(BOK_PATH)}, **AGENT_KWARGS)


# ── one turn, logged (cell 16) ───────────────────────────────────────────────────────────────────────────────
DEMO_DIR = REPO_ROOT / "results" / "demos"


class SavingDemo(AgentDemo):
    """AgentDemo that appends every turn (question, tool calls, answer) to results/demos/<name>.md|.json."""

    def __init__(self, agent, name: str, max_tool_chars: int = 1500, **kwargs):
        super().__init__(agent, **kwargs)
        self.path = DEMO_DIR / f"{name}.md"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_tool_chars = max_tool_chars
        self._events, self._ev_lock = [], threading.Lock()
        publish = self._hub.publish

        def recording_publish(event):
            with self._ev_lock:
                self._events.append(event)
            publish(event)

        self._hub.publish = recording_publish

    def _cut(self, text) -> str:
        text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2)
        if self.max_tool_chars and len(text) > self.max_tool_chars:
            return text[:self.max_tool_chars] + f"\n... [+{len(text) - self.max_tool_chars} chars]"
        return text

    def _run_turn(self, message: str) -> None:
        with self._ev_lock:
            self._events = []
        start = datetime.datetime.now()
        recorder = TurnRecorder()
        token = RECORDER.set(recorder)
        try:
            super()._run_turn(message)
        finally:
            RECORDER.reset(token)
            with self._ev_lock:
                events = list(self._events)
            self._save(message, events, start, recorder)

    def _save(self, message, events, start, recorder=None):
        seconds = (datetime.datetime.now() - start).total_seconds()
        tools, answer = [], None
        for ev in events:
            if ev["type"] == "tool_use":
                tools.append(f"#### `{ev['tool_name']}` ({ev['tool_id']})\n\n```\n{self._cut(ev['input'])}\n```\n")
            elif ev["type"] == "tool_result":
                tools.append(f"Result ({ev['tool_use_id']}):\n\n```\n{self._cut(ev['content'])}\n```\n")
            elif ev["type"] == "assistant":
                answer = ev["text"]
        n_calls = sum(ev["type"] == "tool_use" for ev in events)
        if answer is None:
            answer = "_(no answer: the turn was stopped or failed; see the demo)_"
        block = (f"\n---\n\n## {start:%Y-%m-%d %H:%M:%S} ({seconds:.0f}s, {n_calls} tool calls)\n\n"
                 f"### Question\n\n{message}\n\n### Answer\n\n{answer}\n\n"
                 "<details><summary>Tool calls</summary>\n\n" + "\n".join(tools) + "\n</details>\n")
        usage = recorder.to_dict() if recorder is not None else {}
        if usage:
            llm = "; ".join(f"{m}: {d['calls']} calls, {d['input_tokens']} in / {d['output_tokens']} out tokens"
                            for m, d in usage["llm"].items()) or "none"
            block += (f"\n**Usage:** LLM {llm}. JEV calls: {usage['jev_calls']}. Embedding calls: "
                      f"{usage['embedding_calls']}. Files read: {len(set(usage['read']))}; JEV-approved: "
                      f"{len(set(usage['jev_approved']))}; vector chunks: {len(set(usage['chunks']))}.\n")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(block)
        tool_counts = {}
        for ev in events:
            if ev["type"] == "tool_use":
                tool_counts[ev["tool_name"]] = tool_counts.get(ev["tool_name"], 0) + 1
        sidecar = self.path.with_suffix(".json")
        turns = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else []
        turns.append({"start": start.isoformat(), "seconds": round(seconds, 1), "question": message,
                      "answer": answer, "tool_calls": n_calls, "tools": tool_counts,
                      "jev_model": JEV_MODEL, "llm_model": LLM_MODEL, **usage})
        sidecar.write_text(json.dumps(turns, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved -> {self.path.relative_to(REPO_ROOT)}", flush=True)


# ── retrieval experiment (cell 29) ───────────────────────────────────────────────────────────────────────────
RETRIEVAL_DIR = REPO_ROOT / "experiments" / "benchmark" / "retrieval"
NAIVE_RETRIEVAL_ROLE = ("You answer software engineering questions based on SWEBOK. "
                        "To check swebok, use the appropriate tool to give you the guidelines." + CLOSED_SOURCE)
# v2: control without any BoK access — what the model alone knows of the SWEBOK.
PURE_ROLE = ("You answer software engineering questions based on the SWEBOK Guide (Guide to the Software "
             "Engineering Body of Knowledge), version 4. You have no tools: answer from what you know of the Guide.")


class JevExploreAgent:
    """Demo 3's retrieval tool (jev_explore) behind the agent interface SavingDemo expects."""

    async def answer(self, task, tracer=None, is_async=True):
        return await jev_explore(task)


def retrieval_agents():
    """Fresh retrieval components, one per architecture (the notebook's demo names are kept)."""
    return {
        "0_pure_model": create_agent(role=PURE_ROLE, skills={}, **AGENT_KWARGS),
        "1_coder_explorer": make_explorer(),
        "2_coder_naive": create_agent(role=NAIVE_RETRIEVAL_ROLE, skills={"search_swebok": search_swebok}, **AGENT_KWARGS),
        "3_coder_jev_explore": JevExploreAgent(),
        "4_searcher_jev_search": get_jev_explorer_free(),
        "5_coder_explorer_jev_filter": get_jev_filter_explorer(),
    }


DEMOS = ["0_pure_model", "1_coder_explorer", "2_coder_naive", "3_coder_jev_explore", "4_searcher_jev_search",
         "5_coder_explorer_jev_filter"]


NO_ANSWER = "_(no answer"


def _finished(out: Path) -> bool:
    if not out.exists():
        return False
    turn = json.loads(out.read_text(encoding="utf-8"))[-1]
    return bool(turn.get("answer")) and not turn["answer"].startswith(NO_ANSWER)


def run_info(run_id, questions_file, spec, workers, names):
    import platform
    import subprocess
    from importlib import metadata
    raw = (RETRIEVAL_DIR / questions_file).read_bytes()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    return {"run": run_id, "questions_file": questions_file, "questions_sha256": hashlib.sha256(raw).hexdigest(),
            "n_questions": len(spec["questions"]), "architectures": names, "workers": workers,
            "llm_model": LLM_MODEL, "jev_model": JEV_MODEL, "embedding_model": SWEBOK_MODEL,
            "admission": {"criterion": ADMISSION_CRITERION, "rule": f"score > {ADMISSION_THRESHOLD}"},
            "jev_free_limit": JEV_FREE_LIMIT, "explorer_read_limit": READ_LIMIT, "jev_explore": "no call limit in "
            "retrieval mode (called once per question)", "git_commit": commit, "python": platform.python_version(),
            "packages": {p: metadata.version(p) for p in ("fast_track_lib", "typesafe-sdk", "aisuite", "duckdb")}}


def run_retrieval(run_id, questions_file, workers=6, only=None, demos=None):
    spec = json.loads((RETRIEVAL_DIR / questions_file).read_text(encoding="utf-8"))
    chosen = [q for q in spec["questions"] if only is None or q["id"] in only]
    names = demos or DEMOS
    run_dir = DEMO_DIR / "retrieval_v2" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    info = run_info(run_id, questions_file, spec, workers, names)
    info["started"] = datetime.datetime.now().isoformat()
    (run_dir / "questions_used.json").write_bytes((RETRIEVAL_DIR / questions_file).read_bytes())

    def one(job):
        q, name = job
        base = run_dir / name / q["id"]
        out = base.with_suffix(".json")
        if _finished(out):  # resume: a finished turn is never run twice
            return name, q["id"], "skip"
        for stale in (out, base.with_suffix(".md")):  # a failed turn is run again from scratch
            if stale.exists():
                stale.rename(stale.with_name(stale.name + f".failed-{datetime.datetime.now():%H%M%S}"))
        # v2: keep the full tool inputs and results (max_tool_chars=None) for the public traces
        demo = SavingDemo(retrieval_agents()[name], name=f"retrieval_v2/{run_id}/{name}/{q['id']}",
                          max_tool_chars=None)
        demo._hub.turn_lock.acquire()  # what the web server does before a turn
        demo._run_turn(q["question"] + "\n\n" + spec["answer_format"])
        return name, q["id"], "done" if _finished(out) else "FAILED"

    jobs = [(q, name) for q in chosen for name in names]
    for attempt in (1, 2):  # second pass only for the turns that failed
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for (q, name), (_, qid, status) in zip(jobs, pool.map(one, jobs)):
                print(f"{run_id} [pass {attempt}]: {name} {qid} {status}", flush=True)
                if status == "FAILED":
                    failed.append((q, name))
        if not failed:
            break
        jobs = failed
    info["finished"] = datetime.datetime.now().isoformat()
    info["failed_after_retry"] = [f"{name}/{q['id']}" for q, name in failed]
    (run_dir / "RUN_INFO.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions-file", default="questions.json")
    ap.add_argument("--questions", nargs="*", default=None, help="question ids (default: all)")
    ap.add_argument("--demos", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run_retrieval(a.run, a.questions_file, workers=a.workers, only=a.questions, demos=a.demos)
