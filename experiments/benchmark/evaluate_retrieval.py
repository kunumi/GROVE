"""Retrieval experiment v2: evaluator.

    docker compose run --rm app python experiments/benchmark/evaluate_retrieval_v2.py --questions-file questions_v2.json --runs r1 r2 r3

Reads results/demos/retrieval_v2/<run>/<demo>/<question>.json (written by experiments/run_retrieval.py) and the gold
leaves of experiments/benchmark/retrieval/<questions-file>. Citation parsing and leaf resolution are those of
retrieval/common.py, unchanged. Metrics, per architecture, averaged over questions within a run, then over runs (mean ± sd):

- expected leaves cited: gold leaves the answer cites / gold leaves of the question;
- valid citations: guidelines whose source resolves to a real SWEBOK leaf AND that leaf was brought into the context
  in that same turn (read, approved by Jev, or returned by the vector search). A citation to a leaf the architecture
  never retrieved comes from the model's memory. For the pure model nothing is retrieved, so only "exists" is shown;
- time (s), LLM calls, Jev calls;
- cost (US$): LLM tokens at the Sonnet 4.6 price, Jev input tokens as reported by the API at the Jev 1.14
  price, embeddings estimated the same way. Input tokens are priced as uncached (an upper bound).

Writes results/AVALIACAO_retrieval_v2.md and .csv. No LLM or Jev call is made here.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "retrieval"))
from common import REPO, parse_citations, retrieved_leaves, to_leaf  # noqa: E402

LOGS = REPO / "results" / "demos" / "retrieval_v2"
DEMOS = ["0_pure_model", "1_coder_explorer", "2_coder_naive", "3_coder_jev_explore", "4_searcher_jev_search",
         "5_coder_explorer_jev_filter"]
LABELS = {"0_pure_model": "Pure model (no BoK)", "1_coder_explorer": "Explorer", "2_coder_naive": "Agentic RAG",
          "3_coder_jev_explore": "JEV", "4_searcher_jev_search": "JEV-free", "5_coder_explorer_jev_filter": "JEV filter"}

# US$ per token (prices checked 19/11/2026: Anthropic and OpenRouter price lists)
PRICE = {"llm_in": 3 / 1e6, "llm_out": 15 / 1e6, "jev_in": 0.042 / 1e6, "emb_in": 0.01 / 1e6}
CHARS_PER_TOKEN = 4


LLM_MODEL = "anthropic:claude-sonnet-4-6"
FILTER_MARK = "[JEV filter]"
AMBIGUOUS = 3  # a citation that resolves to more than this many leaves is counted as ambiguous


def agent_reads(turn: dict, demo: str) -> list:
    """The paths the architecture itself read. In the JEV filter, generate_final_recommendation re-reads every ref it
    judges through read_many, after the answer was written; those reads (the tail of `read`, the filter being the last
    step) are not the explorer's, so they are removed here."""
    read = list(turn.get("read", []))
    if demo != "5_coder_explorer_jev_filter" or not turn.get("admission"):
        return read
    tail = sum(len(a["refs"]) for a in turn["admission"])
    head, judged = read[:len(read) - tail], read[len(read) - tail:]
    refs = [r for a in turn["admission"] for r in a["refs"]]
    assert [to_leaf(x) for x in judged] == [to_leaf(r) for r in refs], "JEV filter: tail of read != judged refs"
    return head


def degraded(turn: dict, demo: str) -> bool:
    """The architecture did not run its BoK access (the LLM may still have answered)."""
    answer = turn.get("answer") or ""
    return {"1_coder_explorer": not turn.get("read"),
            "2_coder_naive": not turn.get("embedding_calls"),
            "3_coder_jev_explore": not turn.get("jev_calls"),
            "4_searcher_jev_search": not turn.get("jev_calls"),
            "5_coder_explorer_jev_filter": not answer.startswith(FILTER_MARK)}.get(demo, False)


def evaluate_turn(turn: dict, expected: set, demo: str) -> dict:
    assert expected, "a question without gold leaves"
    llm_models = set(turn.get("llm", {}))
    assert llm_models <= {LLM_MODEL}, f"unexpected LLM models: {llm_models}"
    if "jev_model" in turn:  # v2 runs
        assert turn["jev_model"] == "jev-1.14", turn["jev_model"]
        assert all(m.startswith("typesafe/jev-1.14") for m in turn.get("jev_models", [])), turn.get("jev_models")
        assert not turn.get("jev_calls") or turn.get("jev_input_tokens", 0) > 0, "Jev calls without token usage"
    cites = parse_citations(turn.get("answer") or "")
    cited = {leaf for c in cites for leaf in c["leaves"]}
    # what the architecture brought into context in this turn (cited leaves are NOT added)
    retrieved = retrieved_leaves({**turn, "read": agent_reads(turn, demo)})
    pure = demo == "0_pure_model"
    llm = turn.get("llm", {})
    tin = sum(d["input_tokens"] for d in llm.values())
    tout = sum(d["output_tokens"] for d in llm.values())
    # Jev input tokens as reported by the API (v2 runs); older logs only have the characters sent
    jev_tokens = turn["jev_input_tokens"] if "jev_input_tokens" in turn else turn.get("jev_chars", 0) / CHARS_PER_TOKEN
    emb_tokens = turn.get("embedding_chars", 0) / CHARS_PER_TOKEN
    cost = tin * PRICE["llm_in"] + tout * PRICE["llm_out"] + jev_tokens * PRICE["jev_in"] + emb_tokens * PRICE["emb_in"]
    exists = [c for c in cites if c["leaves"]]
    valid = [c for c in exists if set(c["leaves"]) & retrieved]  # one retrieved leaf is enough
    return {
        "answered": bool(turn.get("answer")) and not (turn.get("answer") or "").startswith("_(no answer"),
        "degraded": degraded(turn, demo),
        "no_citations": not cites,
        "guidelines": len(cites),
        "expected_cited": len(cited & expected) / len(expected),
        "expected_retrieved": None if pure else len(retrieved & expected) / len(expected),
        "leaves_retrieved": None if pure else len(retrieved),
        # a turn without any citation has no citation to judge: left out of the mean (and counted apart)
        "citations_exist": len(exists) / len(cites) if cites else None,
        "citations_valid": None if pure else (len(valid) / len(cites) if cites else None),
        "citations_ambiguous": (sum(len(c["leaves"]) > AMBIGUOUS for c in cites) / len(cites)) if cites else None,
        "seconds": turn.get("seconds", 0),
        "llm_calls": sum(d["calls"] for d in llm.values()),
        "input_tokens": tin,
        "output_tokens": tout,
        "jev_calls": turn.get("jev_calls", 0),
        "jev_tokens_est": jev_tokens,
        "embedding_calls": turn.get("embedding_calls", 0),
        "cost_usd": cost,
    }


TABLE = [("expected_cited", "expected leaves cited", "pct"), ("citations_valid", "valid citations", "pct"),
         ("seconds", "time (s)", "num"), ("llm_calls", "LLM calls", "num"), ("jev_calls", "Jev calls", "num"),
         ("cost_usd", "cost (US$)", "usd")]
EXTRA = [("citations_exist", "citations that exist in the SWEBOK", "pct"),
         ("citations_ambiguous", f"citations resolving to more than {AMBIGUOUS} leaves", "pct"),
         ("leaves_retrieved", "distinct leaves brought into context", "num"),
         ("expected_retrieved", "expected leaves retrieved (not comparable across architectures)", "pct"),
         ("guidelines", "guidelines with a source", "num"), ("input_tokens", "LLM input tokens", "num"),
         ("output_tokens", "LLM output tokens", "num"), ("jev_tokens_est", "Jev input tokens", "num"),
         ("embedding_calls", "embedding calls", "num")]


def fmt(values, kind):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    if kind == "pct":
        s = f"{m:.0%}" + (f" ± {sd:.0%}" if sd is not None else "")
    elif kind == "usd":
        s = f"{m:.3f}" + (f" ± {sd:.3f}" if sd is not None else "")
    else:
        s = (f"{m:,.0f}" if m >= 100 else f"{m:.1f}") + (f" ± {sd:,.1f}" if sd is not None else "")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-file", default="questions.json")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="AVALIACAO_retrieval_v2")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    questions = json.loads((REPO / "experiments" / "benchmark" / "retrieval" / args.questions_file)
                           .read_text(encoding="utf-8"))["questions"]
    rows, missing = [], []
    for run in args.runs:
        for demo in DEMOS:
            for q in questions:
                sidecar = LOGS / run / demo / f"{q['id']}.json"
                if not sidecar.exists():
                    missing.append(f"{run}/{demo}/{q['id']}")
                    continue
                turn = json.loads(sidecar.read_text(encoding="utf-8"))[-1]
                row = evaluate_turn(turn, set(q["expected_leaves"]), demo)
                row.update(run=run, demo=demo, question=q["id"])
                rows.append(row)

    def per_run_means(demo, key):
        """One value per run: the mean over that run's questions."""
        out = []
        for run in args.runs:
            vals = [r[key] for r in rows if r["demo"] == demo and r["run"] == run and r[key] is not None]
            if vals:
                out.append(statistics.mean(vals))
        return out

    if missing and not args.allow_missing:
        sys.exit(f"{len(missing)} turns missing (the comparison must be paired); e.g. {missing[:5]}. "
                 "Use --allow-missing only for a partial look.")
    present = [d for d in DEMOS if any(r["demo"] == d for r in rows)]
    out = ["# Retrieval experiment v2", "",
           f"Runs: {', '.join(args.runs)}. Questions: {len(questions)} (`{args.questions_file}`). Turns evaluated: "
           f"{len(rows)}; missing: {len(missing)}.", "",
           "Per architecture: mean over the questions of each run, then mean ± sd over runs.", "",
           "| metric | " + " | ".join(LABELS[d] for d in present) + " |", "|---|" + "---|" * len(present)]
    for key, label, kind in TABLE + EXTRA:
        cells = []
        for d in present:
            vals = per_run_means(d, key)
            cells.append(fmt(vals, kind) if vals else "—")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    for flag, label in (("answered", "turns without an answer"), ("degraded", "turns where the BoK access did not run"),
                        ("no_citations", "turns without any citation")):
        out.append(f"| {label} | " + " | ".join(
            str(sum((not r[flag]) if flag == "answered" else r[flag] for r in rows if r["demo"] == d)) for d in present)
            + " |")
    out += ["", "## Expected leaves cited, per question (mean over runs)", "",
            "| question | gold leaves | " + " | ".join(LABELS[d] for d in present) + " |",
            "|---|---|" + "---|" * len(present)]
    for q in questions:
        cells = []
        for d in present:
            vals = [r["expected_cited"] for r in rows if r["demo"] == d and r["question"] == q["id"]]
            cells.append(f"{statistics.mean(vals):.0%}" if vals else "—")
        out.append(f"| {q['id']} | {len(q['expected_leaves'])} | " + " | ".join(cells) + " |")
    if missing:
        out += ["", "Missing turns: " + ", ".join(missing)]
    md = REPO / "results" / f"{args.out}.md"
    md.write_text("\n".join(out) + "\n", encoding="utf-8")
    keys = ["run", "demo", "question", "answered", "degraded", "no_citations"] + [k for k, _, _ in TABLE + EXTRA]
    with open(md.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n".join(out))


if __name__ == "__main__":
    main()
