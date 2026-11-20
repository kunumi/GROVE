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
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "retrieval"))
from common import REPO, parse_citations, retrieved_leaves, to_leaf  # noqa: E402

LOGS = REPO / "results" / "demos" / "retrieval_v2"
DEMOS = ["0_pure_model", "1_coder_explorer", "2_coder_naive", "3_coder_jev_explore", "4_searcher_jev_search",
         "5_coder_explorer_jev_filter", "6_full_context"]
LABELS = {"0_pure_model": "Pure model (no BoK)", "1_coder_explorer": "Explorer", "2_coder_naive": "Agentic RAG",
          "3_coder_jev_explore": "JEV", "4_searcher_jev_search": "JEV-free", "5_coder_explorer_jev_filter": "JEV filter",
          "6_full_context": "Full context"}
FULL = "6_full_context"
FULL_CONTEXT_MIN_TOKENS = 150_000  # below this, the BoK was not in the prompt

# US$ per token (prices checked 19/11/2026: Anthropic and OpenRouter price lists)
PRICE = {"llm_in": 3 / 1e6, "llm_out": 15 / 1e6, "jev_in": 0.042 / 1e6, "emb_in": 0.01 / 1e6,
         # Sonnet 4.6 prompt caching (5-minute writes; no long-context surcharge on 4.6)
         "llm_cache_read": 0.30 / 1e6, "llm_cache_write": 3.75 / 1e6}
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
            "5_coder_explorer_jev_filter": not answer.startswith(FILTER_MARK),
            FULL: sum(d["input_tokens"] for d in turn.get("llm", {}).values()) < FULL_CONTEXT_MIN_TOKENS,
            }.get(demo, False)


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
    # what was actually billed: cache hits and writes are inside tin (full context only; 0 elsewhere)
    hit, write = turn.get("cache_read_tokens", 0), turn.get("cache_creation_tokens", 0)
    cost_cached = cost - (hit + write) * PRICE["llm_in"] + hit * PRICE["llm_cache_read"] + write * PRICE["llm_cache_write"]
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
        "cost_cached_usd": cost_cached,
        "cache_read_tokens": hit,
        "cited_leaves": sorted(cited),
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
         ("embedding_calls", "embedding calls", "num"),
         ("cost_cached_usd", "cost actually billed, with prompt caching (US$)", "usd"),
         ("cache_read_tokens", "LLM input tokens read from the prompt cache", "num")]


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


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def full_context_sections(rows, questions, runs, logs) -> list:
    """Footnote on its citations, the paired comparison with JEV and the position (lost-in-the-middle) analysis."""
    out = ["", f"Full context: the whole BoK is in the prompt, so every citation that exists is \"valid\" by "
           "construction, and \"expected leaves retrieved\" is 100% by construction. Its time is not comparable: "
           "it ran in a separate session about 14 hours later, without the other architectures running alongside."]

    # paired, per question: mean over runs of expected leaves cited
    def per_question(demo):
        return {q["id"]: statistics.mean([r["expected_cited"] for r in rows if r["demo"] == demo
                                          and r["question"] == q["id"]] or [float("nan")]) for q in questions}
    full = per_question(FULL)
    out += ["", "## Full context against the other architectures, paired by question", "",
            "Expected leaves cited, mean over runs per question; exact two-sided sign test, ties dropped.", "",
            "| compared with | Full context better | tie | Full context worse | p (sign test) |", "|---|---|---|---|---|"]
    for other in [d for d in DEMOS if d not in (FULL, "0_pure_model") and any(r["demo"] == d for r in rows)]:
        theirs = per_question(other)
        diffs = [full[q] - theirs[q] for q in full if not (math.isnan(full[q]) or math.isnan(theirs[q]))]
        w, l = sum(d > 1e-9 for d in diffs), sum(d < -1e-9 for d in diffs)
        out.append(f"| {LABELS[other]} | {w} | {len(diffs) - w - l} | {l} | {sign_test(w, l):.3f} |")

    # position of each gold leaf in the full context
    layout_file = next((logs / run / "full_context_layout.json" for run in runs
                        if (logs / run / "full_context_layout.json").exists()), None)
    if layout_file is None:
        return out + ["", "(no full_context_layout.json: position analysis skipped)"]
    layout = json.loads(layout_file.read_text(encoding="utf-8"))
    total = layout["chars"]
    where = {x["path"]: (x["start"] + x["end"]) / 2 / total for x in layout["layout"]}
    compare = [d for d in (FULL, "3_coder_jev_explore", "2_coder_naive") if any(r["demo"] == d for r in rows)]
    bins = 5
    counts = {d: [[0, 0] for _ in range(bins)] for d in compare}  # [cited, total] per bin
    leaves_per_bin = [set() for _ in range(bins)]
    for q in questions:
        for leaf in q["expected_leaves"]:
            if leaf not in where:
                continue
            b = min(int(where[leaf] * bins), bins - 1)
            leaves_per_bin[b].add(leaf)
            for d in compare:
                for r in rows:
                    if r["demo"] == d and r["question"] == q["id"]:
                        counts[d][b][0] += leaf in r["cited_leaves"]
                        counts[d][b][1] += 1
    out += ["", "## Expected leaves cited, by the leaf's position in the full context", "",
            f"Position = middle of the leaf in the {total:,}-character context ({layout['leaves']} leaves in the "
            "Guide's order), in fifths. Each cell: share of (gold leaf, run) pairs where the answer cites the leaf. "
            "JEV and Agentic RAG do not see the context; they are the control for how hard the leaves of each fifth "
            "are.", "",
            "| position | gold leaves | " + " | ".join(LABELS[d] for d in compare) + " |",
            "|---|---|" + "---|" * len(compare)]
    for b in range(bins):
        cells = [f"{c / n:.0%} ({c}/{n})" if n else "—" for c, n in (counts[d][b] for d in compare)]
        out.append(f"| {b * 100 // bins}–{(b + 1) * 100 // bins}% | {len(leaves_per_bin[b])} | " + " | ".join(cells) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-file", default="questions.json")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="AVALIACAO_retrieval_v2")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--questions-dir", default=str(REPO / "experiments" / "benchmark" / "retrieval"))
    ap.add_argument("--logs", default=str(LOGS), help="folder with one sub-folder per run")
    ap.add_argument("--out-dir", default=str(REPO / "results"))
    ap.add_argument("--demos", nargs="*", default=None, help="architectures to evaluate (default: all)")
    args = ap.parse_args()
    logs = Path(args.logs).resolve()
    demos = args.demos or DEMOS
    questions = json.loads((Path(args.questions_dir) / args.questions_file).read_text(encoding="utf-8"))["questions"]
    rows, missing = [], []
    for run in args.runs:
        for demo in demos:
            for q in questions:
                sidecar = logs / run / demo / f"{q['id']}.json"
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
    present = [d for d in demos if any(r["demo"] == d for r in rows)]
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
    if FULL in present:
        out += full_context_sections(rows, questions, args.runs, logs)
    if missing:
        out += ["", "Missing turns: " + ", ".join(missing)]
    md = Path(args.out_dir) / f"{args.out}.md"
    md.write_text("\n".join(out) + "\n", encoding="utf-8")
    keys = ["run", "demo", "question", "answered", "degraded", "no_citations"] + [k for k, _, _ in TABLE + EXTRA]
    with open(md.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n".join(out))


if __name__ == "__main__":
    main()
