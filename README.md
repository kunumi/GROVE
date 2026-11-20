# GROVE

Research artifact for *GROVE: Enabling Grounded Knowledge in LLMs with Guided Retrieval Over Verified Evidence*.

GROVE separates evidence selection from writing. Given a task, **Jev** (the decision model) chooses branches in a tree of the SWEBOK Guide and judges which passages are useful. A single call to the language model turns the approved passages into guidelines with source citations.

The evaluation compares **GROVE** with four baselines (Agentic RAG, Explorer, Explorer + Check and GROVE-Free) and a **closed-book control** that has no tools or Guide access. Each configuration answered the same 25 questions in three rounds, giving **450 answers** across the Guide’s 18 knowledge areas. The [spreadsheet](results/retrieval_evaluation.csv) contains one row per answer. The [run records](results/) show what each configuration retrieved, called and answered.

[Configurations and archived names](#configurations-and-archived-names) · [Table II results](#retrieval-results) · [Artifact contents](#explore-the-experiment) · [Notebooks](#use-the-notebooks) · [Reproduce the scoring](#reproduce-the-scoring)

## Configurations and archived names

The first column uses the names in Table I of the paper. The next two columns identify the same configurations in the frozen evaluator report, CSV and run directories.

| Paper configuration | Archived report label | CSV `demo` / run directory | Evidence selection and budget |
|---|---|---|---|
| **GROVE** (reference architecture) | JEV | `3_coder_jev_explore` | One tree walk per question. Jev follows all children at small indexes and selects the top ten at larger ones. It ranks leaves scoring at least 0.3, with a 60,000-character passage budget. One LLM call writes the guidelines. |
| **Agentic RAG** (baseline) | Agentic RAG | `2_coder_naive` | The LLM queries a DuckDB vector index. Each search returns ten passages, with no search limit. |
| **Explorer** | Explorer | `1_coder_explorer` | The LLM navigates the tree. It has four batches of content reads. Reads of indexes and summaries are free. |
| **Explorer + Check** | JEV filter | `5_coder_explorer_jev_filter` | Explorer proposes the guidelines under the same read budget. Jev checks each guideline against its cited leaves and keeps it when its score exceeds 0.5. |
| **GROVE-Free** | JEV-free | `4_searcher_jev_search` | The agent writes the branch-selection question and the criteria for judging leaves. It can make up to four searches and receives approved passages directly. |
| **Closed-book** (control) | Pure model (no BoK) | `0_pure_model` | The LLM answers from its own knowledge, with no tools or Guide access. |

All six configurations received the same question and citation format. The request asked for each guideline’s source as a file path or a verbatim quotation of at least eight words. The retrieval architectures ground their answers in the material they access. The control’s role explicitly asks it to answer from memory. Claude Sonnet 4.6 generated the answers, Jev calls reported `typesafe/jev-1.14-20261014`, and the vector index used Qwen3-Embedding-8B.

## Retrieval results

Table II reports passage recall, citation validity, elapsed time, calls and cost per question. Across the three rounds, GROVE and Agentic RAG cited 90% and 89% of the expected leaves, respectively. GROVE averaged US$0.062 and 30.9 seconds per question, compared with US$0.233 and 60.1 seconds for Agentic RAG.

| Configuration | Passage recall | Valid citations | Seconds | LLM calls | Jev calls | US$/question |
|---|---:|---:|---:|---:|---:|---:|
| Closed-book (no BoK) | 4% ± 2% | n/a | 17.1 | 1.0 | 0 | 0.011 |
| Explorer | 74% ± 5% | 100% ± 0% | 96.8 | 12.4 | 0 | 0.432 |
| Agentic RAG | 89% ± 0% | 98% ± 1% | 60.1 | 3.7 | 0 | 0.233 |
| GROVE | 90% ± 1% | 99% ± 1% | 30.9 | 1.0 | 306 | 0.062 |
| GROVE-Free | 88% ± 3% | 99% ± 1% | 98.5 | 3.2 | 1,073 | 0.294 |
| Explorer + Check | 75% ± 3% | 100% ± 0% | 63.3 | 11.3 | 9.9 | 0.359 |

**Passage recall** (`expected_cited` in the CSV) is the fraction of a question’s expected leaves that its answer cites. Each question has one to six expected leaves. The authors reviewed and froze the answer key before the runs. **Valid citations** is the fraction of parsed guideline citations pointing to at least one existing leaf retrieved for that answer. This measures provenance. Whether the passage supports the guideline remains an open question. For the closed-book control, 2% of citations resolved to an existing leaf and retrieval-based validity is undefined.

Across its 75 answers, GROVE made 195 to 419 Jev calls per question. At the evaluation rate, those calls averaged US$0.0087 per question, included in the US$0.062 total.

The evaluator averages the 25 questions within each round and then the three round means. The ± values are sample standard deviations across rounds. For Explorer and Explorer + Check, citation validity was 100% for every answer in all three rounds. Time is wall-clock time with the six configurations running concurrently. Costs use US$3/15 per million Sonnet input/output tokens, US$0.042 per million Jev input tokens, and US$0.01 per million estimated embedding tokens. Charging all LLM input at the uncached rate gives an upper bound.

**Scope of the comparison.** The evaluation covers one Guide and 25 questions. Budgets differ across configurations, only the authors reviewed the key, and relevant leaves outside the key do not contribute to passage recall. Explorer + Check admitted 740 of 742 guidelines. RAG chunks are mapped to leaves by text overlap, and the vector passages contain links to the Guide’s tables. Tree-based configurations can access table contents. The PDF extraction omitted some passages, and the Guide may have been in the language model’s training data.

## Explore the experiment

| To inspect | Open |
|---|---|
| Scores for every answer | [CSV spreadsheet, one row per run × configuration × question](results/retrieval_evaluation.csv) |
| Aggregate metrics and passage recall by question | [Full evaluation report](results/retrieval_evaluation.md) |
| The 25 questions and their answer key | [Questions with expected leaf paths](experiments/benchmark/retrieval/questions.json) · [Source questions](experiments/benchmark/retrieval/questions_src.json) |
| What an agent read, called and answered | [Round 1](results/round-1/) · [Round 2](results/round-2/) · [Round 3](results/round-3/) |
| How answers were generated and scored | [Runner](experiments/run_retrieval.py) · [Evaluator](experiments/benchmark/evaluate_retrieval.py) · [Citation resolution and chunk mapping](experiments/benchmark/retrieval/) |
| Indexing and earlier architecture code | [Vectorization notebook](experiments/build_vector_index.ipynb) · [Architecture notebook](experiments/retrieval_architecture_prototype.ipynb). The second notebook documents the earlier interactive prototype. |
| Execution progress and filter decisions | [Run log](results/logs/run_log.txt) · [Admission log](results/logs/admission_log.md) |
| Generated code and tests for the order lifecycle task | [Task specification and four implementations](results/ecommerce_demo/README.md) |
| How the evaluation was designed | [Changes after the initial run](docs/what_did_not_work.md) |

To follow one answer from the spreadsheet, take the values in its `run`, `demo`, and `question` columns. For example, the row for `r1`, `2_coder_naive`, `Q01` has both a [readable trace](results/round-1/2_coder_naive/Q01.md) and a [JSON record](results/round-1/2_coder_naive/Q01.json). Each run also contains its configuration, model identifiers and a snapshot of the question set. The benchmark files present the authors’ secondary-source annotations in English. The run snapshots, tool messages, original e-commerce prompt and generated code retain their recorded language, including Portuguese. The restoration script uses the snapshot annotations to verify the original input checksums.

### Read a row in the spreadsheet

| CSV columns | Meaning |
|---|---|
| `run`, `demo`, `question` | Join a score to its run directory, configuration and question ID. There are 75 rows per configuration. |
| `expected_cited`, `citations_valid`, `citations_exist` | Fractions from 0 to 1 for that answer. A blank validity value for the control means the measure is undefined. |
| `seconds`, `llm_calls`, `jev_calls`, `embedding_calls`, `cost_usd` | Elapsed time, service calls and estimated cost for the turn. |
| `input_tokens`, `output_tokens`, `jev_tokens_est` | LLM token counts and Jev input tokens. Despite its historical column name, the last field uses reported Jev usage in these runs. |

## Use the notebooks

Both notebooks can be read directly on GitHub. To open them locally in JupyterLab, run the following commands from the artifact root and open the local URL printed in the terminal. The notebooks preserve the implementation from the original experiment environment. The scoring instructions below use only Python’s standard library.

```sh
docker compose build app
docker compose up jupyter
```

### retrieval_architecture_prototype.ipynb — architecture prototype

The [architecture notebook](experiments/retrieval_architecture_prototype.ipynb) contains the Explorer read policy, vector search, Jev traversal, the guideline filter and interactive agent demos. It expects a local `swebok-v4/root/` tree and a matching `swebok.duckdb` at the working-copy root. The vector database is opened during setup. Live calls use locally supplied Anthropic and OpenRouter credentials. The last cells launch demos or submit a question.

For the **five architectures as evaluated**, see `retrieval_agents()` in the [experiment runner](experiments/run_retrieval.py). The historical notebook uses the moving `jev-latest` alias and assigns the same fixed traversal to the demo variables for GROVE and GROVE-Free. The measured runner requests Jev 1.14 and explicitly wires GROVE-Free to the `jev_search` tool with a four-search budget. The CSV and archived records come from that runner.

### build_vector_index.ipynb — build a searchable index

The [vectorization notebook](experiments/build_vector_index.ipynb) reads one UTF-8 Markdown file, removes image-only lines, splits the text into chunks, embeds them, and writes the passages and vectors to DuckDB. The experiment used a 2,048-byte chunk setting, 2,048-dimensional Qwen3 embeddings and a cosine HNSW index. The final cells issue a sample query and inspect the query plan. OpenRouter receives the chunks for embedding. The database contains copies of the indexed passages and belongs in the private working copy.

**For another body of knowledge**, open a copy of the notebook in Jupyter. Supply the text as Markdown. In the first code cell, replace the three assignments to `ROOT`, `SOURCE` and `DB_PATH` with paths visible to the kernel. For example, if the private working copy contains `local/handbook.md`, use the following paths.

```python
SOURCE = Path("/workspace/local/handbook.md")
DB_PATH = SOURCE.with_name("handbook.duckdb")
```

Keep the original imports and other settings. Choose a fresh output filename because the storage cell replaces its target table. In the chunking cell of your copy, replace the lines starting at `lens =` through the preview with the following. This works with short documents as well as large ones.

```python
if not chunks:
    raise ValueError("The Markdown file contains no indexable text.")
print(f"{len(chunks)} chunks")
```

Run the remaining cells in order and change the sample search question to one suited to your text. This produces a **vector index for semantic search**. The PDF-to-Markdown extraction and the navigable topic tree used by the tree-based architectures are separate inputs. The frozen questions and answer key are specific to the SWEBOK Guide.

## Guide and record availability

The study used the *SWEBOK Guide v4.0a*, available from the [IEEE Computer Society](https://www.computer.org/education/bodies-of-knowledge/software-engineering). The downloadable artifact excludes the Guide PDF, the extracted text, the entire topic tree (including its indexes and summaries), the extracted tables and images, and the vector database. Paths recorded in answers and evaluation metadata identify what the agents consulted. Those references contain no leaf files.

In the local extraction used for the evaluation, the index links connect 372 internal nodes (including the root) and 1,143 distinct leaves. The extraction contains two additional Markdown content files outside that linked tree. Each folder has an index and a summary. Branches cover contiguous parts of the Guide in source order. The top-level folders group topics and do not correspond one-to-one to the 18 knowledge areas. A leaf is cited by path. The PDF-to-tree extraction script is unavailable. The [vectorization notebook](experiments/build_vector_index.ipynb) starts from an existing Markdown extraction and creates the second representation. The [chunk map](experiments/benchmark/retrieval/chunk_to_leaves.json) links the 495 vector passages to leaves by text overlap.

The spreadsheet and evaluation report preserve the original scores. In the records, markers replace Guide text returned by tools, matching passages in generated prose, and the answer key's verbatim evidence phrases. Paths, scores, timings and token counts remain available. Brief wording and source titles can still occur in the records. The [excerpt locations](results/excerpt_locations.json) allow someone with the matching local extraction to reconstruct the original JSON evaluator inputs and verify their checksums.

## Reproduce the scoring

### Summarize the public scores

From the artifact root, this example reproduces the passage recall column of Table II using only the published CSV and Python’s standard library. It prints one line per configuration. Each percentage is the mean of three round means, with the sample standard deviation in percentage points.

```bash
python3 - <<'PYTHON'
import csv
from collections import defaultdict
from statistics import mean, stdev

by_config = defaultdict(lambda: defaultdict(list))
with open("results/retrieval_evaluation.csv", newline="") as scores:
    for row in csv.DictReader(scores):
        by_config[row["demo"]][row["run"]].append(float(row["expected_cited"]))

for config, rounds in sorted(by_config.items()):
    percentages = [100 * mean(values) for values in rounds.values()]
    print(f"{config:28} {mean(percentages):.0f}% ± {stdev(percentages):.0f} pp")
PYTHON
```

### Recalculate citation scores from the recorded answers

Recalculating citation scores requires the matching private Markdown extraction of the Guide, including `root/`, `tables/` and `swebok-v4.md`. The restoration step verifies input checksums before writing to a new or empty directory.

From the artifact root, use Bash, Python 3.11 and Docker. The first Docker invocation may download the public Python image. Scoring then runs offline with the Guide mounted read-only, at the absolute path recorded in the traces. The final commands check the entire CSV and all report tables against the archive. The original evaluator prints its historical report title. The published tables use the configuration mapping above.

```bash
GUIDE=/absolute/path/to/swebok-v4
python3 tools/restore_records.py "$GUIDE" ../scoring-copy
cd ../scoring-copy
docker run --rm --network none -v "$PWD:/workspace" -v "$GUIDE:/workspace/swebok-v4:ro" -w /workspace python:3.11-slim python -B -S experiments/benchmark/evaluate_retrieval.py --questions-file questions.json --runs r1 r2 r3 --out reevaluated
cmp results/retrieval_evaluation.csv results/reevaluated.csv
diff -u <(sed -n '/^|/,$p' results/retrieval_evaluation.md) <(sed -n '/^|/,$p' results/reevaluated.md)
```

## Run the retrieval experiment on new answers

New answers require the original experiment environment and live model services. From the restored scoring directory, set `EXPERIMENT_IMAGE` to the name of your existing experiment image, keep `GUIDE` set as above, and supply the matching vector database. Place your own `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` in a private `.env` file there. Use a fresh run ID for each repetition.

```sh
INDEX=/absolute/path/to/swebok.duckdb
docker run --rm --env-file .env -v "$PWD:/workspace" -v "$GUIDE:/workspace/swebok-v4:ro" -v "$INDEX:/workspace/swebok.duckdb:ro" -w /workspace "${EXPERIMENT_IMAGE:?Set EXPERIMENT_IMAGE to the original experiment image}" python experiments/run_retrieval.py --run replication1 --questions-file questions.json
```

## Code and data checks

The notebooks include selected outputs from their original executions. The vectorization notebook shows chunk statistics, embedding dimensions, search distances and the index-plan check. The architecture notebook shows a saved prototype response. Markers identify omitted Guide text. The evaluated answers from the three rounds are in the [run records](results/). Notebook names, cross-references and interactive demo labels have been edited for the release. All Python files and notebook code cells were syntax-checked with Python 3.11. With the matching private extraction, the restored inputs reproduce the CSV byte for byte and all tables in the evaluation report. The original runner and evaluator are included with their recorded results.

The [e-commerce demonstration](results/ecommerce_demo/README.md) includes each implementation and its generated pytest file. Run each suite in its own process so its `order_lifecycle` import resolves to the paired implementation. For example, with Python 3.12 and pytest installed, use the following commands.

```sh
cd results/ecommerce_demo/explorer
python -m pytest -q
```

All four suites passed when checked locally. These are checks of the archived demonstrations. Table II reports the results of the retrieval evaluation.
