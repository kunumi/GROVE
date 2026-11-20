# Retrieval experiment results

Each round contains 175 answers to the same 25 questions. Five retrieval architectures and the closed-book control contributed 150 answers per round. The later Full context session added 25 per round, bringing the total to 525.

| Round | Answer records | Run metadata |
|---|---|---|
| Round 1 | [175 answers](round-1/) | [Original six](round-1/RUN_INFO.json) · [Full context](round-1/RUN_INFO_6_full_context.json) · [Leaf positions](round-1/full_context_layout.json) |
| Round 2 | [175 answers](round-2/) | [Original six](round-2/RUN_INFO.json) · [Full context](round-2/RUN_INFO_6_full_context.json) · [Leaf positions](round-2/full_context_layout.json) |
| Round 3 | [175 answers](round-3/) | [Original six](round-3/RUN_INFO.json) · [Full context](round-3/RUN_INFO_6_full_context.json) · [Leaf positions](round-3/full_context_layout.json) |

Each answer has a JSON record and a readable Markdown trace. In each round, `6_full_context/` holds the 25 new pairs. Its layout file records leaf paths and character offsets in the shared prompt. Markers replace verbatim Guide passages in the published answers. The [configuration table](../README.md#configurations-and-archived-names) maps the six recorded directory names to their names in the paper.

The spreadsheet contains one row per answer. Its `run` column identifies the round. The report summarizes scores across rounds. [Scores for all 525 answers](retrieval_evaluation.csv) · [Evaluation report](retrieval_evaluation.md)

The separate e-commerce demonstration contains generated implementations and tests for an order lifecycle task. [E-commerce demonstration](ecommerce_demo/README.md)
