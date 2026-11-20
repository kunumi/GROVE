# Retrieval experiment results

Each round tested the same 25 questions with five retrieval architectures and one closed-book control. Each configuration produced 25 answers per round, for 150 answers per round and 450 overall.

| Round | Answer records | Run metadata |
|---|---|---|
| Round 1 | [150 answers](round-1/) | [Run settings](round-1/RUN_INFO.json) |
| Round 2 | [150 answers](round-2/) | [Run settings](round-2/RUN_INFO.json) |
| Round 3 | [150 answers](round-3/) | [Run settings](round-3/RUN_INFO.json) |

Each answer has a JSON record and a readable Markdown trace. The [configuration table](../README.md#configurations-and-archived-names) maps the six recorded directory names to their names in the paper.

The spreadsheet contains one row per answer. Its `run` column identifies the round. The report summarizes scores across rounds. [Scores for all 450 answers](retrieval_evaluation.csv) · [Evaluation report](retrieval_evaluation.md)

The separate e-commerce demonstration contains generated implementations and tests for an order lifecycle task. [E-commerce demonstration](ecommerce_demo/README.md)
