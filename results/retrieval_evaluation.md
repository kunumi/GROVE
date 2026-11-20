# Retrieval results

Each of the five retrieval architectures and the closed-book control answered the same 25 questions in three rounds. All 450 answers were evaluated.

The tables use the evaluator’s original labels. “Pure model (no BoK)” denotes the closed-book control, and “expected leaves cited” is passage recall. Each value is averaged over the questions in a round, then summarized as the mean ± sample standard deviation across the three rounds.

| metric | Pure model (no BoK) | Explorer | Agentic RAG | JEV | JEV-free | JEV filter |
|---|---|---|---|---|---|---|
| expected leaves cited | 4% ± 2% | 74% ± 5% | 89% ± 0% | 90% ± 1% | 88% ± 3% | 75% ± 3% |
| valid citations | — | 100% ± 0% | 98% ± 1% | 99% ± 1% | 99% ± 1% | 100% ± 0% |
| time (s) | 17.1 ± 0.4 | 96.8 ± 1.5 | 60.1 ± 8.0 | 30.9 ± 0.2 | 98.5 ± 2.4 | 63.3 ± 0.9 |
| LLM calls | 1.0 ± 0.0 | 12.4 ± 0.2 | 3.7 ± 0.1 | 1.0 ± 0.0 | 3.2 ± 0.1 | 11.3 ± 0.1 |
| Jev calls | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 306 ± 3.8 | 1,073 ± 44.5 | 9.9 ± 0.3 |
| cost (US$) | 0.011 ± 0.000 | 0.432 ± 0.004 | 0.233 ± 0.014 | 0.062 ± 0.001 | 0.294 ± 0.019 | 0.359 ± 0.006 |
| citations that exist in the SWEBOK | 2% ± 1% | 100% ± 0% | 98% ± 1% | 99% ± 1% | 99% ± 1% | 100% ± 0% |
| citations resolving to more than 3 leaves | 0% ± 0% | 0% ± 0% | 0% ± 0% | 0% ± 0% | 0% ± 0% | 0% ± 0% |
| distinct leaves brought into context | — | 9.9 ± 0.2 | 99.7 ± 2.1 | 25.4 ± 0.5 | 32.4 ± 1.6 | 10.2 ± 0.4 |
| expected leaves retrieved (not comparable across architectures) | — | 76% ± 4% | 97% ± 0% | 93% ± 0% | 93% ± 5% | 75% ± 3% |
| guidelines with a source | 8.4 ± 0.3 | 9.9 ± 0.2 | 11.6 ± 0.1 | 11.4 ± 0.1 | 11.9 ± 0.4 | 9.9 ± 0.3 |
| LLM input tokens | 148 ± 0.0 | 118,800 ± 1,385.3 | 68,161 ± 4,467.3 | 12,550 ± 126.7 | 70,739 ± 5,560.6 | 103,473 ± 1,624.2 |
| LLM output tokens | 697 ± 20.6 | 5,038 ± 140.8 | 1,921 ± 60.9 | 1,040 ± 26.9 | 3,487 ± 55.7 | 3,225 ± 62.9 |
| Jev input tokens | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 206,550 ± 3,010.3 | 708,878 ± 33,817.3 | 7,147 ± 167.3 |
| embedding calls | 0.0 ± 0.0 | 0.0 ± 0.0 | 5.7 ± 0.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| turns without an answer | 0 | 0 | 0 | 0 | 0 | 0 |
| turns where the BoK access did not run | 0 | 0 | 0 | 0 | 0 | 0 |
| turns without any citation | 0 | 0 | 0 | 0 | 0 | 0 |

## Expected leaves cited, per question (mean over runs)

| question | gold leaves | Pure model (no BoK) | Explorer | Agentic RAG | JEV | JEV-free | JEV filter |
|---|---|---|---|---|---|---|---|
| Q01 | 2 | 0% | 67% | 100% | 100% | 100% | 67% |
| Q02 | 1 | 0% | 100% | 100% | 100% | 67% | 100% |
| Q03 | 1 | 33% | 100% | 100% | 100% | 100% | 100% |
| Q04 | 2 | 0% | 100% | 100% | 100% | 100% | 100% |
| Q05 | 2 | 33% | 83% | 100% | 83% | 100% | 100% |
| Q06 | 2 | 0% | 100% | 100% | 100% | 100% | 100% |
| Q07 | 4 | 0% | 100% | 75% | 67% | 83% | 100% |
| Q08 | 5 | 7% | 40% | 67% | 67% | 60% | 40% |
| Q09 | 6 | 0% | 89% | 100% | 94% | 100% | 100% |
| Q10 | 3 | 11% | 67% | 100% | 67% | 67% | 67% |
| Q11 | 2 | 17% | 83% | 100% | 100% | 100% | 83% |
| Q12 | 4 | 0% | 58% | 75% | 100% | 75% | 58% |
| N01 | 3 | 0% | 33% | 56% | 100% | 67% | 56% |
| N02 | 2 | 0% | 100% | 100% | 100% | 100% | 100% |
| N03 | 2 | 0% | 17% | 100% | 100% | 83% | 0% |
| N04 | 3 | 0% | 100% | 100% | 100% | 100% | 100% |
| N05 | 3 | 0% | 89% | 78% | 100% | 89% | 78% |
| N06 | 3 | 0% | 67% | 100% | 100% | 100% | 78% |
| N07 | 3 | 0% | 100% | 100% | 100% | 89% | 100% |
| N08 | 3 | 0% | 100% | 33% | 100% | 100% | 67% |
| N09 | 3 | 0% | 67% | 67% | 67% | 67% | 100% |
| N10 | 1 | 0% | 0% | 100% | 100% | 100% | 0% |
| N11 | 3 | 0% | 100% | 100% | 100% | 100% | 100% |
| N12 | 2 | 0% | 83% | 100% | 100% | 100% | 83% |
| N13 | 3 | 0% | 0% | 78% | 0% | 44% | 0% |
