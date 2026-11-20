# Retrieval results

Each of the seven configurations answered the same 25 questions in three rounds. The original session contributed 450 answers and the separate Full context session contributed 75.

The tables use the evaluator’s original labels. “Pure model (no BoK)” denotes the closed-book control, and “expected leaves cited” is passage recall. Each value is averaged over the questions in a round, then summarized as the mean ± sample standard deviation across the three rounds.

| metric | Pure model (no BoK) | Explorer | Agentic RAG | JEV | JEV-free | JEV filter | Full context |
|---|---|---|---|---|---|---|---|
| expected leaves cited | 4% ± 2% | 74% ± 5% | 89% ± 0% | 90% ± 1% | 88% ± 3% | 75% ± 3% | 80% ± 1% |
| valid citations | — | 100% ± 0% | 98% ± 1% | 99% ± 1% | 99% ± 1% | 100% ± 0% | 96% ± 1% |
| time (s) | 17.1 ± 0.4 | 96.8 ± 1.5 | 60.1 ± 8.0 | 30.9 ± 0.2 | 98.5 ± 2.4 | 63.3 ± 0.9 | 32.0 ± 0.7 |
| LLM calls | 1.0 ± 0.0 | 12.4 ± 0.2 | 3.7 ± 0.1 | 1.0 ± 0.0 | 3.2 ± 0.1 | 11.3 ± 0.1 | 1.0 ± 0.0 |
| Jev calls | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 306 ± 3.8 | 1,073 ± 44.5 | 9.9 ± 0.3 | 0.0 ± 0.0 |
| cost (US$) | 0.011 ± 0.000 | 0.432 ± 0.004 | 0.233 ± 0.014 | 0.062 ± 0.001 | 0.294 ± 0.019 | 0.359 ± 0.006 | 0.863 ± 0.001 |
| citations that exist in the SWEBOK | 2% ± 1% | 100% ± 0% | 98% ± 1% | 99% ± 1% | 99% ± 1% | 100% ± 0% | 96% ± 1% |
| citations resolving to more than 3 leaves | 0% ± 0% | 0% ± 0% | 0% ± 0% | 0% ± 0% | 0% ± 0% | 0% ± 0% | 0% ± 0% |
| distinct leaves brought into context | — | 9.9 ± 0.2 | 99.7 ± 2.1 | 25.4 ± 0.5 | 32.4 ± 1.6 | 10.2 ± 0.4 | 1,143 ± 0.0 |
| expected leaves retrieved (not comparable across architectures) | — | 76% ± 4% | 97% ± 0% | 93% ± 0% | 93% ± 5% | 75% ± 3% | 100% ± 0% |
| guidelines with a source | 8.4 ± 0.3 | 9.9 ± 0.2 | 11.6 ± 0.1 | 11.4 ± 0.1 | 11.9 ± 0.4 | 9.9 ± 0.3 | 12.2 ± 0.4 |
| LLM input tokens | 148 ± 0.0 | 118,800 ± 1,385.3 | 68,161 ± 4,467.3 | 12,550 ± 126.7 | 70,739 ± 5,560.6 | 103,473 ± 1,624.2 | 281,256 ± 0.0 |
| LLM output tokens | 697 ± 20.6 | 5,038 ± 140.8 | 1,921 ± 60.9 | 1,040 ± 26.9 | 3,487 ± 55.7 | 3,225 ± 62.9 | 1,302 ± 41.2 |
| Jev input tokens | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 206,550 ± 3,010.3 | 708,878 ± 33,817.3 | 7,147 ± 167.3 | 0.0 ± 0.0 |
| embedding calls | 0.0 ± 0.0 | 0.0 ± 0.0 | 5.7 ± 0.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| cost actually billed, with prompt caching (US$) | 0.011 ± 0.000 | 0.432 ± 0.004 | 0.233 ± 0.014 | 0.062 ± 0.001 | 0.294 ± 0.019 | 0.359 ± 0.006 | 0.104 ± 0.001 |
| LLM input tokens read from the prompt cache | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 281,151 ± 0.0 |
| turns without an answer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| turns where the BoK access did not run | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| turns without any citation | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Expected leaves cited, per question (mean over runs)

| question | gold leaves | Pure model (no BoK) | Explorer | Agentic RAG | JEV | JEV-free | JEV filter | Full context |
|---|---|---|---|---|---|---|---|---|
| Q01 | 2 | 0% | 67% | 100% | 100% | 100% | 67% | 100% |
| Q02 | 1 | 0% | 100% | 100% | 100% | 67% | 100% | 100% |
| Q03 | 1 | 33% | 100% | 100% | 100% | 100% | 100% | 100% |
| Q04 | 2 | 0% | 100% | 100% | 100% | 100% | 100% | 100% |
| Q05 | 2 | 33% | 83% | 100% | 83% | 100% | 100% | 50% |
| Q06 | 2 | 0% | 100% | 100% | 100% | 100% | 100% | 67% |
| Q07 | 4 | 0% | 100% | 75% | 67% | 83% | 100% | 75% |
| Q08 | 5 | 7% | 40% | 67% | 67% | 60% | 40% | 40% |
| Q09 | 6 | 0% | 89% | 100% | 94% | 100% | 100% | 83% |
| Q10 | 3 | 11% | 67% | 100% | 67% | 67% | 67% | 100% |
| Q11 | 2 | 17% | 83% | 100% | 100% | 100% | 83% | 100% |
| Q12 | 4 | 0% | 58% | 75% | 100% | 75% | 58% | 100% |
| N01 | 3 | 0% | 33% | 56% | 100% | 67% | 56% | 67% |
| N02 | 2 | 0% | 100% | 100% | 100% | 100% | 100% | 100% |
| N03 | 2 | 0% | 17% | 100% | 100% | 83% | 0% | 50% |
| N04 | 3 | 0% | 100% | 100% | 100% | 100% | 100% | 100% |
| N05 | 3 | 0% | 89% | 78% | 100% | 89% | 78% | 78% |
| N06 | 3 | 0% | 67% | 100% | 100% | 100% | 78% | 56% |
| N07 | 3 | 0% | 100% | 100% | 100% | 89% | 100% | 67% |
| N08 | 3 | 0% | 100% | 33% | 100% | 100% | 67% | 44% |
| N09 | 3 | 0% | 67% | 67% | 67% | 67% | 100% | 89% |
| N10 | 1 | 0% | 0% | 100% | 100% | 100% | 0% | 100% |
| N11 | 3 | 0% | 100% | 100% | 100% | 100% | 100% | 89% |
| N12 | 2 | 0% | 83% | 100% | 100% | 100% | 83% | 83% |
| N13 | 3 | 0% | 0% | 78% | 0% | 44% | 0% | 56% |

Full context: the whole BoK is in the prompt, so every citation that exists is "valid" by construction, and "expected leaves retrieved" is 100% by construction. Its time is not comparable: it ran in a separate session about 14 hours later, without the other architectures running alongside.

## Full context against the other architectures, paired by question

Expected leaves cited, mean over runs per question; exact two-sided sign test, ties dropped.

| compared with | Full context better | tie | Full context worse | p (sign test) |
|---|---|---|---|---|
| Explorer | 9 | 7 | 9 | 1.000 |
| Agentic RAG | 4 | 11 | 10 | 0.180 |
| JEV | 4 | 9 | 12 | 0.077 |
| JEV-free | 5 | 8 | 12 | 0.143 |
| JEV filter | 8 | 8 | 9 | 1.000 |

## Expected leaves cited, by the leaf's position in the full context

Position = middle of the leaf in the 1,263,835-character context (1143 leaves in the Guide's order), in fifths. Each cell: share of (gold leaf, run) pairs where the answer cites the leaf. JEV and Agentic RAG do not see the context; they are the control for how hard the leaves of each fifth are.

| position | gold leaves | Full context | JEV | Agentic RAG |
|---|---|---|---|---|
| 0–20% | 6 | 83% (15/18) | 100% (18/18) | 78% (14/18) |
| 20–40% | 18 | 81% (46/57) | 89% (51/57) | 93% (53/57) |
| 40–60% | 19 | 72% (41/57) | 93% (53/57) | 74% (42/57) |
| 60–80% | 16 | 77% (37/48) | 85% (41/48) | 94% (45/48) |
| 80–100% | 8 | 75% (18/24) | 62% (15/24) | 92% (22/24) |
