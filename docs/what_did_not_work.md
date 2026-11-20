# Changes after the initial run

The initial run on 18 November 2026 informed the design of the retrieval evaluation. The paper reports results from the three rounds on 19 November.

| Issue in the initial run | Change for the reported experiment |
|---|---|
| Jev judged citations produced by architectures that also used Jev. | The evaluator resolves citations against the local tree, checks turn-level retrieval records, and measures passage recall against a frozen answer key. This removes Jev from scoring. |
| Calibration of Explorer + Check used leaves from the answer key. | The filter used its original admission criterion and a fixed threshold above 0.5. The log records the judgments so readers can see how selective it was. |
| The architectures received different task instructions. | All six configurations received the same question and citation format. Retrieval-specific roles and tools still vary by design. |
| GROVE-Free had one search opportunity. | It could issue up to four searches and revise the query and criteria. |
| A single run gave no measure of variation between runs. | Each configuration answered every question in three independent runs. The report gives sample standard deviations across runs. |
| A moving Jev model alias made the executed version hard to identify. | The runner requested Jev 1.14 and recorded the model identifier returned by the service. |
| Missing or truncated extracted passages left some expected leaves unsupported. | The authors checked evidence phrases against the available extraction, reviewed the expected leaves, and froze the revised key before running. Gaps in the extraction still limit passage recall. |
