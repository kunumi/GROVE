# E-commerce order lifecycle demonstration

Each directory contains an implementation and its generated pytest suite from one execution of the [order lifecycle task](SPEC.md) on 18 November 2026. The [original prompt in Portuguese](SPEC.md) defines the interface and behavior. The generated code, comments, exception messages and tests retain the language used by each architecture. The paper reports retrieval metrics from a separate experiment.

| Directory | Configuration |
|---|---|
| [explorer](explorer/) | Explorer |
| [agentic_rag](agentic_rag/) | Agentic RAG |
| [jev](jev/) | GROVE |
| [jev_filter](jev_filter/) | Explorer + Check |

GROVE-Free acted as a search agent in the order lifecycle task. Its draft answer did not come from a coding-agent execution, so the demonstration includes only the four coding configurations.
