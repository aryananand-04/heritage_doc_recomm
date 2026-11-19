# 0008: Retrieval stack of text, graph, fusion, reranking and learning to rank

- Status: accepted, except RotatE (proposed)
- Date: 2026-11-19

## Context

A graph method only counts as better if it beats strong text baselines, not only
the MiniLM baseline the conference version used. Reviewers in information
retrieval expect BM25, a query-expansion variant, modern multilingual dense
models, a cross-encoder and a learned ranker. The graph also has to enter the
ranking in a way that can be measured on its own.

## Options considered

How graph evidence enters the ranking:

| Option | For | Against |
| --- | --- | --- |
| A. Late fusion of ranked lists (reciprocal rank fusion) | No tuning; each list stays a separate system | Cannot weigh signals per query |
| B. Graph features in LambdaMART | Learned weights from real labels; SHAP shows which signals carry it | Needs labels and cross-validation |
| C. Entity-enriched documents (labels and aliases added to the text) | Simple; helps BM25 directly | Mixes graph knowledge into text, so it cannot test structure |
| D. A + B, with C as a control | Measures fusion and learned ranking; C removes the easy explanation | Most systems to run |

## Decision

Option D.

| Stage | Systems |
| --- | --- |
| Candidates | BM25, BM25+RM3, MiniLM, BGE-M3, multilingual-e5, PPR from the query's linked entities; RotatE (proposed) |
| Controls | Entity-expanded BM25; translate-then-search with IndicTrans2 |
| Fusion | Reciprocal rank fusion |
| Reranking | bge-reranker-v2-m3 on the top 50 |
| Learning to rank | LightGBM LambdaMART, 5-fold cross-validation split by query |
| Diversity | Maximal marginal relevance across document types, as its own system |

The 4B reranker (Qwen3-Reranker-4B) stays an upper bound only, and is the first
item dropped if time slips.

## Consequences

- Every stage writes a TREC run file, so each is evaluated alone.
- If entity-expanded BM25 matches the graph methods, the paper says the gain
  comes from knowing the entities, not from structure.
- IndicTrans2 model size and how to run it free get their own options comparison
  (a later decision record).
