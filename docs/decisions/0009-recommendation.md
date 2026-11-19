# 0009: Hybrid recommender of meta-paths, PPR and dense similarity, with a diversity re-rank

- Status: accepted
- Date: 2026-11-19

## Context

HeritageRec recommends documents related to a document ("more like this") and
documents of other types tied to a site. There is no click or rating data, so
recommenders that learn from user interactions have nothing to learn from.

## Options considered

| Option | For | Against |
| --- | --- | --- |
| A. Meta-path similarity (shares dynasty, site, style, place, tradition) | Interpretable; each result has a path | Misses documents with few entity links |
| B. Personalized PageRank from the document's entities | Uses the whole graph | Pulled towards hubs without down-weighting |
| C. Dense item-to-item similarity | Works for any document with text | No graph explanation; tests text, not structure |
| D. Knowledge-graph embeddings (RotatE) | Learned structure | Needs training; weaker explanations |
| E. Graph neural recommenders (KGAT, LightGCN) | Strong with interaction data | Learn from clicks, which HeritageRec does not have |
| F. Hybrid of A, B and C, then a diversity re-rank | Each signal covers another's gap; diversity spreads results across types | More parts to evaluate |

## Decision

Option F, with RotatE (D) added as a signal and evaluated alone. Graph neural
recommenders (E) are not used, and the paper says why in one sentence.

- Signals are fused by reciprocal rank fusion, then re-ranked by maximal marginal
  relevance across document types.
- Evaluation: 40 seed documents, judged like queries, with NDCG@10 and type
  coverage@10, and each signal reported alone.

## Consequences

- The recommender reuses the graph, the dense index and the diversity step of the
  retrieval pipeline.
- Seed documents for evaluation are chosen before any recommender run, stratified
  by document type.
