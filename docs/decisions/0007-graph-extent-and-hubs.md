# 0007: One-hop graph, hub down-weighting, PPR on the full graph, SimRank on candidates

- Status: accepted
- Date: 2026-11-19

## Context

Each hop out from the seeds multiplies the graph. Two hops reach entities such as
India, Hinduism or a whole century, which connect almost everything, so a graph
built that far makes every pair of documents look related. The graph methods also
differ in cost: Personalized PageRank is a sparse matrix iteration and scales to
the full graph, while exact SimRank compares every pair of nodes and does not.

## Options considered

Extent:

| Option | For | Against |
| --- | --- | --- |
| A. Seeds only | Smallest | Misses the links that make graph paths interesting |
| B. Seeds plus one hop | Matches the corpus (decision 0005); paths stay meaningful | Some hubs still enter |
| C. Two hops | Most connections | Hub explosion; documents related only through very general entities |

Hubs:

| Option | For | Against |
| --- | --- | --- |
| A. No weighting | Simplest | Hubs dominate PPR and meta-paths |
| B. Remove hubs above a degree threshold | Simple | Throws away real facts; threshold is arbitrary |
| C. Inverse-frequency down-weighting | Keeps every fact, lowers the weight of very common entities | One more parameter, tuned on dev |

SimRank:

| Option | For | Against |
| --- | --- | --- |
| A. Exact SimRank on the full graph | Exact | Does not scale to this graph |
| B. SimRank over each query's candidate set | Exact within the set; cheap | Only scores candidates |
| C. Approximate SimRank (for example Monte Carlo) | Covers the whole graph | Approximation error to report |

## Decision

- Extent B: the core graph is seeds plus one hop. Two hops are measured by a
  count-only probe and reported, but not used.
- Hubs C: inverse-frequency down-weighting, with the weight tuned on the 40 dev
  queries only.
- Personalized PageRank runs on the full graph.
- SimRank runs over each query's candidate set (B), or as an approximation (C) if
  the candidate set is too narrow on dev. The choice and any approximation error
  are reported.

## Consequences

- The data card reports node and edge counts at one hop, and the two-hop probe
  counts beside them.
- SimRank results are labelled with the variant used.
