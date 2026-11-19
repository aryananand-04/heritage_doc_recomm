# 0006: Graph edges are kept in separate layers; RQ1 uses the Wikidata layer only

- Status: accepted
- Date: 2026-11-19

## Context

RQ1 asks whether graph structure improves retrieval over text. If the graph
mixes edges of different kinds, a positive result cannot say which kind of
structure helped, and an edge built from text could carry text similarity into
the graph. The conference version had this problem: embedding-similarity and
cluster edges sat in the same graph as factual links.

## Options considered

| Option | Edges | For | Against |
| --- | --- | --- | --- |
| A. Wikidata statements only | Core factual relations | Cleanest test; fully reproducible from Wikidata | Thin for objects and culture, where Wikidata is sparse |
| B. A plus Wikipedia hyperlinks | Core, plus article → article links | Dense, editor-made links between topics | Links are editorial, not typed facts |
| C. B plus Getty AAT and geography | B, plus object type and material, and spatial relations | Covers objects and places well | More sources to keep in step |
| D. C plus LLM-extracted triples | C, plus relations an LLM reads from text | Most coverage | Noisy, hard to reproduce, and partly derived from text, so it blurs RQ1 |

## Decision

Option C, with every edge labelled by its layer so each layer can be switched on
or off.

| Layer | Use |
| --- | --- |
| Core (Wikidata statements, sitelinks, derived periods) | Headline RQ1 result |
| Wikipedia hyperlinks | Addition on top of core |
| Getty AAT (object type, material, depicts) | Addition on top of core |
| Geography (administrative containment, proximity) | Addition on top of core |
| Mentions (entity linking on text) | Addition on top of core |
| LLM-extracted triples | Labelled ablation only; never in the headline result |

Hyperlinks are entity links made by editors, not text similarity, so they fit the
rule that the graph holds only structured facts and entity links.

## Consequences

- The edge list carries a `layer` column; the graph builder takes the layers to
  load as an argument.
- The results include a table of core versus core plus each layer.
- The LLM-extraction model and prompt get their own short options comparison
  before that ablation is built, and the extracted triples are saved with model
  name, version and date.
