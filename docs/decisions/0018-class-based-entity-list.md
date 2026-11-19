# 0018: Indian heritage buildings and sites by Wikidata class become a fourth entity list

- Status: accepted (author approved on 2026-11-19)
- Date: 2026-11-19

## Context

Corpus v1 has 3,928 documents against a target of about 10,000 (decision 0015,
`docs/data_card.md`). The seed strata take World Heritage, Tentative List and a
sample of ASI sites, plus the rest of the ASI monuments (decision 0015); most
Indian temples, mosques, forts and palaces with Wikipedia articles are in none of
the lists. The expansion probe (`results/probes/expansion_v1.md`) counts Indian
items (country India) of 14 heritage classes: 24,269 items, 4,569 with an article
in English, Hindi or Kannada, and 3,516 of those not already in an entity list
(articles: en 3,345, hi 565, kn 399; 171 with Hindi or Kannada only).

## Options considered

| Option | For | Against |
| --- | --- | --- |
| A. No class-based list | Type mix unchanged | Corpus stays well under the target |
| B. Every item of the 14 classes, country India, with an article | Same kind of rule as the ASI list; in scope by definition; adds Hindi- and Kannada-only articles | Sites dominate further (Hindu temples are 2,337 of the 4,569) |
| C. B with a cap per state | Balances states | Drops real items; the cap is arbitrary |

## Decision

Option B. The classes are those in `src/corpus/expansion_probe.py` (Hindu temple,
temple, temple complex, Jain temple, mosque, church building, fort, fortification,
fortress, palace, tomb, mausoleum, stepwell, archaeological site), matched on
direct class (P31), country India (P17). Items already in another entity list or
removed by a hand scan are left out. The list is
`data/seeds/class_list_v1.csv`, built by `src/corpus/class_list.py` from the
probe's cached answers, and collected by the Wikipedia collector like the others.

## Consequences

- The data card reports documents per entity list and the document-type mix, so
  the larger share of sites is visible; results are reported per document type.
- A random sample of the list is checked by hand for scope, as for the neighbours.
