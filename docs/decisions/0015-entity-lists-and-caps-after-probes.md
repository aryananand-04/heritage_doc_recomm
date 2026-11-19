# 0015: Entity lists and collection caps, set from the probes

- Status: accepted (author may revise the caps)
- Date: 2026-11-19

## Context

Decision 0005 set a target of about 10,000 documents and left the final number to
a count-only probe of the one-hop neighbours. Decision 0013 proposed per-seed caps
for the museum and library sources, waiting for per-seed counts. Both probes ran on
19 November 2026:

- `results/probes/neighbours_v1.md`: 4,329 entities lie one hop from the 1,196
  seeds; 2,000 of them are in scope (country India, or no country) and have a
  Wikipedia article in English, Hindi or Kannada.
- `results/probes/collections_v1.md`: per-seed search counts for the 696 non-ASI
  seeds, with a precision sample.

The neighbour drift sample showed that only about 20 of 50 sampled neighbours were
heritage. The rest were districts, cities and villages (through "located in"),
modern musicians (through "genre"), universities, a wildlife sanctuary and a book.

Seeds changed before the probes: hand scans removed 4 people (Janamejaya and Alli
Raani, legendary; Goethe and Catherine of Valois, not Indian heritage) and 15
global styles (Gothic, baroque, Victorian, brutalist, Moroccan and others), whose
articles are not about Indian heritage. The styles stay graph nodes. Seeds went
from 1,215 to 1,196.

## Options considered

Neighbours:

| Option | For | Against |
| --- | --- | --- |
| A. Every in-scope neighbour with an article (2,000) | Largest | About half are not heritage (drift sample) |
| B. Neighbours whose Wikidata class falls under a heritage root, people by the seed date rule | Keeps the drift sample's heritage items, drops places and institutions | Rule to maintain |
| C. Hand-pick neighbours | Precise | Not reproducible at this size |

Growth beyond the neighbours:

| Option | For | Against |
| --- | --- | --- |
| A. None | Simplest | Corpus well below the target |
| B. Every ASI monument with an English article not already a seed | In scope by definition; no drift | Adds sites, which already dominate the type mix |
| C. Two-hop neighbours | Larger | Hub explosion (decision 0007) |

## Decision

**Neighbours (option B), `src/corpus/neighbour_select.py`.** A neighbour becomes a
document entity when:

- its class (P31, up P279) falls under a heritage root: inscription, manuscript,
  treatise, literary work dated before 1900, painting, sculpture, archaeological
  artefact, archaeological site, built structure, cave, garden, ancient city,
  heritage railway, raga, thaat, festival, music festival, music genre, type of
  dance, dynasty, historical country, princely state, architectural style;
- none of its direct classes is excluded (railway station, airport, hotel,
  hospital, cricket field, universities, colleges, schools, libraries, research
  institutes, film and literary festivals, films, list articles);
- for a person, the seed date rule holds (decision 0002);
- it is not removed by the seed hand scans and not already in the ASI list.

Districts, cities, villages and other places stay graph nodes, not documents. The
roots and exclusions were chosen from the classes the probe found. Result: 568
entities (374 site, 147 person/dynasty/style, 36 culture, 5 manuscript, 5 object,
1 inscription), listed in `data/seeds/neighbours_v1.csv`.

**ASI monuments (option B), `src/corpus/asi_remainder.py`.** The 469 ASI monuments
with an English article that are not seeds become document entities, in
`data/seeds/asi_remainder_v1.csv`. It is a separate list, so the seed sample and
every random draw after it stay unchanged.

**Caps (per seed, for the 696 non-ASI seeds).** A collector looks at up to the
first 20 search results per seed and keeps, in rank order, up to the cap of
records whose metadata names the seed.

| Source | Seeds with hits | Median hits (seeds with any) | Precision sample | Cap | Upper bound at the cap |
| --- | --- | --- | --- | --- | --- |
| Met | 281 | 11 | 17 of 35 | 10 | 1,910 |
| Cleveland | 104 | 3.5 | 10 of 18 | 5 | 329 |
| Wellcome manuscripts | 92 | 4 | 16 of 43 | 5 | 302 |
| Wellcome pictures | 112 | 4.5 | 16 of 43 | 3 | 272 |
| Internet Archive | 227 | 7 | 26 of 31 | 5 | 834 |

The Met and Internet Archive caps are higher than proposed in 0013 because they
are the main sources of object and manuscript documents, which the Wikipedia-heavy
entity lists lack. Upper bounds count every hit up to the cap; the collectors keep
only records that name the seed, so the real counts will be lower.

**Size.** Wikipedia covers 2,233 entities (1,196 seeds, 469 ASI, 568 neighbours) in
up to three languages; the museum and library caps give at most 3,647 records. The
real total is known only after collection. It is likely below 10,000; the
specification target becomes the number the collectors produce, reported in the
data card, and is not padded with out-of-scope documents.

## Consequences

- The collectors read three entity lists: `seeds_v1.csv`, `asi_remainder_v1.csv`,
  `neighbours_v1.csv`.
- The data card reports documents per entity list as well as per source.
- The author's hand check of the 50-entity drift sample in
  `results/probes/neighbour_select_v1.md` is recorded before the snapshot.
- Caps can change after collection by re-running a collector from the cached raw
  responses.
