# 0017: Museum objects are collected by place, then linked to entities in three layers

- Status: accepted (author approved collecting by place on 2026-11-19; linking tools open, see below)
- Date: 2026-11-19

## Context

Collecting museum records by seed name (decisions 0013, 0015) kept 42 Met and
124 Cleveland objects: name searches over whole collections are imprecise, and the
matching rule that removes the noise also removes most real objects. Objects are
227 of the 3,928 documents in corpus v1 (`docs/data_card.md`).

The expansion probe (`results/probes/expansion_v1.md`, 19 November 2026) counts
objects by place instead:

| Source | Objects (search total, before filtering) |
| --- | --- |
| Met, geoLocation India | 4,367 |
| Met, Asian Art department, "India" | 1,878 |
| Cleveland, culture "India" (CC0) | 3,202 |
| Art Institute of Chicago, place of origin India (public domain) | 347 |

The Met's search ignores its public-domain filter, so its public-domain share is
known only after each object is fetched.

Collecting by place means an object no longer enters through a seed, so its
entity links have to be made after collection. Every record must still link to at
least one Wikidata entity (decision 0004).

## Options considered

Linking:

| Option | For | Against |
| --- | --- | --- |
| A. Layered rules: links the museum gives; then each structured field matched against a gazetteer of the right entity type; then free text, as lower-confidence mentions | Uses the records' own structure; each link carries the layer that made it; homonyms are limited by type (a period field is matched only against dynasties and periods) | Gazetteer and rules to build and test |
| B. A reconciliation service (for example OpenRefine's Wikidata reconciliation) | Ready-made | Depends on an external service at build time; harder to reproduce exactly |
| C. A trained entity linker on the text | Handles free text | Trained mostly on news and Wikipedia; licences to check per model; less transparent |

## Decision

Collect by place, and link with option A.

**Collection.** Met objects whose place is India (geoLocation India, and the Asian
Art department search for "India"), checked one by one for `isPublicDomain`;
Cleveland objects with culture India or South Asia and a CC0 licence; Art
Institute of Chicago objects with place of origin India that are public domain.
Text and licence rules stay as in the seed-name collectors. Objects already in the
corpus are not collected twice.

**Linking, in three layers; every link records its layer:**

1. Links the museum gives: Met tag and artist Wikidata IDs, Cleveland's own
   Wikidata ID for the object. Copied as they are.
2. Structured fields matched against a gazetteer built from Wikidata labels and
   aliases (en, hi, kn), each field only against one entity type: culture and
   place fields against places; period and dynasty fields against dynasties and
   periods; object type against art forms and styles. Dynasty names follow the
   polity-word rule in `src/corpus/matching.py`.
3. Titles and descriptions: entities named in free text become "mentions" with a
   confidence score, in the mentions edge layer (decision 0006), never the core
   layer.

An object with no layer 1 or layer 2 link to anything more specific than India is
linked to India (Q668) and counted as unlinked in the data card.

**Measurement.** 200 links, stratified by layer, are checked by hand by the author;
precision is reported per layer. This is the 200-link check in the specification
(Section 5).

## Open

- Option C as a comparison for layer 3, once the gazetteer exists; the model
  choice goes through its own comparison with licences checked.

## Consequences

- Specification Section 2 ("every document enters through a Wikidata entity")
  changes for objects: they enter by place and are linked after collection.
- The gazetteer planned for query entity linking (specification, Section 5) is
  built now, in the corpus phase, and shared by both uses.
- The data card reports objects by source, by linking layer and unlinked.
