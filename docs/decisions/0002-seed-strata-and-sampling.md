# 0002: Seed entities are drawn from fixed strata, with every state represented

- Status: proposed
- Date: 2026-11-18

## Context

Every document in the rebuilt corpus enters through a Wikidata seed entity
(specification, section 3). Which entities become seeds decides what the corpus
covers, so the selection has to be written down as rules that a script applies,
not as a hand-picked list. The aim is broad coverage of Indian heritage across
all states and union territories, major and minor: sites, objects, manuscripts
and texts, cultural items, people, dynasties and styles.

Probes on 18 November 2026 showed three things that shape the rules:

- The obvious dynasty rule (type "dynasty" with country India) misses most major
  polities. Chola, Mughal, Gupta, Hoysala, Rashtrakuta and Vijayanagara are typed
  "historical country", and several have no country statement at all.
- ASI monuments are very uneven by state, so an unstratified sample would be
  dominated by a few states.
- Wikidata covers cultural heritage thinly and inconsistently. Dance forms are
  recorded as subclasses rather than instances; the intangible heritage property
  also carries Malaysian, Dutch and Iraqi lists; crafts with a geographical
  indication are not recorded at all; and folk music and folk theatre have almost
  no class membership. Wikipedia categories and list articles fill these gaps,
  with every item still checked on Wikidata.

## Options considered

| Option | For | Against |
| --- | --- | --- |
| A. Rule-based strata from Wikidata, with Wikipedia lists where Wikidata is thin, plus recorded hand scans | Reproducible; every exclusion has a reason on file; coverage can be measured per state | Needs several rules and a scan step for culture |
| B. Wikidata rules only | Simplest and fully automatic | Misses most crafts, folk music and theatre, and major dynasties |
| C. A hand-curated list | Fully controlled | Not reproducible; hard to defend; slow |
| D. Random sample of all Indian heritage items on Wikidata | Unbiased by design | Dominated by the best-documented types and states |

## Decision

Option A. The strata are:

| Stratum | Rule | Take |
| --- | --- | --- |
| World Heritage sites | Heritage designation World Heritage Site (P1435), country India | All |
| Tentative List sites | Heritage designation Tentative List (P1435), country India | All |
| Dynasties and historical states | Type dynasty with country India, plus any dynasty or historical country whose capital (P36) is in India | All, listed for a hand scan |
| ASI monuments | ASI Monument ID (P1371), country India, English article; sites already in the two strata above are excluded | About 500, sampled by state (below) |
| Cultural heritage | Sub-strata below | About 210 |
| Styles | Architectural style (P149) or movement (P135) of seeded sites, one hop | All reached |
| People | Rule set at the first checkpoint, see below | All that pass the rule |
| Top-up | For thin states and union territories, see below | Until 5 seeds and one cultural seed, where items exist |

Cultural sub-strata, with the author's quotas:

| Sub-stratum | Source and rule | Take |
| --- | --- | --- |
| Intangible heritage | P3259 limited to the UNESCO Representative List, UNESCO Masterpieces and India's national list, country or origin India | All |
| Classical and traditional dance | Subclasses of "Indian classical dance", hand-scanned | All kept after the scan; the eight officially recognised classical forms are flagged |
| Folk dance | Subclasses or instances of "Indian folk dance", plus Vattakkali | Up to 40, stratified by state |
| Classical music | Subclasses of "Indian classical music", hand-scanned to genres and forms | 15 |
| Folk music | Wikipedia category "Indian folk music" and its regional subcategories, hand-scanned | 20, stratified by state |
| Folk theatre | Wikipedia category tree "Theatre of India", hand-scanned to traditional forms | 20, stratified by state |
| Painting schools | Members of "Indian painting" on Wikidata, plus schools linked from the "Indian painting" article, typed as an art style, school, movement or genre (country check below) | All found, hand-scanned |
| Crafts with geographical indication | Rows of the "Geographical indications in India" article of type handicraft or textiles, with an Indian state or union territory, linked to Wikidata (country check below) | Up to 60, stratified by state |

Hand-scan decisions (kept and removed items, with reasons) are stored in
`data/seeds/curated/culture_scan.csv`, so the scan is part of the record.

ASI sampling by state: every state and union territory gets its available
monuments up to a minimum of 5; the remaining places up to the total are shared
in proportion to each state's count (largest remainder), and monuments are drawn
at random within each state with a fixed, recorded seed.

State assignment: an entity's state is the state or union territory reached by
following "located in" (P131) upwards. Cultural items without a location use
location (P276), indigenous to (P2341) or country of origin (P495); failing
those, states named in the article's English Wikipedia categories ("Dances of
Assam", "Culture of Kerala"); failing those, the state recorded in the hand scan
or the source list. The source used is recorded for every seed. Items that
belong to many states are counted as "multi-state".

Country check for crafts and painting schools: these items rarely carry a
location on Wikidata, so the Indian origin comes from the source list (the GI
table's state column, the "Indian painting" article), and an item is rejected
when Wikidata gives it a country (P17) other than India.

Decisions taken by the author at the first seed checkpoint (18 November 2026):

- Dynasties: colonial administrative entities (British Raj, Company rule,
  Dominion of India, Eastern States Union, Ajmer-Merwara) are removed, as
  recorded in `data/seeds/curated/dynasty_scan.csv`. Princely states are kept
  only when a direct Wikidata statement links them to a site or cultural seed.
- People: historical figures born before 1900 (or, without a birth date, who died
  before 1900) who commissioned, designed or founded a seeded site (P88, P84,
  P112) or belonged to a seeded dynasty (P53). People with no dates are left out.
- Painting schools and folk dance: removals and replacements are recorded in the
  culture hand-scan file, with reasons.
- Top-up: any state or union territory with fewer than 5 seeds, or no cultural
  seed, is topped up from its English Wikipedia categories "Culture of X",
  "Monuments and memorials in X" and "Festivals in X" (one level of
  subcategories, skipping cuisine, cinema, people, organisations, sport and
  media). Items are typed on Wikidata as a cultural item or a site; people,
  films, food, institutions, administrative bodies and overview articles
  ("Music of X") are excluded, and the same country check applies. The report
  lists every addition and every place that still cannot be filled.

## Consequences

- Before any collection, the seeds script writes a report of seed counts per
  stratum, per state and union territory, and per document type. States, union
  territories or types that come out thin are listed with proposed adjustments,
  and the author decides.
- Object and manuscript documents come from museum and archive records matched to
  seed entities; their counts are measured by the collector probes, not here.
- A seed that appears in several strata is kept once, in the first stratum that
  selects it, and its other strata are recorded.
