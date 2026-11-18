# 0001: Corpus scope is the Republic of India

- Status: accepted
- Date: 2026-11-17

## Context

The paper asks whether graph structure beats text for Indian heritage
retrieval. The live corpus (`data/classified/classified_documents.json`,
303 Wikipedia articles) mixes Indian sites with sites from elsewhere:
the Colosseum, Versailles, Persepolis, Angkor Thom and others. A keyword
check on 7 Nov 2026 flagged 133 of the 303 as likely non-Indian.

Keyword matching is not reliable enough to decide on its own. A second
pass, over Wikipedia categories, flagged 176 documents and wrongly
included Mamallapuram, which is in Tamil Nadu. The audit needs a written
rule for what counts as Indian, and a reliable way to apply it.

Some cases are hard under any rule: Mohenjo-daro and Harappa (now in
Pakistan), Kathmandu Durbar Square (Nepal), Angkor (Cambodia, with
strong Indian influence), and articles about people or concepts rather
than places.

## Options considered

| Option | Rule | For | Against |
| --- | --- | --- | --- |
| A. Republic of India | Wikidata country (P17) is India (Q668) | One check, easy to state in the paper; matches the ASI Monument ID (P1371) seeds planned for the rebuild | Leaves out Indus Valley sites now in Pakistan |
| B. Pre-1947 subcontinent | Present-day India, Pakistan, Bangladesh | Historically coherent for older periods | The boundary has to be argued; no single Wikidata property gives it |
| C. South Asia | India, Pakistan, Bangladesh, Nepal, Sri Lanka, Bhutan | Widest coverage | Drifts away from "Indian heritage"; harder to defend to reviewers |

How each document is checked:

| Method | For | Against |
| --- | --- | --- |
| Keywords in categories and entities | Offline, fast | Noisy in both directions (see Context) |
| Wikidata country from each article's QID, then a hand check | Uses the same source as the planned rebuild; reproducible; raw responses can be kept | About 14 API calls; people and concepts have no country |
| Hand-label all 303 | Most direct | About an hour; harder to reproduce |

## Decision

Option A: a document is in scope when the Wikidata item for its article
has country (P17) = India (Q668).

The audit proposes a label for each document from Wikidata: `keep`
(P17 is India), `drop` (another country) or `review` (no item, or no
P17). Every `drop` and `review`, and a sample of `keep`, is then checked
by hand. The hand-checked label is final.

## Consequences

- Indus Valley sites in Pakistan, and sites in Nepal, Bangladesh and Sri
  Lanka, are out of scope. The paper says so in one sentence.
- Articles about people, dynasties or concepts (for example Shah Jahan or
  Mughal architecture) usually have no P17. They are judged by hand: kept
  when the subject is Indian.
- The same rule applies to the rebuilt corpus. Wikidata seeds are
  filtered on P17 = Q668.
- The audit outputs record the label for each document and why, so the
  number removed can be reported and checked.
