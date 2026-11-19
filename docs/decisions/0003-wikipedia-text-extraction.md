# 0003: Wikipedia text comes from the TextExtracts API

- Status: proposed
- Date: 2026-11-18

## Context

Most documents in the rebuilt corpus are Wikipedia articles in English, Hindi and
Kannada (one per seed entity and language, through Wikidata sitelinks). Each needs
clean plain text for BM25 and dense retrieval, with enough structure (section
headings) to keep long articles readable, and a record of exactly which revision
was used so the corpus can be rebuilt.

Infobox facts are not needed from the text: the same facts come from Wikidata and
enter the knowledge graph, which keeps graph and text evidence separate
(specification, section 4).

## Options considered

| Option | Text quality | Structure kept | Revision recorded | Effort | Limits |
| --- | --- | --- | --- | --- | --- |
| A. TextExtracts API (`prop=extracts`, `explaintext`) | Clean prose; templates, tables and references removed by the server | Section headings, as `== Heading ==` lines | Yes, with `prop=revisions` in the same request | Low | One full-text extract per request; tables and infobox text are dropped |
| B. REST HTML (`/page/html`) + trafilatura or BeautifulSoup | Good, depends on our cleaning code | Full HTML structure | Yes (ETag carries the revision) | Medium: cleaning rules to write and test | Navigation boxes and references must be stripped by hand |
| C. Wikitext (`action=parse`) + mwparserfromhell | Uneven: templates must be expanded or dropped one by one | Full | Yes | High | Template handling differs per language wiki |
| D. Wikipedia dumps (pages-articles or HTML dumps) | As B or C after parsing | Full | Yes (dump date) | High: multi-GB download for three languages | Fixed to the dump date, which may be weeks old |

Benchmarks built on Wikipedia, such as KILT, use a fixed dump snapshot (option D),
which is the strongest choice for very large corpora. This corpus targets
1,500–3,000 documents in total (specification, section 3), where per-article API
calls with recorded revision IDs give the same reproducibility at a fraction of
the effort.

## Decision

Option A. For every article, request the plain-text extract and the current
revision ID together, save the raw response untouched, and store the revision ID
in the record's raw-response pointer. Hindi and Kannada articles are fetched from
their own wikis the same way.

## Consequences

- One request per article (the API returns a single full extract per call), paced
  at about one per second; the collector logs the actual count and duration.
- Tables inside articles are lost. Monument lists and similar tables are not used
  as document text; where a source table matters (for example the geographical
  indication list), it is parsed separately.
- Re-running the build fetches the latest revisions. The saved raw responses and
  revision IDs are what make a snapshot reproducible, and they are archived with it.
