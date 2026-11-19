# 0011: Release on Zenodo and Hugging Face, with a ranx-compatible loader

- Status: accepted
- Date: 2026-11-19

## Context

A benchmark is reused when it can be found, cited and loaded in a few lines.
Resource tracks look for this directly: the SIGIR 2027 resource call asks for a
stated licence and encourages persistent identifiers and archival hosting
([call](https://sigir2027.org/pages/submit-resource.html)); the CIKM 2026
resource call required dataset papers to publish data on a service that issues
a DOI, such as Zenodo, and to describe how the data is made FAIR
([call](https://cikm2026.diag.uniroma1.it/resource-papers/)).

## Options considered

| Option | For | Against |
| --- | --- | --- |
| A. GitHub repo only | No extra work | No DOI; not archival |
| B. Zenodo DOI | Citable, archival | No easy loading |
| C. Zenodo DOI plus Hugging Face Datasets | Citable and easy to find and load | Two copies to keep identical |
| D. C plus a loader returning ranx-compatible qrels and runs | One-line evaluation for others | A small package to maintain |

## Decision

Option D.

- Zenodo holds the tagged code, the corpus snapshot, raw-response archive,
  queries, qrels and run files, with one DOI per release.
- Hugging Face Datasets holds the corpus, queries and qrels, with a dataset card
  generated from the data card. Its checksums match the Zenodo files.
- The loader returns the queries, corpus, qrels and runs, with qrels and runs in
  TREC format readable by ranx.

## Licences

- Queries, qrels and judgements: CC BY 4.0.
- Wikipedia and Wikisource text keeps its own licence (CC BY-SA), so the corpus
  snapshot cannot be relicensed as CC BY. The data card states the licence of
  each part.
- Museum and library records keep the licence each record carries.

## Consequences

- The release checklist includes a checksum comparison between Zenodo and
  Hugging Face.
- The loader gets its own tests in `tests/`.
