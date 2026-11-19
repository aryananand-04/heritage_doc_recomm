# 0004: One validated record schema; snapshot in git, raw responses on Zenodo

- Status: proposed
- Date: 2026-11-18

## Context

The corpus combines Wikipedia articles, museum object records (Met, Cleveland),
manuscript records (Wellcome) and public-domain texts (Internet Archive). Every
system and every evaluation step reads the same records, so the record format has
to be fixed, checked on every build, and stored so that a release can be rebuilt
from a tagged commit.

Every record carries the fields required by the specification (section 3):

| Field | Meaning |
| --- | --- |
| id | Stable document id, built from source and source id |
| title | Title as given by the source |
| text | Plain text used for retrieval |
| language | BCP 47 code: en, hi or kn |
| doc_type | site, object, manuscript_text, inscription, culture or person_dynasty_style |
| source | wikipedia, met, cleveland, wellcome, internet_archive |
| url | Public page for the record |
| licence | Licence URI (Creative Commons or a public-domain mark) |
| entity_ids | Wikidata QIDs the record is about or linked to; the first is the main entity |
| coordinates | Latitude and longitude, where known |
| period_start, period_end | Years (negative for BCE), where known |
| fetched_at | Fetch date and time (UTC) |
| sha256 | Checksum of the text field |
| raw_ref | Path of the saved raw response, plus the revision or source version |

## Options considered

Validation:

| Option | For | Against |
| --- | --- | --- |
| A. pydantic models | Types, value checks and clear errors in one place; already installed as a FastAPI dependency | Adds an explicit dependency |
| B. dataclasses + JSON Schema | Standard formats | Two definitions to keep in step |
| C. pandera on a DataFrame | Good for column checks | Awkward for nested fields and per-record errors |

Storage:

| Option | For | Against |
| --- | --- | --- |
| A. Snapshot as gzipped JSON Lines in git; raw responses on Zenodo with a checksum manifest in git | Diffable, versioned with the code; raw data still public and citable | Raw data is not in the repo itself |
| B. Everything in git | Simplest | Raw responses for a few thousand records are too large for a healthy repository |
| C. Git LFS | Large files tracked with the code | LFS bandwidth limits on free accounts; harder for others to clone |
| D. Parquet or SQLite snapshot | Fast to load | Not diffable; binary files in review |

## Decision

- Records are pydantic models (validation option A). The build stops on the first
  invalid record and names it.
- The snapshot is `data/corpus_v1/records.jsonl.gz`, committed (storage option A),
  with its SHA-256 in `data/corpus_v1/manifest.json`.
- Raw responses are saved gzipped under `data/raw/corpus_v1/`, which is
  gitignored. The manifest lists every raw file with its size and SHA-256, and the
  raw files are archived on Zenodo with the release.

## Consequences

- pydantic becomes an explicit dependency in `pyproject.toml`.
- If the snapshot grows beyond about 50 MB, it moves to Zenodo too, and only the
  manifest stays in git; the data card says which.
- Anyone can check a downloaded raw archive against the committed manifest.
