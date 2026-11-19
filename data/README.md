# data

| Folder | Contents | In git |
| --- | --- | --- |
| `seeds/` | Entity lists for corpus v1 (`seeds_v1.csv`, `asi_remainder_v1.csv`, `neighbours_v1.csv`, `class_list_v1.csv`), the seed report, and the hand-scan files in `curated/` | Yes |
| `corpus_v1/` | `manifest.json` (snapshot and raw-file checksums), `links_v1.jsonl.gz` (object entity links), a record sample; the snapshot itself is rebuilt locally or downloaded at release | Partly |
| `corpus_v1/staging/` | One file of validated records per collector, before the snapshot | No (rebuilt) |
| `gazetteer/` | `gazetteer_v1.jsonl.gz`: Wikidata labels and aliases (en, hi, kn) by entity type, used for linking | Yes |
| `raw/` | Every API response, saved untouched and gzipped, named by a hash of the request | No (archived on Zenodo at release) |
| `balanced/`, `classified/`, `metadata/`, `ontology/`, `processed/`, `embeddings/`, `evaluation/`, `entity_importance/` | Files of the conference version, superseded by corpus v1 | Some |

How corpus v1 is built: `bash scripts/build_corpus.sh` (see the top-level README). What each
number means: [`docs/data_card.md`](../docs/data_card.md). Every hand change:
[`docs/data_notes.md`](../docs/data_notes.md).
