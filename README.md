# HeritageRec

A reproducible retrieval and recommendation study for Indian heritage: does knowledge-graph
evidence help people find documents about Indian heritage better than text alone, in English,
Hindi and Kannada?

HeritageRec builds a multilingual corpus of Indian heritage documents (sites, objects,
manuscripts and texts, inscriptions, cultural practices, people and dynasties), links every
document to Wikidata entities, and compares text, graph and hybrid retrieval on a public
benchmark.

> **Status:** corpus v1 is being built (November 2026 to January 2027); the benchmark and
> results follow. The earlier conference version (ICICACS 2026) is superseded; its files are
> kept in [`legacy/`](legacy/) for traceability only.

<!-- Add an overview figure here: docs/figures/overview.png -->

## Research questions

| # | Question |
| --- | --- |
| RQ1 | Does graph evidence (Personalized PageRank, SimRank, meta-paths) improve ranking over BM25 and multilingual dense retrieval? |
| RQ2 | Does combining graph and text signals beat each alone, and which signals matter? |
| RQ3 | Do graph signals help Hindi and Kannada queries find English documents? |
| RQ4 | Are graph-path explanations faithful to the ranking? |

The full design is in [`docs/plan/specification.md`](docs/plan/specification.md).

## The corpus

Every document is about the heritage of the Republic of India, carries its source, licence and
the raw response it came from, and is linked to at least one Wikidata entity. Counts by source,
language, document type, licence and state are in the generated **[data card](docs/data_card.md)**.

| Source | What it gives | Licence of the text |
| --- | --- | --- |
| Wikipedia (en, hi, kn) | Articles on sites, dynasties, people, styles and cultural forms | CC BY-SA 4.0 |
| The Metropolitan Museum of Art | Public-domain objects from India | CC0 |
| Cleveland Museum of Art | CC0 objects from India | CC0 |
| Art Institute of Chicago | Public-domain artworks from India | CC0 (descriptions CC BY 4.0) |
| Wellcome Collection | Catalogue records of pictures | CC0 |
| Internet Archive | Public-domain books, including the Imperial Gazetteer of India | Public domain |

How the documents were chosen:

1. **Entities from Wikidata, by recorded rules:** World Heritage and Tentative List sites, ASI
   monuments, dynasties, cultural forms, styles and historical people; entities one link away
   that are heritage; and Indian temples, forts, mosques and palaces by class.
2. **Wikipedia text** for each entity in English, Hindi and Kannada, with revision IDs.
3. **Museum objects collected by place** (India), then linked to entities in three layers: links
   the museum gives, structured fields matched against a Wikidata gazetteer, and names in free text.
4. **Strict scope and licence rules:** only public-domain, CC0 or CC BY(-SA) material; nothing
   synthetic or translated by machine.

Every hand decision and every rule added because of something found in the data is logged in
[`docs/data_notes.md`](docs/data_notes.md). Design decisions, each with the options considered,
are in [`docs/decisions/`](docs/decisions/).

## Rebuilding the corpus

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/azzs11/heritagerec.git
cd heritagerec
uv sync
cp .env.example .env    # set CONTACT_EMAIL (sent in the User-Agent, as Wikimedia asks)
bash scripts/build_corpus.sh
```

A first run downloads from the public APIs and takes several hours. Every response is saved
under `data/raw/` before it is used, so later runs make no requests and rebuild a byte-identical
snapshot. Re-downloading later gives a different corpus (articles and catalogues change), so the
frozen snapshot, not a fresh run, is the reference version.

## Data availability

The full snapshot and the raw responses will be archived on Zenodo with a DOI, and the
benchmark published on Hugging Face, at the v1.0 release. The repository holds the code, the
data card, the manifest (record counts and SHA-256 checksums of the snapshot and of every raw
file) and the collection reports, so a downloaded copy can be checked against it.

## Results

<!-- Results tables and figures will be added at the v1.0 release, generated from results/paper_v1/. -->

To be added.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/corpus/` | Corpus v1: seed selection, probes, entity lists, collectors, gazetteer, linking, snapshot, data card |
| `data/seeds/` | Entity lists and the hand-scan files in `curated/` |
| `data/corpus_v1/` | Manifest and links; the snapshot itself is rebuilt locally or downloaded at release |
| `results/` | Probe counts, collection and linking reports, audit of the old corpus |
| `docs/` | Data card, data notes, decision records, specification |
| `tests/` | Tests (`uv run --group dev pytest tests/`) |
| `legacy/` | Superseded conference-version files, kept for traceability |
| `api/`, `frontend/` | Demo application from the conference version |

## Licence

The code is released under the [Apache License 2.0](LICENSE). The documents keep the licences
of their sources (CC BY-SA 4.0, CC BY 4.0, CC0 or public domain), recorded on every record and
summarised in the [data card](docs/data_card.md).

## Citation

<!-- Add the paper and the Zenodo DOI here at release. -->

To be added at release.

## Authors

<!-- Confirm the author list and order with the co-authors. -->

Akchhya Singh, with Aryan Anand and the co-authors of the conference version.
