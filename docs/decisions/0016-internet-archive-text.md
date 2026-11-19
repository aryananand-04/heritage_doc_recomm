# 0016: Internet Archive documents are the public-domain book text, not the uploader's description

- Status: accepted (author may revise)
- Date: 2026-11-19

## Context

The Internet Archive is the largest source of manuscript and text documents
(decision 0015: 227 seeds with hits, cap 5). An item has two kinds of text:

- the metadata (title, description, subjects), written by whoever uploaded it;
- the digitised work itself, as OCR text (`_djvu.txt`).

The collector searches only items marked not in copyright, public domain or CC0
(the probe's filter). That licence covers the work. No Internet Archive page
found on 19 November 2026 states a licence for the uploader-written metadata, so
reusing the description as document text has no clear licence.

The work's text is long: a digitised book runs to hundreds of thousands of
characters, against a median of about 3,500 for an English Wikipedia article
(`results/collection/wikipedia_v1.json`). Dense encoders read only the first few
hundred to few thousand tokens of a text.

## Options considered

| Option | Licence | For | Against |
| --- | --- | --- | --- |
| A. Metadata only (title, description, subjects) | Unclear | Short; clean | No stated licence; descriptions vary from rich to empty |
| B. Full OCR text as one document | Public domain, from the item's own status | The real document; licence clear | Very long; OCR noise; dense models see only the start |
| C. Full OCR text, split into passages for indexing, scored per document by its best passage (MaxP) | As B | As B; every part of a long text can be found; standard for long documents | One more indexing step |

## Decision

Option C.

- A record's text is the item's OCR text with whitespace normalised and nothing
  else changed. Subjects and title are used only to decide that the item names the
  seed and India (`src/corpus/matching.py`), never as document text.
- The licence on each record is the item's own `licenseurl` when it has one, and
  the Public Domain Mark when the item is marked not in copyright.
- Only items in English, Hindi or Kannada (the corpus languages) are kept; others
  are counted and reported.
- Items with no OCR text, or with very little (under 500 characters), are dropped
  and counted.
- Passage splitting belongs to indexing, not to the corpus: the snapshot keeps one
  record per item. The passage length and overlap are set with the retrieval
  systems (decision 0008), and every system scores a document by its best passage.

## Consequences

- The snapshot will likely exceed the 50 MB limit in decision 0004, so it moves to
  Zenodo with the raw responses, and only the manifest stays in git.
- The data card reports length by source, so long book texts are visible next to
  short museum records.
