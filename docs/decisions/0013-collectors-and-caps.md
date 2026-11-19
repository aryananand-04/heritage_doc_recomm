# 0013: One collector per source, tried on a sample first, with per-seed caps

- Status: proposed (caps waiting for the author)
- Date: 2026-11-19

## Context

The corpus combines Wikipedia, museum, library, manuscript and inscription
records. Sources differ in size, licence and how well they name Indian entities.
Count-only probes on 18 November 2026 (`results/probes/`) showed:

| Source | Probe count | What it means |
| --- | --- | --- |
| Wikipedia | 8 of 10 sampled seeds have English articles, 3 Hindi, 2 Kannada | Hindi and Kannada documents will be sparse |
| Met | 1,878 public-domain items for "India" in one department | Short records |
| Cleveland | 3,144 for culture "India", 4 for "South Asia" | Short records with descriptions |
| Wellcome | 638 works in Indian languages | Manuscripts and pictures |
| Internet Archive | 1,685,145 texts with subject India; 157 of 200 sampled had no licence URL | Licence filtering removes most |
| SIDDHAM | 61 records for "india" in the Zenodo community | Inscriptions; licence per record |
| Wikisource | 141 hits for one entity (Taj Mahal) | Needs a sample of seeds to size |
| Smithsonian | Not counted; no API key | Left out unless a key is added |

## Options considered

| Option | For | Against |
| --- | --- | --- |
| A. No caps | Takes everything | Famous places flood the corpus |
| B. One cap for all sources | Simple | Ignores how differently sources cover entities |
| C. Per-source caps per seed | Balanced; each source contributes what it covers well | Caps to set and justify |

## Proposed decision

Option C, with these caps per seed (proposed, waiting for the author):

| Source | Cap per seed |
| --- | --- |
| Wikipedia | One article per language (en, hi, kn) |
| Met | 5 |
| Cleveland | 5 |
| Wellcome | 3 manuscripts, 2 pictures |
| Internet Archive | 2 |
| SIDDHAM (proposed) | To set after the record-level licence check, if accepted |
| Wikisource (proposed) | To set after probing a sample of 10 seeds, if accepted |

**How each collector is built and run**

1. Tried on a small sample of seeds first; the sample output is checked before
   the full run.
2. Keeps only records whose metadata names the seed (title, subject or linked
   entity), not records that only match a keyword.
3. Saves every raw response untouched under `data/raw/corpus_v1/`, cached by a
   hash of the request, and retries on rate limits and server errors.
4. Records the licence URI of every record; records with no licence that allows
   reuse are dropped, and the count dropped is reported.
5. Cleans into the record schema of decision 0004 in a separate step.

**Order:** Wikipedia (seeds, then one-hop neighbours), Met, Cleveland, Wellcome,
Internet Archive. SIDDHAM and Wikisource are proposed (specification, Section 10)
and are collected only if accepted and the corpus gates leave time.

## Consequences

- The data card reports, per source: records fetched, kept, dropped for licence,
  dropped for not naming the seed, and capped.
- Changing a cap after collection means re-running that collector from the cached
  raw responses, not re-fetching.
