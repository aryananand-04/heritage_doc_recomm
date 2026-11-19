# 0010: Two open-weight LLM judges, validated by two people on about 50 test queries

- Status: proposed. Until it is accepted, the original design stands: one LLM judge and the
  author's 15-query validation (specification, Section 9).
- Date: 2026-11-19

## Context

Relevance labels come from an LLM judge with a fixed, UMBRELA-style prompt on a
0–3 scale, which is accepted in IR when it is disclosed and its agreement with
people is measured. One judge model can carry its own bias, and one human
validator gives no ceiling to compare the judge against. The first plan used one
judge and 15 queries judged by the author.

## Options considered

LLM judges:

| Option | For | Against |
| --- | --- | --- |
| A. One open-weight judge | Simplest | One model's bias goes unmeasured |
| B. Two open-weight judges from different families | Judge disagreement is measured; free | Twice the judging runs |
| C. A paid API judge | Possibly stronger | Cost; harder to reproduce |

Human validation:

| Option | For | Against |
| --- | --- | --- |
| A. 15 queries, the author | Under an hour | No human ceiling; small |
| B. About 50 queries, the author and one more person | Human–human ceiling; enough queries per language to report | Several hours per person |

## Proposed decision

- Judges B: two open-weight models from different families, chosen by an options
  comparison (licence checked on the model card), neither used inside any
  compared system. A paid API judge only if both agree poorly with the humans,
  and then both runs are reported.
- Validation B: about 50 test queries, including Hindi and Kannada ones, judged
  by the author and one more person, each blind to the LLM labels, to the
  systems and to the other person's labels.
- Reported: LLM–LLM, human–human and LLM–human weighted kappa and raw agreement,
  overall, per language and per system (judge-bias check).

**Judging sheets.** Built by a script from the real test pools once they exist
(benchmark block, after the corpus freeze); no sheet is filled with placeholder
rows before then. The same sheet design serves the 15-query validation of the
original design.

| Column | Shown to the judge | Notes |
| --- | --- | --- |
| pair_id | Yes | Random, so order reveals nothing |
| query | Yes | Original language |
| title, text excerpt, link | Yes | The same fields the LLM judge sees |
| grade | Yes, a 0–3 dropdown | Definitions on a header tab |
| note | Yes | Optional |
| system, LLM grade, other judge's grade | No | Kept only in the pool file |

Each person gets their own copy in a different random order. A script reads both
back and writes TREC qrels and the agreement report.

## Open

- The judge models (options comparison in week 2 for the first judge; the
  second only if this record is accepted).
- Pool depth for the human sheets: top 5 per system may give each person too many
  pairs at 50 queries. The pool size is measured first; if it is too large, the
  options are a smaller depth or a fixed subset of systems, chosen before anyone
  judges.
- The sheet tool (one Google Sheet per judge, or another tool that keeps the two
  people blind to each other).

## Consequences

- The disclosure says: queries LLM-generated and author-curated; labels judged by
  two LLMs; a validation set judged by two people, with every agreement figure.
- The second person is named in the paper's acknowledgements only with their
  consent.
