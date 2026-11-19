# 0005: About 10,000 documents from the seeds plus their one-hop neighbours

- Status: accepted
- Date: 2026-11-19

## Context

The specification first set a target of 1,500–3,000 documents. The seed list
(decision 0002) now holds 1,215 entities, more than the 800–900 first planned.
A corpus of that first size is small next to common retrieval test collections:
the small BEIR collections NFCorpus and SciFact hold about 3.6K and 5.2K
documents, and ArguAna 8,674 ([BEIR](https://arxiv.org/abs/2104.08663)).

A small corpus also weakens RQ1. With few distractors, BM25 already ranks the
right page high, so graph evidence has little room to show an effect. The cost of
judging does not grow with the corpus, because pools are the top 10 per system
per query whatever the corpus size.

## Options considered

| Option | Size | What goes in | For | Against |
| --- | --- | --- | --- | --- |
| A. Seeds only | 1.5–3K | Documents for the 1,215 seeds | Smallest effort | Few distractors; weak test of RQ1 |
| B. Seeds plus one-hop neighbours | About 10K | A, plus Wikipedia articles of entities one hop from a seed along selected relations | More distractors; richer graph paths; same judging cost | Scope drift needs a hand check |
| C. B plus Internet Archive full texts | 30K+ | B, plus public-domain books split into passages | Largest | OCR noise, passage splitting, heavy licence filtering |
| D. Random sample of all Indian heritage items | Any | Unstratified | Simple | Dominated by the best-documented states and types (see decision 0002) |

## Decision

Option B, with a target of about 10,000 documents.

- Neighbours are reached along the core relations of decision 0006 only (for
  example built by, located in, member of dynasty, architectural style).
- A neighbour is added when it passes the India scope rule (decision 0001) and
  has a Wikipedia article in English, Hindi or Kannada.
- A count-only probe of the one-hop neighbours runs before collection. The
  probe, not this record, fixes the final number.
- A random sample of added entities is checked by hand for scope drift, and the
  drift rate is reported in the data card.
- The data card reports the document-type mix (Section 2 of the
  specification), because expansion adds mostly Wikipedia articles and may tilt
  the mix away from objects, manuscripts and inscriptions.

## Consequences

- The specification target changes to about 10,000 documents.
- A neighbour probe script and its output in `results/probes/` come before any
  neighbour collector.
- Embedding the corpus once on a free Colab GPU stays feasible at this size.
