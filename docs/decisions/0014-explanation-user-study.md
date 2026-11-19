# 0014: A small explanation user study with consent and anonymity

- Status: proposed
- Date: 2026-11-19

## Context

Fidelity tests show whether an explanation matches the ranking. They do not show
whether explanations help people. RQ4 now includes a small user study: do path
explanations help people judge whether a result is relevant? Data from people
needs consent and protection. The CIKM 2026 resource call asked for review-board
procedures to be documented for data collected from people
([call](https://cikm2026.diag.uniroma1.it/resource-papers/)), and other venues
may ask the same.

## Options considered

| Option | For | Against |
| --- | --- | --- |
| A. No study; fidelity only | No ethics work | RQ4 says nothing about people |
| B. Within-subjects study, 12–20 people, each judging results with and without explanations | Small sample still gives paired comparisons | Order effects to balance |
| C. Between-subjects study | No carry-over between conditions | Needs far more people |

## Proposed decision

Option B, 12–20 participants.

**Protocol (proposed)**

- Each participant judges the relevance of a fixed set of query–result pairs,
  half with explanations and half without. Which half gets explanations is
  counterbalanced across participants.
- Measures: agreement with the reference labels, time per judgement, and a short
  trust and usefulness questionnaire.
- About 30–45 minutes per participant, online.

**Consent and anonymity**

- Written informed consent before the study: purpose, what is recorded, how long
  it is kept, that taking part is voluntary, and that anyone may stop or withdraw
  their data until analysis starts.
- Participants are stored only under codes (P01, P02, ...). Names and contact
  details, if collected for scheduling, are kept in a separate file outside the
  repo and deleted after the study.
- No free-text answer is published if it could identify someone; only aggregate
  results and anonymised per-code responses are released.
- The released data card states the consent wording and the ethics approval or
  exemption.

## Open

- Ethics route: whether the author's institution requires review-board approval
  or an exemption for a study of this kind, and how long it takes.
- Recruitment: who the participants are (students, heritage enthusiasts,
  colleagues).

## Consequences

- If accepted, the study is scheduled after the corpus freeze, once approval is
  in place; until then the release reports fidelity only.
