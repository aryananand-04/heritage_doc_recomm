# 0012: Hugging Face Spaces demo with path explanations and a graph explorer

- Status: proposed
- Date: 2026-11-19

## Context

Reviewers and readers should be able to try queries and see why each result was
returned. The repo already has a FastAPI backend, a Next.js frontend and a
Streamlit app, from the conference version.

## Options considered

| Option | For | Against |
| --- | --- | --- |
| A. No demo | No work | Explanations stay abstract |
| B. Hugging Face Spaces demo with path explanations | Free hosting; reviewers try queries | Must fit the free tier's memory |
| C. B plus a graph explorer (browse an entity's neighbourhood and paths) | Shows the graph itself, not only results | More front-end work |

## Proposed decision

Option C. The demo shows ranked results with their path and signal explanations;
the explorer shows an entity's neighbours by edge layer and the path from a query
to a result. Every number it shows is read from release files, never written in
by hand.

## Open

- Framework: Gradio or Streamlit on Spaces, or a static page reading precomputed
  results. Chosen by a short options comparison if this record is accepted,
  checking the free tier's memory against the index sizes.

## Consequences

- Listed under "proposed" in the specification (Section 10) until accepted.
