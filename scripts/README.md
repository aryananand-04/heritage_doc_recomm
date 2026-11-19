# scripts

| Script | What it does |
| --- | --- |
| `build_corpus.sh` | Builds corpus v1 end to end, or rebuilds it from the saved raw responses |
| `audit_corpus.py` | Audits the conference-version corpus (scope, sources, duplicates) |
| `cleaning_report.py` | Writes the cleaning report from the audit and the author's checks |
| `probe_counts.py` | Count-only probe of every candidate source, run before any collector |
| `create-release.sh`, `reindex_pipeline.py`, `verify_splits.py` | Conference-version tooling |

Run Python scripts with `uv run python scripts/<name>.py`.
