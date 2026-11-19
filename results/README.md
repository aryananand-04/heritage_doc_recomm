# results

| Folder | Contents | Made by |
| --- | --- | --- |
| `audit/` | Audit and cleaning report of the conference-version corpus | `scripts/audit_corpus.py`, `scripts/cleaning_report.py` |
| `probes/` | Count-only probes run before collecting: sources, neighbours, expansion; scope samples for hand checks | `scripts/probe_counts.py`, `src/corpus/*_probe.py`, `src/corpus/neighbour_select.py`, `src/corpus/class_list.py` |
| `collection/` | One report per collector: records looked at, kept, and left out with the reason | `src/corpus/*_collector.py` |
| `linking/` | Gazetteer counts, object-linking counts by layer, and the 200-link sample for the hand check | `src/corpus/gazetteer.py`, `src/corpus/link_objects.py` |
| `evaluation_report.json` | Conference-version evaluation, superseded (see `legacy/`) | — |

Results of the benchmark will go in `paper_v1/`, each file recording its query count, date,
commit and configuration.
