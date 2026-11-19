# tests

Run all tests with `uv run --group dev pytest tests/`. They need no network.

| File | Covers |
| --- | --- |
| `test_seeds.py` | Seed selection rules: quotas, order-independent sampling, date rules |
| `test_neighbour_probe.py`, `test_neighbour_select.py` | Neighbour counting and the heritage-class rule |
| `test_wikipedia_collector.py` | Wikipedia text cleaning and the record schema |
| `test_met_collector.py`, `test_collectors.py` | Matching rules for museums and libraries, dynasty names, Internet Archive language and licence, repeat scans |
| `test_link_objects.py` | Gazetteer short forms, qualifiers and longest-match linking |
| `test_smoke.py`, `test_recommender.py`, `test_monitoring_demo.py` | Conference-version code |

`conftest.py` keeps two old scripts out of collection until they are repaired (reasons inside).
