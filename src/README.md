# src

| Folder | Contents |
| --- | --- |
| `corpus/` | Corpus v1: seed selection, probes, entity lists, collectors, shared matching rules, gazetteer, object linking, snapshot and data card. Run modules with `uv run python -m src.corpus.<module>`; the order is in `scripts/build_corpus.sh` |
| `2_preprocessing/` … `8_streamlit_app/`, `fallback/` | Pipeline of the conference version (preprocessing, representation, knowledge graph, ranking, query system, evaluation, Streamlit app), to be replaced by the corpus v1 systems |

The numbered folders are inherited; new code goes in folders with descriptive names.
