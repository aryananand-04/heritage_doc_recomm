"""Tests for the object linker's matching rules in src/corpus/link_objects.py (no network)."""

import gzip
import json

from src.corpus.link_objects import Gazetteer, core_names


def gazetteer(tmp_path, entries):
    path = tmp_path / "g.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for qid, etype, names in entries:
            f.write(json.dumps({"qid": qid, "type": etype, "label": {"en": names[0]}, "names": {"en": names}}) + "\n")
    return Gazetteer(path)


def test_core_names_keep_sultanate_and_skip_common_words():
    assert core_names("Mughal Empire", "polity") == ["Mughal"]
    assert core_names("Kingdom of Marwar", "polity") == ["Marwar"]
    assert core_names("Delhi Sultanate", "polity") == []
    assert core_names("South district", "place") == []
    assert core_names("Kangra district", "place") == ["Kangra"]


def test_longest_match_and_full_name_beats_short_form(tmp_path):
    g = gazetteer(
        tmp_path,
        [("Q1", "place", ["Lucknow"]), ("Q2", "place", ["Lucknow district"]), ("Q3", "place", ["Uttar Pradesh"])],
    )
    found = dict(g.find("India (Uttar Pradesh, Lucknow)", {"place"}))
    assert [c["qid"] for c in found["lucknow"]] == ["Q1"]
    assert [c["qid"] for c in found["uttar pradesh"]] == ["Q3"]


def test_entity_qualifier_applies_to_all_its_names(tmp_path):
    g = gazetteer(
        tmp_path,
        [("Q10", "polity", ["Pala Empire"]), ("Q11", "polity", ["Pala dynasty (Kamarupa)", "Pala dynasty"])],
    )
    bengal = {c["qid"] for _, cands in g.find("Pala period, Bengal", {"polity"}) for c in cands}
    assam = {c["qid"] for _, cands in g.find("Pala dynasty, Kamarupa", {"polity"}) for c in cands}
    assert bengal == {"Q10"}
    assert "Q11" in assam


def test_free_text_uses_full_names_only(tmp_path):
    g = gazetteer(tmp_path, [("Q20", "place", ["Krishna district"])])
    assert g.find("Krishna lifting Mount Govardhan", {"place"}, derived=False) == []
