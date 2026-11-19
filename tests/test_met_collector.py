"""Tests for the Met collector's matching and record building in src/corpus/met_collector.py (no network)."""

from datetime import UTC, datetime

from src.corpus.met_collector import names_seed, object_text, tag_qids, to_record

RUMAL = {
    "objectID": 1,
    "title": "Cover (Rumal)",
    "objectName": "Cover",
    "culture": "India (Himachal Pradesh, Chamba)",
    "objectDate": "18th century",
    "objectBeginDate": 1700,
    "objectEndDate": 1799,
    "objectURL": "https://www.metmuseum.org/art/collection/search/1",
    "tags": [{"term": "Krishna", "Wikidata_URL": "https://www.wikidata.org/wiki/Q42"}, {"term": "Flowers"}],
}


def test_words_in_any_order_match():
    assert names_seed(RUMAL, "Chamba Rumal")


def test_name_without_india_does_not_match():
    baalbek = {"title": "The Temple of the Sun at Baalbec", "culture": "", "country": "Lebanon"}
    assert not names_seed(baalbek, "Sun Temple")


def test_missing_word_does_not_match():
    madurai = {"title": "Sundareshvara feeding sugarcane", "culture": "India, Madurai, Tamil Nadu"}
    assert not names_seed(madurai, "Nayaks of Kalahasti")


def test_text_is_the_records_own_fields_and_tags_become_entity_links():
    text = object_text(RUMAL)
    assert text.splitlines()[0] == "Cover (Rumal)"
    assert "Culture: India (Himachal Pradesh, Chamba)" in text
    assert tag_qids(RUMAL) == ["Q42"]
    rec = to_record(RUMAL, ["Q7"], "x", datetime(2026, 11, 19, tzinfo=UTC))
    assert rec.entity_ids == ["Q7", "Q42"]
    assert (rec.period_start, rec.period_end) == (1700, 1799)
