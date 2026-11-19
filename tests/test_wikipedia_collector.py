"""Tests for record building in src/corpus/wikipedia_collector.py and the schema in records.py (no network)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.corpus.records import Record, sha256
from src.corpus.wikipedia_collector import clean_text, to_record

ENTITY = {"qid": "Q1", "doc_type": "site"}
NOW = datetime(2026, 11, 19, tzinfo=UTC)


def response(extract="Hampi is a village.", **page):
    base = {
        "pageid": 42,
        "title": "Hampi",
        "fullurl": "https://en.wikipedia.org/wiki/Hampi",
        "extract": extract,
        "revisions": [{"revid": 7}],
    }
    return {"query": {"pages": [base | page]}}


def test_clean_text_cuts_tail_sections_and_empty_headings():
    text = "Intro.\n\n== History ==\nOld.\n\n== Gallery ==\n\n== See also ==\nVirupaksha\n\n== References =="
    assert clean_text(text, "en") == "Intro.\n\n== History ==\nOld."


def test_clean_text_keeps_a_heading_whose_subsection_has_text():
    text = "Intro.\n\n== Temples ==\n=== Virupaksha ===\nMain temple."
    assert clean_text(text, "en") == text


def test_clean_text_hindi_tail():
    assert clean_text("परिचय।\n\n== सन्दर्भ ==\nकड़ी", "hi") == "परिचय।"


def test_clean_text_kannada_see_also():
    assert clean_text("ಪರಿಚಯ.\n\n== ಇದನ್ನೂ ನೋಡಿ ==\nಒಡಿಸ್ಸಿ", "kn") == "ಪರಿಚಯ."


def test_to_record_builds_a_valid_record_with_revision():
    rec, outcome = to_record(ENTITY, "en", response(), "data/raw/x.json.gz", NOW)
    assert outcome == "kept"
    assert rec.id == "wikipedia:en:42"
    assert rec.raw_ref.endswith("#revid=7")
    assert rec.sha256 == sha256("Hampi is a village.")


@pytest.mark.parametrize(
    "page,reason",
    [
        ({"missing": True}, "missing"),
        ({"pageprops": {"disambiguation": ""}}, "disambiguation"),
        ({"extract": ""}, "empty"),
    ],
)
def test_to_record_skips_with_reason(page, reason):
    rec, outcome = to_record(ENTITY, "en", response(**page), "x", NOW)
    assert rec is None and outcome == reason


def test_record_rejects_mismatched_checksum_and_bad_qid():
    rec, _ = to_record(ENTITY, "en", response(), "x", NOW)
    with pytest.raises(ValidationError):
        Record(**(rec.model_dump() | {"sha256": "0" * 64}))
    with pytest.raises(ValidationError):
        Record(**(rec.model_dump() | {"entity_ids": ["Hampi"]}))
