"""Tests for the Cleveland, Wellcome and Internet Archive collectors' matching rules (no network)."""

from src.corpus import cleveland_collector as cle
from src.corpus import internet_archive_collector as ia
from src.corpus import matching
from src.corpus import wellcome_collector as wel


def test_matching_words_any_order_and_india():
    assert matching.names_seed("Cover (Rumal) India (Himachal Pradesh, Chamba)", "Chamba Rumal")
    assert not matching.names_seed("The Temple of the Moon", "Sun Temple")
    assert matching.names_india("Northern India, Pahari Kingdom of Chamba")
    assert matching.plain("<i>Ragamala</i> &amp; more") == "Ragamala & more"


def test_cleveland_scattered_words_in_a_long_description_are_not_enough():
    obj = {
        "title": "Nine Celestial Bodies (Navagraha)",
        "tombstone": "Nine Celestial Bodies (Navagraha), 700s. Central India. Sandstone.",
        "description": "The sun leads the planets, as carved on many a temple doorway.",
        "culture": ["Central India"],
    }
    assert not cle.names_seed(obj, "Sun Temple")
    obj["description"] = "Made for a Sun Temple in Central India."
    assert cle.names_seed(obj, "Sun Temple")


def test_wellcome_indian_language_counts_as_india_and_copyright_notes_are_dropped():
    work = {
        "title": "Charaka Samhita",
        "languages": [{"id": "san"}],
        "notes": [
            {"noteType": {"id": "general-note"}, "contents": ["Palm-leaf copy."]},
            {"noteType": {"id": "copyright-note"}, "contents": ["(c) someone"]},
        ],
    }
    assert wel.names_seed(work, "Charaka Samhita")
    assert "(c) someone" not in wel.work_text(work)
    assert "Palm-leaf copy." in wel.work_text(work)


def test_ia_script_decides_the_language():
    assert ia.script_language("The Narrative of Domingo Paes, a Portuguese traveller") == "en"
    assert ia.script_language("ಹಂಪಿ ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ರಾಜಧಾನಿ") == "kn"
    assert ia.script_language("یہ کتاب اردو میں ہے and a few words") is None
    hindi = "हम्पी विजयनगर साम्राज्य की राजधानी थी"
    assert ia.language({"language": ["hin"]}, hindi) == "hi"
    assert ia.language({"language": ["san"]}, hindi) is None  # Devanagari, but not Hindi


def test_ia_licence_only_public_domain_or_cc0():
    assert ia.licence({"licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/"}).startswith("https://")
    assert ia.licence({"possible-copyright-status": "NOT_IN_COPYRIGHT"}) == ia.PD_MARK
    assert ia.licence({"licenseurl": "https://creativecommons.org/licenses/by-nc/4.0/"}) is None


def test_dynasty_name_needs_a_dynasty_word():
    assert matching.names_dynasty("A coin of the Nanda dynasty", "Nanda")
    assert matching.names_dynasty("Sculpture, Chola period, 11th century", "Chola")
    assert matching.names_dynasty("struck in the period of the Nandas", "Nanda")
    assert not matching.names_dynasty("Nanda and the elders, page from a Bhagavata Purana", "Nanda")


def test_dynasty_titles_and_mughal_india_count():
    assert matching.names_dynasty("Maharana Bhim Singh of Mewar (r. 1778–1828) with a Consort", "Mewar")
    assert matching.names_dynasty("Portrait of Suraj Singh Rathor, Raja of Marwar", "Marwar")
    assert matching.names_dynasty("c. 1560. Mughal India, court of Akbar", "Mughal")
    assert not matching.names_dynasty("Matsya, Fish Avatara of Vishnu", "Matsya")


def test_titles_need_of_and_qualifiers_must_appear():
    assert not matching.names_dynasty("Portrait of Prince Sur Singh Rathor of Bikaner", "Sur")
    assert not matching.names_dynasty("Pala period, Bengal, 11th century", "Pala", "Pala dynasty (Kamarupa)")
    assert matching.names_dynasty("Pala period, Kamarupa (Assam)", "Pala", "Pala dynasty (Kamarupa)")
    assert matching.names_dynasty("Pala period, Bengal, 11th century", "Pala", "Pala Empire")


def test_ia_year_from_title_and_undated_items():
    assert ia.item_year({"date": "1888"}) == 1888
    assert ia.item_year({"title": "Rajput Painting By Basil Gray Indian Art 1948 London"}) == 1948
    assert ia.item_year({"title": "An Album Of Nandalal Bose"}) is None


def test_repeat_scans_keep_the_longest_copy():
    from datetime import UTC, datetime

    from src.corpus.ia_series_collector import drop_repeat_scans
    from src.corpus.records import Record, sha256

    words = " ".join(f"word{i} alpha beta gamma delta" for i in range(60000))

    def rec(i, text):
        return Record(
            id=f"internet_archive:v{i}", title="Imperial Gazetteer", text=text, language="en",
            doc_type="manuscript_text", source="internet_archive", url="https://archive.org/details/x",
            licence="https://creativecommons.org/publicdomain/mark/1.0/", entity_ids=["Q668"],
            fetched_at=datetime(2026, 11, 19, tzinfo=UTC), sha256=sha256(text), raw_ref="x",
        )  # fmt: skip

    kept, groups = drop_repeat_scans(
        [rec(1, words), rec(2, words + " extra page"), rec(3, words.replace("alpha", "zeta"))]
    )
    assert [r.id for r in kept] == ["internet_archive:v2", "internet_archive:v3"]
    assert groups == [["internet_archive:v2", "internet_archive:v1"]]
