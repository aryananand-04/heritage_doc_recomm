"""Tests for the neighbour probe's counting rules in src/corpus/neighbour_probe.py (no network)."""

from src.corpus.neighbour_probe import scope_of, summarise


def attrs(india=False, country=False, human=False, en=False, hi=False, kn=False, label=""):
    return {"india": india, "country": country, "human": human, "en": en, "hi": hi, "kn": kn, "label": label}


def test_scope_classes():
    assert scope_of(attrs(india=True, country=True)) == "india"
    assert scope_of(attrs()) == "no_country"
    assert scope_of(attrs(country=True)) == "other_country"


def test_seeds_are_not_counted_as_neighbours():
    edges = [("S1", "P84", "S2", "out"), ("S1", "P84", "N1", "out")]
    table = {"N1": attrs(india=True, country=True, en=True), "S2": attrs(india=True, country=True, en=True)}
    out = summarise(edges, table, seeds={"S1", "S2"})
    assert out["all"]["entities"] == 1


def test_neighbour_reached_twice_counts_once_overall_and_once_per_relation():
    edges = [("S1", "P84", "N1", "out"), ("S2", "P149", "N1", "in"), ("S2", "P149", "N2", "in")]
    table = {"N1": attrs(india=True, country=True, en=True), "N2": attrs(country=True, hi=True)}
    out = summarise(edges, table, seeds={"S1", "S2"})
    assert out["all"]["entities"] == 2
    assert out["by_relation"]["P84 out"]["entities"] == 1
    assert out["by_relation"]["P149 in"]["entities"] == 2
    assert out["hubs"][0] == {"qid": "N1", "label": "", "seeds_linked": 2}


def test_other_country_is_out_of_scope_even_with_an_article():
    edges = [("S1", "P84", "N1", "out"), ("S1", "P84", "N2", "out")]
    table = {"N1": attrs(country=True, en=True), "N2": attrs(human=True, kn=True)}
    out = summarise(edges, table, seeds={"S1"})["all"]
    assert out["in_scope_with_any_article"] == 1
    assert out["in_scope_with_article"] == {"en": 0, "hi": 0, "kn": 1}
    assert out["india_with_any_article"] == 0
    assert out["humans"] == 1
