"""Tests for the seed selection rules in src/corpus/seeds.py (no network)."""

import random

from src.corpus.seeds import Seed, StateResolver, allocate, round_robin


def test_allocate_gives_minimum_then_shares_rest_by_size():
    quota = allocate({"A": 100, "B": 10, "C": 2}, total=20, minimum=5)
    assert quota["C"] == 2  # fewer than the minimum: takes all it has
    assert quota["B"] >= 5 and quota["A"] >= 5
    assert sum(quota.values()) == 20
    assert quota["A"] > quota["B"]  # the rest goes by size


def test_allocate_never_exceeds_what_is_available():
    quota = allocate({"A": 3, "B": 4}, total=50, minimum=5)
    assert quota == {"A": 3, "B": 4}


def test_allocate_minimum_only_for_floor_keys():
    quota = allocate({"State": 10, "unresolved": 10}, total=6, minimum=5, floor_keys=["State"])
    assert quota["State"] >= 5
    assert sum(quota.values()) == 6


def test_round_robin_spreads_across_groups_and_is_repeatable():
    groups = {"A": ["a1", "a2", "a3"], "B": ["b1"], "C": ["c1", "c2"]}
    first = round_robin(groups, cap=4, rng=random.Random(1))
    again = round_robin(groups, cap=4, rng=random.Random(1))
    assert first == again
    assert {x[0] for x in first} == {"a", "b", "c"}  # every group represented
    assert len(first) == 4


class _FakeResolver(StateResolver):
    def __init__(self):
        self.states = {"S1": "Karnataka", "S2": "Kerala"}
        self.parents = {"D1": {"S1"}, "T1": {"D1"}, "X": {"S1", "S2"}}
        self.facts = {}
        self.cat_states = {}


def test_state_of_follows_located_in_chain():
    r = _FakeResolver()
    assert r.state_of("T1") == {"Karnataka"}
    assert r.state_of("X") == {"Karnataka", "Kerala"}


def test_assign_uses_hint_only_without_wikidata_location():
    r = _FakeResolver()
    r.facts = {"Q1": {"P131": {"T1"}}, "Q2": {}}
    located, hinted = Seed("Q1", "a", "culture"), Seed("Q2", "b", "culture")
    r.assign(located, hint="Kerala")
    r.assign(hinted, hint="Gujarat; Rajasthan")
    assert located.state == "Karnataka" and located.state_source == "located in (P131)"
    assert hinted.state == "multi-state" and hinted.states == ["Gujarat", "Rajasthan"]


def test_year_of_handles_bce_and_unknown_values():
    from src.corpus.seeds import year_of

    assert year_of("1592-01-05T00:00:00Z") == 1592
    assert year_of("-0304-01-01T00:00:00Z") == -304
    assert year_of("http://www.wikidata.org/.well-known/genid/abc") is None
    assert year_of(None) is None


def test_born_before_falls_back_to_death_year():
    from src.corpus.seeds import born_before

    assert born_before({"birth": 1592}, 1900)
    assert not born_before({"birth": 1905, "death": 1890}, 1900)  # birth date wins when present
    assert born_before({"death": 1799}, 1900)
    assert not born_before({}, 1900)
