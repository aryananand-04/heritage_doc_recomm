"""Tests for the neighbour selection rule in src/corpus/neighbour_select.py (no network)."""

from src.corpus.neighbour_select import classify

BUILT = "Q811979"
LITERARY = "Q7725634"
MANUSCRIPT = "Q87167"


def test_person_follows_the_seed_date_rule():
    assert classify(["Q5"], set(), {"birth": 1542})[0] == "person_dynasty_style"
    assert classify(["Q5"], set(), {"birth": 1950})[0] is None
    assert classify(["Q5"], set(), None)[0] is None


def test_excluded_class_wins_over_a_heritage_root():
    doc_type, reason = classify(["Q1248784"], {BUILT}, None)  # an airport is a built structure
    assert doc_type is None and "airport" in reason


def test_first_root_in_priority_order_sets_the_type():
    assert classify(["Q42"], {BUILT, MANUSCRIPT}, None)[0] == "manuscript_text"
    assert classify(["Q42"], {BUILT}, None)[0] == "site"


def test_literary_work_needs_a_date_before_the_limit():
    assert classify(["Q42"], {LITERARY}, None, work_year=1590)[0] == "manuscript_text"
    assert classify(["Q42"], {LITERARY}, None, work_year=1955)[0] is None
    assert classify(["Q42"], {LITERARY}, None)[0] is None


def test_no_heritage_root_is_left_out():
    doc_type, reason = classify(["Q1149652"], set(), None)  # a district of India
    assert doc_type is None and "no heritage class" in reason
