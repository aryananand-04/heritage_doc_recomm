"""The rule every museum and library collector uses to decide that a record is about a seed.

A record names a seed when every content word of the seed's search term appears in
the record's descriptive text (in any order, so a Met "Cover (Rumal)" from "Chamba"
matches "Chamba Rumal"), and, where the source is not India-only, the record's place
fields name India. Exact-phrase matching missed real objects; matching on the name
alone let in Baalbek for "Sun Temple" and Yosemite for "Valley of Flowers".
"""

import html
import re

STOPWORDS = {"of", "the", "and", "in", "at", "on", "de", "la", "a", "an"}


def content_words(search: str) -> list[str]:
    words = re.findall(r"[^\W\d_]+", search.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def has_word(text: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", text.lower()) is not None


def names_seed(text: str, search: str) -> bool:
    words = content_words(search)
    return bool(words) and all(has_word(text, w) for w in words)


POLITY_WORDS = r"(?:dynasty|dynasties|empire|kingdom|sultanate|period|era|rulers?|kings?|court|state|region|school)"
# Titles count only with "of" before the name ("Raja of Marwar"): a title directly before a
# name is usually a person ("Prince Sur Singh" is not the Sur Empire).
TITLE_WORDS = r"(?:emperors?|empress|sultans?|maharajas?|maharanas?|rajas?|ranas?|raos?|nawabs?|princes?|princess)"
# Words that may follow the name only ("Mughal India", "Mughal emperor").
AFTER_ONLY = r"(?:india|style|painting)"


def qualifier(label: str) -> str:
    """The bracketed qualifier of a label: 'Pala dynasty (Kamarupa)' -> 'Kamarupa'."""
    m = re.search(r"\(([^)]*)\)\s*$", label)
    return m.group(1) if m else ""


def names_dynasty(text: str, search: str, label: str = "") -> bool:
    """A dynasty seed's name next to a word that marks it as a polity ("Nanda dynasty",
    "Chola period", "period of the Nandas", "Maharana Bhim Singh of Mewar", "Mughal India").
    The bare name is not enough: "Nanda" alone matched paintings of Nanda, Krishna's
    foster-father, and "Matsya" the fish avatar of Vishnu."""
    name = r"\s+".join(re.escape(w) for w in search.lower().split())
    t = text.lower()
    # Two dynasties can share a name (the Palas of Bengal and of Kamarupa); the label's
    # qualifier must then appear too.
    q = qualifier(label)
    if q and not names_seed(text, q):
        return False
    after = re.search(rf"\b{name}s?\b[\s,-]+(?:{POLITY_WORDS}|{AFTER_ONLY})\b", t)
    polity_before = re.search(rf"\b{POLITY_WORDS}\s+(?:of\s+)?(?:the\s+)?{name}s?\b", t)
    # A title, then up to four words of a personal name, then "of" ("Maharana Bhim Singh of Mewar").
    title_of = re.search(rf"\b{TITLE_WORDS}\b(?:\s+[\w().–-]+){{0,4}}?\s+of\s+(?:the\s+)?{name}s?\b", t)
    return bool(after or polity_before or title_of)


def names_india(text: str) -> bool:
    return re.search(r"\bindia", text, re.I) is not None


def plain(value) -> str:
    """Plain text from an API field that may hold HTML markup or entities."""
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()
