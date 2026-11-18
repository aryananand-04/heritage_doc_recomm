"""Select the seed entities for corpus v1 and report their coverage.

Applies the strata in docs/decisions/0002-seed-strata-and-sampling.md. Nothing
is collected here: the output is the list of seed entities and a report of how
they spread over strata, states and union territories, and document types, for
the author to approve before any collection starts.

Usage:
    uv run python -m src.corpus.seeds

Outputs:
    data/seeds/seeds_v1.csv        one row per seed entity
    data/seeds/seed_report.md      coverage report
    data/raw/seeds_v1/*.json.gz    raw responses (gitignored; archived on release)
"""

import csv
import random
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import unquote

from bs4 import BeautifulSoup

from src.corpus.wikimedia import ROOT, Wikimedia, qid, values

RUN = "seeds_v1"
RAW_DIR = ROOT / "data/raw" / RUN
OUT_DIR = ROOT / "data/seeds"
SCAN_FILE = OUT_DIR / "curated/culture_scan.csv"
DYNASTY_SCAN_FILE = OUT_DIR / "curated/dynasty_scan.csv"

RANDOM_SEED = 20261118
ASI_TOTAL = 500
ASI_MIN_PER_STATE = 5
FOLK_DANCE_CAP = 40
GI_CAP = 60
THIN_STATE = 5  # a state or union territory with fewer seeds than this is topped up
PEOPLE_BORN_BEFORE = 1900

INDIA = "Q668"
WORLD_HERITAGE = "Q9259"
TENTATIVE_LIST = "Q1459900"
DYNASTY = "Q164950"
HISTORICAL_COUNTRY = "Q3024240"
KINGDOM = "Q417175"
EMPIRE = "Q48349"
INDIAN_FOLK_DANCE = "Q5464449"
INDIAN_PAINTING = "Q2744438"
DISTRICT_OF_INDIA = "Q1149652"
ICH_LISTS = (
    "Representative List of the Intangible Cultural Heritage of Humanity",
    "Masterpieces of the Oral and Intangible Heritage of Humanity",
    "National List for Intangible Cultural Heritage (ICH)",
)
LANGS = ("en", "hi", "kn")
LABEL = 'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }'
HAS_EN = "?art schema:about ?i ; schema:isPartOf <https://en.wikipedia.org/> ."

# Strata in priority order: a seed selected by several strata is kept in the first.
DOC_TYPE = {
    "world_heritage": "site",
    "tentative_list": "site",
    "asi_monument": "site",
    "dynasty": "person_dynasty_style",
    "culture": "culture",
    "style": "person_dynasty_style",
    "person": "person_dynasty_style",
    "topup_site": "site",
    "topup_culture": "culture",
}


@dataclass
class Seed:
    qid: str
    label: str
    stratum: str
    sub_stratum: str = ""
    state: str = ""
    states: list = field(default_factory=list)
    state_source: str = ""
    flags: list = field(default_factory=list)
    also_in: list = field(default_factory=list)
    status: str = "selected"
    sitelinks: set = field(default_factory=set)

    @property
    def doc_type(self) -> str:
        return DOC_TYPE[self.stratum]


def batches(items, size=200):
    items = list(items)
    for i in range(0, len(items), size):
        yield i // size, items[i : i + size]


# ------------------------------------------------------------------ states


class StateResolver:
    """Finds the Indian state or union territory an entity lies in."""

    def __init__(self, wm: Wikimedia):
        self.wm = wm
        rows = wm.sparql(
            "states",
            f"""SELECT ?s ?sLabel WHERE {{
                VALUES ?t {{ wd:Q12443800 wd:Q467745 }} ?s wdt:P31 ?t .
                FILTER NOT EXISTS {{ ?s wdt:P576 ?end }} {LABEL} }}""",
        )
        self.states = {qid(r["s"]): r["sLabel"] for r in rows}
        self.parents: dict[str, set] = {}
        self.facts: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self.cat_states: dict[str, list] = {}

    def load(self, name: str, qids) -> None:
        """Fetch location facts for entities, then their P131 chains."""
        todo = sorted(set(qids) - set(self.facts))
        for n, chunk in batches(todo):
            rows = self.wm.sparql(
                f"{name}_facts_{n:02d}",
                f"""SELECT ?i ?p ?v WHERE {{ VALUES ?i {{ {values(chunk)} }}
                    VALUES ?p {{ wdt:P131 wdt:P276 wdt:P495 wdt:P2341 wdt:P36 wdt:P17 }}
                    ?i ?p ?v . FILTER(isIRI(?v)) }}""",
            )
            for q in chunk:
                self.facts[q]  # record that it was looked up, even with no facts
            for r in rows:
                self.facts[qid(r["i"])][qid(r["p"])].add(qid(r["v"]))
        start = {v for q in todo for p in ("P131", "P276", "P495", "P2341", "P36") for v in self.facts[q][p]}
        self._climb(name, start)

    def _climb(self, name: str, start: set) -> None:
        frontier = {q for q in start if q not in self.states}
        for depth in range(8):
            todo = sorted(q for q in frontier if q not in self.parents)
            if not todo:
                return
            for n, chunk in batches(todo):
                rows = self.wm.sparql(
                    f"{name}_up{depth}_{n:02d}",
                    f"SELECT ?x ?p WHERE {{ VALUES ?x {{ {values(chunk)} }} OPTIONAL {{ ?x wdt:P131 ?p }} }}",
                )
                for r in rows:
                    self.parents.setdefault(qid(r["x"]), set())
                    if "p" in r:
                        self.parents[qid(r["x"])].add(qid(r["p"]))
            frontier = {p for q in todo for p in self.parents.get(q, ()) if p not in self.states}

    def state_of(self, q: str, seen=None) -> set:
        if q in self.states:
            return {self.states[q]}
        seen = seen or set()
        if q in seen:
            return set()
        seen.add(q)
        found = set()
        for p in self.parents.get(q, ()):
            found |= self.state_of(p, seen)
        return found

    def assign(self, seed: Seed, hint: str = "") -> None:
        facts = self.facts.get(seed.qid, {})
        rules = (
            (("P131",), "located in (P131)"),
            (("P276",), "location (P276)"),
            (("P2341", "P495"), "indigenous to or origin (P2341, P495)"),
            (("P36",), "capital (P36)"),
        )
        for props, source in rules:
            found = set().union(*(self.state_of(v) for p in props for v in facts.get(p, ())))
            if found:
                seed.states, seed.state_source = sorted(found), source
                break
        else:
            hinted = [h.strip() for h in hint.split(";") if h.strip()]
            if self.cat_states.get(seed.qid):
                seed.states, seed.state_source = self.cat_states[seed.qid], "Wikipedia categories"
            elif hinted:
                seed.states, seed.state_source = hinted, "hand scan or source list"
        seed.state = "multi-state" if len(seed.states) > 1 else (seed.states[0] if seed.states else "")

    def not_foreign(self, q: str) -> bool:
        """False only when Wikidata gives a country (P17) and India is not among them.

        Crafts and art styles rarely carry a location on Wikidata, so absence of
        evidence is not treated as evidence of a foreign item; the source list
        (the GI table, the Indian painting article) supplies the Indian origin.
        """
        countries = self.facts.get(q, {}).get("P17", set())
        return not countries or INDIA in countries

    def load_categories(self, wm: Wikimedia, name: str, qids) -> None:
        """States named in an article's English Wikipedia categories ('Dances of Assam')."""
        names = {s: s for s in self.states.values()}
        names |= {"Delhi": "National Capital Territory of Delhi", "Orissa": "Odisha"}
        pattern = re.compile(
            r"\b(?:of|in|from) (" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b"
        )
        titles = en_titles(wm, f"{name}_titles", qids)
        by_title = {t: q for q, t in titles.items()}
        for n, chunk in batches(sorted(by_title), 50):
            params = {"action": "query", "prop": "categories", "clshow": "!hidden", "cllimit": "max", "redirects": "1"}
            params["titles"], page = "|".join(chunk), 0
            while True:
                data = wm.wiki(f"{name}_cats_{n:02d}_{page}", params)
                for p in data.get("query", {}).get("pages", []):
                    q = by_title.get(p["title"])
                    for c in p.get("categories", []):
                        for m in pattern.finditer(c["title"]):
                            if q:
                                self.cat_states.setdefault(q, [])
                                if names[m.group(1)] not in self.cat_states[q]:
                                    self.cat_states[q].append(names[m.group(1)])
                if "continue" not in data:
                    break
                params, page = {**params, **data["continue"]}, page + 1
        for q in self.cat_states:
            self.cat_states[q].sort()


# --------------------------------------------------------------- selection


def allocate(counts: dict[str, int], total: int, minimum: int, floor_keys=None) -> dict[str, int]:
    """Share `total` places over groups: up to `minimum` each, the rest by largest remainder.

    Only groups in `floor_keys` (default: all) are guaranteed the minimum.
    """
    floor_keys = set(counts) if floor_keys is None else set(floor_keys)
    alloc = {k: min(c, minimum) if k in floor_keys else 0 for k, c in counts.items()}
    remaining = total - sum(alloc.values())
    spare = {k: counts[k] - alloc[k] for k in counts}
    pool = sum(spare.values())
    if remaining <= 0 or pool == 0:
        return alloc
    if remaining >= pool:
        return dict(counts)
    quota = {k: remaining * spare[k] / pool for k in counts}
    extra = {k: int(quota[k]) for k in counts}
    left = remaining - sum(extra.values())
    for k in sorted(counts, key=lambda k: (-(quota[k] - extra[k]), k))[:left]:
        extra[k] += 1
    return {k: alloc[k] + extra[k] for k in counts}


def round_robin(by_group: dict[str, list], cap: int, rng: random.Random) -> list:
    """Take one item per group in turn (groups in name order) until `cap` items."""
    # Shuffle groups in name order: the generator is shared, so the order it is used in must
    # not depend on how the groups were built (sets iterate differently from run to run).
    queues = {g: rng.sample(sorted(by_group[g]), len(by_group[g])) for g in sorted(by_group)}
    out = []
    while len(out) < cap and any(queues.values()):
        for g in sorted(queues):
            if queues[g] and len(out) < cap:
                out.append(queues[g].pop(0))
    return out


# ------------------------------------------------------------------ strata


def labelled(rows, var="i") -> dict[str, str]:
    return {qid(r[var]): r.get(f"{var}Label", "") for r in rows}


def designation(wm, name, item) -> dict[str, str]:
    return labelled(
        wm.sparql(
            name,
            f"SELECT DISTINCT ?i ?iLabel WHERE {{ ?i wdt:P1435 wd:{item} ; wdt:P17 wd:{INDIA} . {LABEL} }}",
        )
    )


def dynasties(wm) -> dict[str, tuple[str, str]]:
    rows = wm.sparql(
        "dynasties",
        f"""SELECT DISTINCT ?i ?iLabel ?rule WHERE {{
          {{ ?i wdt:P31/wdt:P279* wd:{DYNASTY} ; wdt:P17 wd:{INDIA} . BIND("dynasty, country India" AS ?rule) }}
          UNION
          {{ VALUES ?t {{ wd:{DYNASTY} wd:{HISTORICAL_COUNTRY} wd:{KINGDOM} wd:{EMPIRE} }}
             ?i wdt:P31 ?t ; wdt:P36 ?cap . ?cap wdt:P17 wd:{INDIA} . BIND("capital in India" AS ?rule) }}
          {LABEL} }}""",
    )
    out: dict[str, tuple[str, str]] = {}
    for r in rows:
        q = qid(r["i"])
        rules = set(out.get(q, ("", ""))[1].split("; ")) - {""} | {r["rule"]}
        out[q] = (r.get("iLabel", ""), "; ".join(sorted(rules)))
    return out


def asi_monuments(wm) -> dict[str, str]:
    return labelled(
        wm.sparql(
            "asi_monuments",
            f"SELECT DISTINCT ?i ?iLabel WHERE {{ ?i wdt:P1371 ?id ; wdt:P17 wd:{INDIA} . {HAS_EN} {LABEL} }}",
        )
    )


def intangible_heritage(wm) -> dict[str, str]:
    names = " ".join(f'"{n}"@en' for n in ICH_LISTS)
    lists = [
        qid(r["s"]) for r in wm.sparql("ich_lists", f"SELECT ?s WHERE {{ VALUES ?l {{ {names} }} ?s rdfs:label ?l }}")
    ]
    return labelled(
        wm.sparql(
            "ich",
            f"""SELECT DISTINCT ?i ?iLabel WHERE {{ VALUES ?s {{ {values(lists)} }} ?i wdt:P3259 ?s .
                {{ ?i wdt:P17 wd:{INDIA} }} UNION {{ ?i wdt:P495 wd:{INDIA} }} {HAS_EN} {LABEL} }}""",
        )
    )


def folk_dances(wm) -> dict[str, str]:
    return labelled(
        wm.sparql(
            "folk_dance",
            f"""SELECT DISTINCT ?i ?iLabel WHERE {{
                {{ ?i wdt:P279+ wd:{INDIAN_FOLK_DANCE} }} UNION {{ ?i wdt:P31/wdt:P279* wd:{INDIAN_FOLK_DANCE} }}
                {HAS_EN} {LABEL} }}""",
        )
    )


def titles_to_qids(wm, name: str, titles) -> dict[str, str]:
    """English Wikipedia titles -> QIDs, following redirects."""
    out = {}
    for n, chunk in batches(sorted(set(titles)), 50):
        data = wm.wiki(
            f"{name}_pageprops_{n:02d}",
            {
                "action": "query",
                "prop": "pageprops",
                "ppprop": "wikibase_item",
                "redirects": "1",
                "titles": "|".join(chunk),
            },
        )
        q = data.get("query", {})
        renamed = {x["from"]: x["to"] for x in q.get("normalized", []) + q.get("redirects", [])}
        found = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in q.get("pages", [])}
        for t in chunk:
            final = t
            for _ in range(3):
                final = renamed.get(final, final)
            if found.get(final):
                out[t] = found[final]
    return out


def painting_candidates(wm) -> dict[str, str]:
    """Wikidata members of 'Indian painting' plus schools linked from its article sections."""
    members = labelled(
        wm.sparql(
            "painting_members",
            f"""SELECT DISTINCT ?i ?iLabel WHERE {{
                {{ ?i wdt:P31/wdt:P279* wd:{INDIAN_PAINTING} }} UNION {{ ?i wdt:P279+ wd:{INDIAN_PAINTING} }}
                UNION {{ ?i wdt:P361 wd:{INDIAN_PAINTING} }} {HAS_EN} {LABEL} }}""",
        )
    )
    html = wm.wiki("indian_painting_article", {"action": "parse", "page": "Indian painting", "prop": "text|revid"})
    soup = BeautifulSoup(html["parse"]["text"], "html.parser")
    wanted = re.compile(
        r"Mughal|Deccan|Rajput|Pahari|Malwa|Mysore|Tanjore|Pattachitra|Assam|Other regional|Company|Bengal school|Vernacular"
    )
    titles, take = set(), False
    for el in soup.find_all(["h2", "h3", "h4", "p", "ul", "div"]):
        if el.name in ("h2", "h3", "h4"):
            take = bool(wanted.search(el.get_text(" ", strip=True)))
            continue
        if take:
            for a in el.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/wiki/") and ":" not in href[6:]:
                    titles.add(unquote(href[6:].split("#")[0]).replace("_", " "))
    from_article = titles_to_qids(wm, "painting_links", titles)
    candidates = dict(members)
    for t, q in from_article.items():
        candidates.setdefault(q, t)
    return candidates


def painting_filter(wm, candidates: dict[str, str]) -> dict[str, str]:
    """Keep candidates typed as a style, school, movement or genre of art."""
    # Art-specific types only: plain "movement" or "tradition" also matches religious movements.
    keep_types = re.compile(
        r"art movement|art style|artistic|painting|school of|style of|genre of|art form|art genre", re.I
    )
    types = defaultdict(set)
    for n, chunk in batches(sorted(candidates)):
        for r in wm.sparql(
            f"painting_types_{n:02d}",
            f"""SELECT ?i ?tLabel WHERE {{ VALUES ?i {{ {values(chunk)} }} ?i wdt:P31|wdt:P279 ?t .
                ?t rdfs:label ?tLabel FILTER(lang(?tLabel) = "en") }}""",
        ):
            types[qid(r["i"])].add(r["tLabel"])
    return {q: label for q, label in candidates.items() if any(keep_types.search(t) for t in types[q])}


def gi_crafts(wm) -> list[dict]:
    """Handicraft rows of the GI table, with the state given in the table."""
    page = wm.wiki(
        "gi_article",
        {"action": "parse", "page": "Geographical indications in India", "prop": "text|revid", "redirects": "1"},
    )
    soup = BeautifulSoup(page["parse"]["text"], "html.parser")
    rows = []
    for table in soup.select("table.wikitable"):
        trs = table.find_all("tr")
        head = [c.get_text(" ", strip=True).lower() for c in trs[0].find_all(["th", "td"])]
        if "type" not in head:
            continue
        i_name = next(i for i, h in enumerate(head) if "geographical indication" in h)
        i_type, i_state = head.index("type"), next(i for i, h in enumerate(head) if "state" in h)
        for tr in trs[1:]:
            cells = tr.find_all(["th", "td"])
            if len(cells) <= max(i_name, i_type, i_state):
                continue
            link = cells[i_name].find("a", href=True)
            rows.append(
                {
                    "name": cells[i_name].get_text(" ", strip=True),
                    "type": cells[i_type].get_text(" ", strip=True),
                    "state": cells[i_state].get_text(" ", strip=True),
                    "title": unquote(link["href"][6:]).replace("_", " ")
                    if link and link["href"].startswith("/wiki/")
                    else "",
                }
            )
    return rows


def one_hop(wm, name: str, sources, props: str, extra: str = "") -> dict[str, str]:
    out = {}
    for n, chunk in batches(sorted(sources)):
        out |= labelled(
            wm.sparql(
                f"{name}_{n:02d}",
                f"""SELECT DISTINCT ?i ?iLabel WHERE {{ VALUES ?s {{ {values(chunk)} }}
                    ?s {props} ?i . {extra} {HAS_EN} {LABEL} }}""",
            )
        )
    return out


def princely_states(wm, dyn) -> set[str]:
    out = set()
    for n, chunk in batches(sorted(dyn)):
        rows = wm.sparql(
            f"dynasty_princely_{n:02d}",
            f'SELECT ?i WHERE {{ VALUES ?i {{ {values(chunk)} }} ?i wdt:P31 ?t . ?t rdfs:label "princely state"@en }}',
        )
        out |= {qid(r["i"]) for r in rows}
    return out


def linked_to(wm, name: str, sources, targets) -> set[str]:
    """Sources with a direct Wikidata statement to or from any of the targets."""
    targets, found = set(targets), set()
    for n, chunk in batches(sorted(sources), 25):
        rows = wm.sparql(
            f"{name}_{n:02d}",
            f"""SELECT DISTINCT ?s ?o WHERE {{ VALUES ?s {{ {values(chunk)} }}
                {{ ?s ?p ?o }} UNION {{ ?o ?p ?s }}
                FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/")) FILTER(isIRI(?o)) }}""",
        )
        found |= {qid(r["s"]) for r in rows if qid(r["o"]) in targets}
    return found


def capitals(wm, name: str, qids) -> dict[str, set]:
    out: dict[str, set] = defaultdict(set)
    for n, chunk in batches(sorted(qids)):
        for r in wm.sparql(
            f"{name}_{n:02d}", f"SELECT ?i ?c WHERE {{ VALUES ?i {{ {values(chunk)} }} ?i wdt:P36 ?c }}"
        ):
            out[qid(r["i"])].add(qid(r["c"]))
    return out


def districts_of(wm, name: str, qids) -> dict[str, set]:
    """Districts of India an entity lies in, following 'located in' (P131) upwards."""
    out: dict[str, set] = defaultdict(set)
    # Small batches: walking P131* for many items at once times out on the query service.
    for n, chunk in batches(sorted(qids), 40):
        for r in wm.sparql(
            f"{name}_{n:02d}",
            f"""SELECT DISTINCT ?i ?d WHERE {{ VALUES ?i {{ {values(chunk)} }}
                ?i wdt:P131* ?d . ?d wdt:P31 wd:{DISTRICT_OF_INDIA} . }}""",
        ):
            out[qid(r["i"])].add(qid(r["d"]))
    return out


def dynasty_end_years(wm, name: str, people, dynasties) -> dict[str, dict]:
    """For each person, the end year (P576) of every seeded dynasty they belong to (P53)."""
    out: dict[str, dict] = defaultdict(dict)
    if not people:
        return out
    for n, chunk in batches(sorted(people)):
        for r in wm.sparql(
            f"{name}_{n:02d}",
            f"""SELECT ?i ?dyn ?dynLabel ?end WHERE {{ VALUES ?i {{ {values(chunk)} }} VALUES ?dyn {{ {values(dynasties)} }}
                ?i wdt:P53 ?dyn . ?dyn wdt:P576 ?end . {LABEL} }}""",
        ):
            y = year_of(r.get("end"))
            if y is not None:
                label = r.get("dynLabel", qid(r["dyn"]))
                out[qid(r["i"])][label] = min(y, out[qid(r["i"])].get(label, y))
    return out


def year_of(value: str | None) -> int | None:
    """'1592-01-05T00:00:00Z' -> 1592; '-0304-01-01T00:00:00Z' -> -304; unknown values -> None."""
    m = re.match(r"^(-?\d+)-", value or "")
    return int(m.group(1)) if m else None


def born_before(years: dict, limit: int) -> bool:
    """Birth year before the limit; failing that, death year; failing that, floruit (P1317)."""
    for key in ("birth", "death", "floruit"):
        if key in years:
            return years[key] < limit
    return False


def date_basis(years: dict) -> str:
    for key in ("birth", "death", "floruit"):
        if key in years:
            return f"{key} {years[key]}"
    return "no dates"


def life_years(wm, people) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(dict)
    for n, chunk in batches(sorted(people)):
        rows = wm.sparql(
            f"people_dates_{n:02d}",
            f"""SELECT ?i ?b ?d ?f WHERE {{ VALUES ?i {{ {values(chunk)} }}
                OPTIONAL {{ ?i wdt:P569 ?b }} OPTIONAL {{ ?i wdt:P570 ?d }} OPTIONAL {{ ?i wdt:P1317 ?f }} }}""",
        )
        for r in rows:
            for key, var in (("birth", "b"), ("death", "d"), ("floruit", "f")):
                y = year_of(r.get(var))
                if y is not None:
                    out[qid(r["i"])][key] = min(y, out[qid(r["i"])].get(key, y))
    return out


TOPUP_CATEGORIES = ("Culture of", "Monuments and memorials in", "Festivals in")
TOPUP_SKIP_SUBCATS = re.compile(
    r"cuisine|food|cinema|film|people|organi[sz]ations|sport|media|television|education|politic|"
    r"writers|artists|musicians|stubs|images|by city|companies|books|novels",
    re.I,
)
TOPUP_EXCLUDE = re.compile(
    r"human|film|cuisine|dish|food|beverage|organi[sz]ation|company|business|university|college|school|institut|"
    r"television|newspaper|magazine|political|sport|club|award|museum|library|government|ministry|hospital|"
    r"airport|station|stadium|hotel|restaurant|band|album|song|book|website|literary work|group|diocese|administrative",
    re.I,
)
TOPUP_CULTURE = re.compile(
    r"festival|dance|music|tradition|ritual|craft|art form|theatre|cultural|ceremony|custom|textile|painting|"
    r"holiday|fair|celebration|observance|performing art",
    re.I,
)
TOPUP_SITE = re.compile(
    r"monument|memorial|temple|church|mosque|fort|palace|archaeolog|building|structure|tomb|stupa|monastery|"
    r"cave|shrine|heritage|statue|gurdwara|ruin|tower|gate|lighthouse|cathedral|basilica|chapel|mausoleum|"
    r"ghat|stepwell|garden|historic site",
    re.I,
)


def topup_candidates(wm, resolver: "StateResolver", state: str, exclude: set) -> list[tuple]:
    """Items in a state's culture, monument and festival categories, typed and checked on Wikidata."""
    slug = re.sub(r"\W+", "_", state.lower())
    place = {"National Capital Territory of Delhi": "Delhi"}.get(state, state)
    titles = set()
    for base in TOPUP_CATEGORIES:
        for name in (f"{base} {place}", f"{base} the {place}"):
            todo, depth = [name], 0
            while todo and depth < 2:
                next_level = []
                for cat in todo:
                    data = wm.wiki(
                        f"topup_{slug}_cat",
                        {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}", "cmlimit": "max"},
                    )
                    for m in data.get("query", {}).get("categorymembers", []):
                        if m["ns"] == 0:
                            titles.add(m["title"])
                        elif m["ns"] == 14 and not TOPUP_SKIP_SUBCATS.search(m["title"]):
                            next_level.append(m["title"][9:])
                todo, depth = next_level, depth + 1
    by_qid = {q: t for t, q in titles_to_qids(wm, f"topup_{slug}", titles).items() if q not in exclude}
    if not by_qid:
        return []
    resolver.load(f"topup_{slug}", by_qid)
    types = defaultdict(set)
    for n, chunk in batches(sorted(by_qid)):
        for r in wm.sparql(
            f"topup_{slug}_types_{n:02d}",
            f"""SELECT ?i ?tLabel WHERE {{ VALUES ?i {{ {values(chunk)} }} ?i wdt:P31 ?t .
                ?t rdfs:label ?tLabel FILTER(lang(?tLabel) = "en") }}""",
        ):
            types[qid(r["i"])].add(r["tLabel"])
    out = []
    for q, title in sorted(by_qid.items()):
        t = " | ".join(sorted(types[q]))
        # Overview articles ("Music of Meghalaya") describe a whole field, not one heritage item.
        overview = re.match(r"(Music|Culture|Dances?|Festivals|Cuisine|Art|Arts|Architecture) of ", title)
        if not t or overview or TOPUP_EXCLUDE.search(t) or not resolver.not_foreign(q):
            continue
        if TOPUP_CULTURE.search(t):
            out.append((q, title, "topup_culture", state))
        elif TOPUP_SITE.search(t):
            out.append((q, title, "topup_site", state))
    return out


def top_up(wm, resolver: "StateResolver", seeds: dict, rng: random.Random) -> tuple[list, list]:
    """Add seeds until each state or union territory has THIN_STATE seeds and one cultural seed, where items exist."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for s in seeds.values():
        if s.status == "selected":
            for st in s.states:
                counts[st][s.doc_type] += 1
    added, unfilled = [], []
    for st in sorted(resolver.states.values()):
        total, has_culture = sum(counts[st].values()), counts[st]["culture"] > 0
        if total >= THIN_STATE and has_culture:
            continue
        taken = set(seeds) | {a[0] for a in added}
        cands = topup_candidates(wm, resolver, st, taken)
        culture = [c for c in cands if c[2] == "topup_culture"]
        picks = []
        if not has_culture and culture:
            picks.append(rng.choice(culture))
        pool = [c for c in cands if c not in picks]
        rng.shuffle(pool)
        for c in pool:
            if total + len(picks) >= THIN_STATE:
                break
            picks.append(c)
        added += picks
        final_total = total + len(picks)
        final_culture = has_culture or any(c[2] == "topup_culture" for c in picks)
        if final_total < THIN_STATE or not final_culture:
            unfilled.append((st, final_total, "yes" if final_culture else "no"))
    return added, unfilled


def en_titles(wm, name: str, qids) -> dict[str, str]:
    """QID -> English Wikipedia article title."""
    out = {}
    for n, chunk in batches(sorted(qids)):
        for r in wm.sparql(
            f"{name}_{n:02d}",
            f"""SELECT ?i ?t WHERE {{ VALUES ?i {{ {values(chunk)} }}
                ?a schema:about ?i ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?t . }}""",
        ):
            out[qid(r["i"])] = r["t"]
    return out


def sitelinks(wm, qids) -> dict[str, set]:
    out = defaultdict(set)
    sites = " ".join(f"<https://{lang}.wikipedia.org/>" for lang in LANGS)
    for n, chunk in batches(sorted(qids)):
        for r in wm.sparql(
            f"sitelinks_{n:02d}",
            f"""SELECT ?i ?site WHERE {{ VALUES ?i {{ {values(chunk)} }} VALUES ?site {{ {sites} }}
                ?a schema:about ?i ; schema:isPartOf ?site . }}""",
        ):
            out[qid(r["i"])].add(r["site"].split("//")[1].split(".")[0])
    return out


# -------------------------------------------------------------------- main


def build() -> tuple[list[Seed], dict]:
    wm = Wikimedia(RAW_DIR)
    rng = random.Random(RANDOM_SEED)
    resolver = StateResolver(wm)
    seeds: dict[str, Seed] = {}
    notes: dict = {"states_found": len(resolver.states)}

    def add(q: str, label: str, stratum: str, sub: str = "", flags=(), status: str = "selected") -> Seed:
        if q in seeds:
            seeds[q].also_in.append(f"{stratum}/{sub}" if sub else stratum)
            seeds[q].flags += [f for f in flags if f not in seeds[q].flags]
            return seeds[q]
        seeds[q] = Seed(q, label, stratum, sub, flags=list(flags), status=status)
        return seeds[q]

    # Sites
    whs = designation(wm, "world_heritage", WORLD_HERITAGE)
    tentative = designation(wm, "tentative_list", TENTATIVE_LIST)
    for q, label in whs.items():
        add(q, label, "world_heritage")
    for q, label in tentative.items():
        add(q, label, "tentative_list")

    asi = {q: label for q, label in asi_monuments(wm).items() if q not in seeds}
    resolver.load("asi", asi)
    by_state = defaultdict(list)
    for q in asi:
        probe = Seed(q, asi[q], "asi_monument")
        resolver.assign(probe)
        by_state[probe.state or "unresolved"].append(q)
    quota = allocate(
        {s: len(v) for s, v in by_state.items()},
        ASI_TOTAL,
        ASI_MIN_PER_STATE,
        floor_keys=[s for s in by_state if s in resolver.states.values()],
    )
    notes["asi_available"] = {s: len(v) for s, v in by_state.items()}
    notes["asi_quota"] = quota
    for s in sorted(by_state):
        for q in rng.sample(sorted(by_state[s]), quota[s]):
            add(q, asi[q], "asi_monument")

    # Dynasties and historical states. Colonial administrative entities are
    # removed (author decision, recorded in DYNASTY_SCAN_FILE); princely states
    # wait until the other seeds exist, because they are kept only when linked to one.
    dyn_removed = {
        r["qid"] for r in csv.DictReader(open(DYNASTY_SCAN_FILE, encoding="utf-8")) if r["decision"] == "remove"
    }
    all_dynasties = {q: v for q, v in dynasties(wm).items() if q not in dyn_removed}
    princely = princely_states(wm, all_dynasties)
    notes["dynasties_removed"] = len(dyn_removed)
    for q, (label, rule) in all_dynasties.items():
        if q not in princely:
            add(q, label, "dynasty", flags=[f"rule: {rule}"])

    # Culture
    scan = list(csv.DictReader(open(SCAN_FILE, encoding="utf-8")))
    removed = {r["qid"] for r in scan if r["decision"] == "remove"}
    hints = {r["qid"]: r["state_hint"] for r in scan if r["decision"] == "keep"}

    for q, label in intangible_heritage(wm).items():
        add(q, label, "culture", "intangible_heritage")
    for sub in ("classical_dance", "classical_music"):
        for r in scan:
            if r["sub_stratum"] == sub and r["decision"] == "keep":
                add(
                    r["qid"],
                    r["label"],
                    "culture",
                    sub,
                    flags=["recognised classical form"] if r["recognised_classical"] else [],
                )

    folk = {q: label for q, label in folk_dances(wm).items() if q not in removed}
    for r in scan:
        if r["sub_stratum"] == "folk_dance" and r["decision"] == "keep":
            folk[r["qid"]] = r["label"]
    folk = {q: label for q, label in folk.items() if q not in seeds}
    resolver.load("folk_dance", folk)
    resolver.load_categories(wm, "folk_dance", folk)
    # Forms the author chose by hand are always in; the rest are drawn by state in turn.
    chosen = [r["qid"] for r in scan if r["sub_stratum"] == "folk_dance" and r["decision"] == "keep"]
    for q in chosen:
        if q in folk:
            add(q, folk[q], "culture", "folk_dance")
    folk_by_state = defaultdict(list)
    for q in folk:
        if q in seeds:
            continue
        probe = Seed(q, folk[q], "culture")
        resolver.assign(probe, hints.get(q, ""))
        folk_by_state[probe.state or "unresolved"].append(q)
    notes["folk_dance_available"] = len(folk)
    for q in round_robin(folk_by_state, FOLK_DANCE_CAP - sum(1 for q in chosen if q in seeds), rng):
        add(q, folk[q], "culture", "folk_dance")

    for sub in ("folk_music", "folk_theatre"):
        for r in scan:
            if r["sub_stratum"] == sub and r["decision"] == "keep":
                add(r["qid"], r["label"], "culture", sub)

    painting = painting_candidates(wm)
    resolver.load("painting", painting)
    typed = painting_filter(wm, painting)
    notes["painting_candidates"] = len(painting)
    notes["painting_typed"] = len(typed)
    for q, label in sorted(typed.items(), key=lambda kv: kv[1]):
        if resolver.not_foreign(q) and q not in removed:
            add(q, label, "culture", "painting_school")

    gi_rows = gi_crafts(wm)
    notes["gi_rows"] = len(gi_rows)
    notes["gi_types"] = dict(Counter(r["type"] for r in gi_rows).most_common())
    # The table spells the type several ways ("Handicraft", "Handicrafts", "Handi Crafts", "Textiles").
    craft_type = re.compile(r"^(handicraft|textile)", re.I)
    indian = set(resolver.states.values()) | {"Delhi", "Orissa"}

    def in_indian_state(state_cell: str) -> bool:
        return any(part.strip() in indian for part in re.split(r",|/|&| and ", state_cell))

    crafts = [
        r
        for r in gi_rows
        if craft_type.match(r["type"].replace(" ", "")) and r["title"] and in_indian_state(r["state"])
    ]
    title_qid = titles_to_qids(wm, "gi_links", [r["title"] for r in crafts])
    gi_state, gi_label = {}, {}
    for r in crafts:
        q = title_qid.get(r["title"])
        if q and q not in seeds:
            gi_state.setdefault(q, r["state"])
            gi_label.setdefault(q, r["title"])
    resolver.load("gi", gi_state)
    gi_ok = {q for q in gi_state if resolver.not_foreign(q) and q not in removed}
    notes["gi_handicraft_rows"] = len(crafts)
    notes["gi_linked_items"] = len(gi_state)
    notes["gi_in_india"] = len(gi_ok)
    gi_by_state = defaultdict(list)
    for q in gi_ok:
        gi_by_state[gi_state[q]].append(q)
    for q in round_robin(gi_by_state, GI_CAP, rng):
        hints[q] = gi_state[q]
        add(q, gi_label[q], "culture", "gi_craft")

    # Styles reached in one hop from the seeded sites
    sites = [q for q, s in seeds.items() if s.doc_type == "site"]
    for q, label in one_hop(wm, "styles", sites, "wdt:P149|wdt:P135").items():
        add(q, label, "style")

    # Princely states (author decision): kept when a Wikidata statement links them to a
    # site or cultural seed, or when their capital lies in the same district as a seeded site.
    anchors = [q for q, s in seeds.items() if s.doc_type in ("site", "culture")]
    linked = linked_to(wm, "princely_links", princely, anchors)
    capital = capitals(wm, "princely_capitals", princely)
    cap_districts = districts_of(wm, "princely_capital_districts", {c for cs in capital.values() for c in cs})
    site_districts = set().union(*districts_of(wm, "site_districts", sites).values())
    same_district = {
        q for q in princely if any(cap_districts.get(c, set()) & site_districts for c in capital.get(q, ()))
    }
    for q in sorted(princely):
        if q in linked or q in same_district:
            label, rule = all_dynasties[q]
            why = "linked to another seed" if q in linked else "capital in the district of a seeded site"
            add(q, label, "dynasty", flags=[f"rule: {rule}", f"princely state: {why}"])
    kept_princely = linked | same_district
    notes["princely"] = {
        "total": len(princely),
        "kept": len(kept_princely),
        "by_link": len(linked),
        "by_district": len(same_district - linked),
        "dropped": len(princely) - len(kept_princely),
        "capital_without_district": sum(
            1 for q in princely if not any(cap_districts.get(c) for c in capital.get(q, ()))
        ),
    }

    # People (author decision): historical figures who commissioned, designed or founded a
    # seeded site, or belonged to a seeded dynasty; dated before 1900 by birth, death or
    # floruit, or, with no dates at all, members of a seeded dynasty that ended before 1900.
    dyn = [q for q, s in seeds.items() if s.stratum == "dynasty"]
    people = one_hop(wm, "people_sites", sites, "wdt:P88|wdt:P84|wdt:P112", "?i wdt:P31 wd:Q5 .")
    people |= one_hop(wm, "people_dynasties", dyn, "^wdt:P53", "?i wdt:P31 wd:Q5 .")
    years = life_years(wm, people)
    undated = [q for q in people if not years.get(q)]
    ended = dynasty_end_years(wm, "people_dynasty_ends", undated, dyn)
    kept, via_dynasty, borderline = {}, [], []
    for q in sorted(people):
        if born_before(years.get(q, {}), PEOPLE_BORN_BEFORE):
            kept[q] = date_basis(years[q])
            if max(years[q].values()) >= PEOPLE_BORN_BEFORE - 50:
                borderline.append(f"{people[q]} ({date_basis(years[q])})")
        elif not years.get(q) and any(y < PEOPLE_BORN_BEFORE for y in ended.get(q, {}).values()):
            name, end = min(ended[q].items(), key=lambda kv: kv[1])
            kept[q] = f"no dates; member of {name}, ended {end}"
            via_dynasty.append(f"{people[q]} ({kept[q]})")
    notes["people"] = {
        "candidates": len(people),
        "kept": len(kept),
        "by_dates": len(kept) - len(via_dynasty),
        "by_dynasty_end": len(via_dynasty),
        "undated_left_out": sum(1 for q in undated if q not in kept),
        "borderline": borderline,
        "via_dynasty": via_dynasty,
    }
    for q, basis in kept.items():
        add(q, people[q], "person", flags=[basis])

    # States, then a top-up for states and union territories that are still thin
    def place_all() -> None:
        resolver.load("all", seeds)
        resolver.load_categories(wm, "culture", [q for q, s in seeds.items() if s.doc_type == "culture"])
        for s in seeds.values():
            resolver.assign(s, hints.get(s.qid, ""))

    place_all()
    added, unfilled = top_up(wm, resolver, seeds, rng)
    for q, label, kind, state in added:
        hints[q] = state
        add(q, label, kind)
    notes["topup"], notes["topup_unfilled"] = added, unfilled
    place_all()
    links = sitelinks(wm, seeds)
    for q, s in seeds.items():
        s.sitelinks = links.get(q, set())

    notes["gi_note"] = "The GI article carried a 'needs updating' notice (April 2025) when fetched."
    notes["fetched"], notes["reused"] = wm.fetched, wm.reused
    notes["state_names"] = sorted(resolver.states.values())
    return list(seeds.values()), notes


def write_csv(seeds: list[Seed]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{RUN}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "label", "stratum", "sub_stratum", "doc_type", "status", "state", "states",
                    "state_source", "en", "hi", "kn", "also_in", "flags"])  # fmt: skip
        for s in sorted(seeds, key=lambda s: (list(DOC_TYPE).index(s.stratum), s.sub_stratum, s.label)):
            w.writerow([s.qid, s.label, s.stratum, s.sub_stratum, s.doc_type, s.status, s.state, "; ".join(s.states),
                        s.state_source, *("yes" if lang in s.sitelinks else "" for lang in LANGS),
                        "; ".join(s.also_in), "; ".join(s.flags)])  # fmt: skip


def table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def write_report(seeds: list[Seed], notes: dict) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    selected = [s for s in seeds if s.status == "selected"]
    proposed = [s for s in seeds if s.status == "proposed"]
    types = ["site", "culture", "person_dynasty_style"]

    def key(s):
        return f"{s.stratum}/{s.sub_stratum}" if s.sub_stratum else s.stratum

    strata = Counter(key(s) for s in selected)
    by_stratum = [
        [k, n] + [sum(1 for s in selected if key(s) == k and lang in s.sitelinks) for lang in LANGS]
        for k, n in sorted(strata.items(), key=lambda kv: list(DOC_TYPE).index(kv[0].split("/")[0]))
    ]
    per_state = defaultdict(Counter)
    for s in selected:
        for st in s.states or ["(no state)"]:
            per_state[st][s.doc_type] += 1
    rows = []
    for st in notes["state_names"] + sorted(set(per_state) - set(notes["state_names"])):
        c = per_state.get(st, Counter())
        rows.append([st] + [c[t] for t in types] + [sum(c.values())])
    thin = [r for r in rows if r[0] in notes["state_names"] and r[-1] < THIN_STATE]
    no_culture = [r[0] for r in rows if r[0] in notes["state_names"] and r[2] == 0]
    no_site = [r[0] for r in rows if r[0] in notes["state_names"] and r[1] == 0]

    def listing(stratum, sub=""):
        return "; ".join(
            f"{s.label} ({s.state or 'no state'})"
            for s in sorted(seeds, key=lambda s: s.label)
            if s.stratum == stratum and (not sub or s.sub_stratum == sub)
        )

    out = [
        f"# Seed report, {RUN}",
        "",
        f"Generated by `src/corpus/seeds.py` on {date.today().isoformat()} at commit `{commit}`, from the rules in"
        " decision 0002. Nothing has been collected yet.",
        "",
        f"{len(selected)} seed entities are selected; {len(proposed)} people are proposed and not yet selected."
        f" Wikidata lists {notes['states_found']} current states and union territories.",
        "",
        "## By stratum, with Wikipedia articles available",
        "",
        *table(["Stratum", "Seeds", "en", "hi", "kn"], by_stratum),
        "",
        "## By document type",
        "",
        *table(["Document type", "Seeds"], [[t, sum(1 for s in selected if s.doc_type == t)] for t in types]),
        "",
        "Objects and manuscripts or texts are not seed entities: they come from museum and archive records"
        " matched to these seeds, and are counted by the collector probes.",
        "",
        "## By state and union territory",
        "",
        "An entity that spans several states is counted in each of them.",
        "",
        *table(["State or union territory", *types, "Total"], rows),
        "",
        "## Coverage gaps",
        "",
        f"- Fewer than {THIN_STATE} seeds: " + (", ".join(f"{r[0]} ({r[-1]})" for r in thin) or "none"),
        "- No cultural seed: " + (", ".join(no_culture) or "none"),
        "- No site seed: " + (", ".join(no_site) or "none"),
        "",
        "## Selection details",
        "",
        f"- ASI monuments available by state: {notes['asi_available']}",
        f"- ASI places allocated by state: {notes['asi_quota']}",
        f"- Folk dance candidates: {notes['folk_dance_available']}; selected up to {FOLK_DANCE_CAP} by state in turn.",
        f"- Painting school candidates: {notes['painting_candidates']}; typed as a style, school, movement or genre:"
        f" {notes['painting_typed']}; kept if in India.",
        f"- GI table rows: {notes['gi_rows']}, by type {notes['gi_types']}; handicraft rows with an article:"
        f" {notes['gi_handicraft_rows']}; distinct linked items: {notes['gi_linked_items']}; not placed in another country on Wikidata:"
        f" {notes['gi_in_india']}; selected up to {GI_CAP} by state in turn. {notes['gi_note']}",
        f"- Dynasties: {notes['dynasties_removed']} colonial administrative entities removed (author decision)."
        f" Princely states: {notes['princely']['total']}; kept {notes['princely']['kept']}"
        f" ({notes['princely']['by_link']} linked to another seed, {notes['princely']['by_district']} with the capital"
        f" in the district of a seeded site); dropped {notes['princely']['dropped']};"
        f" capital not placed in any district on Wikidata: {notes['princely']['capital_without_district']}.",
        f"- People: {notes['people']['candidates']} candidates; kept {notes['people']['kept']}"
        f" ({notes['people']['by_dates']} by a birth, death or floruit date before {PEOPLE_BORN_BEFORE},"
        f" {notes['people']['by_dynasty_end']} undated members of a seeded dynasty that ended before {PEOPLE_BORN_BEFORE});"
        f" undated and left out: {notes['people']['undated_left_out']}.",
        "- People near the date limit (a date from 1850 on): " + ("; ".join(notes["people"]["borderline"]) or "none"),
        "- People kept through a dynasty end date: " + ("; ".join(notes["people"]["via_dynasty"]) or "none"),
        "- Top-up for thin states and union territories (from 'Culture of', 'Monuments and memorials in' and"
        " 'Festivals in' categories): "
        + (
            "; ".join(f"{label} ({state}, {kind.removeprefix('topup_')})" for _, label, kind, state in notes["topup"])
            or "none"
        ),
        "- Places that could not be filled to 5 seeds and one cultural seed, because Wikipedia and Wikidata"
        " hold no more qualifying items for them (state, seeds, has cultural seed): "
        + ("; ".join(f"{st} ({n}, {c})" for st, n, c in notes["topup_unfilled"]) or "none"),
        f"- API responses fetched: {notes['fetched']}; reused from cache: {notes['reused']}.",
        "",
        "## Lists for hand scan",
        "",
        f"**Dynasties and historical states:** {listing('dynasty')}",
        "",
        f"**Folk dance:** {listing('culture', 'folk_dance')}",
        "",
        f"**Painting schools:** {listing('culture', 'painting_school')}",
        "",
        f"**GI crafts:** {listing('culture', 'gi_craft')}",
        "",
        f"**Styles:** {listing('style')}",
        "",
        f"**People:** {listing('person')}",
        "",
    ]
    (OUT_DIR / "seed_report.md").write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    seeds, notes = build()
    write_csv(seeds)
    write_report(seeds, notes)
    selected = sum(1 for s in seeds if s.status == "selected")
    print(f"{selected} seeds selected, {len(seeds) - selected} proposed; wrote data/seeds/{RUN}.csv and seed_report.md")


if __name__ == "__main__":
    main()
