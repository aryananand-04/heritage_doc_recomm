"""Gazetteer: Wikidata labels and aliases (en, hi, kn) for every entity a document can link to.

Shared by object linking (decision 0017) and, later, query entity linking
(specification, Section 5). Each entry carries one entity type, so a field is
matched only against the types it can name:

- place: Indian states, union territories, districts, cities and towns (Wikidata
  classes, country India);
- polity: dynasties, historical countries, kingdoms and princely states in the
  entity lists;
- style, culture, person, site: from the entity lists' strata.

Usage:
    uv run python -m src.corpus.gazetteer

Output:
    data/gazetteer/gazetteer_v1.jsonl.gz   one entry per entity: qid, type, names by language
    results/linking/gazetteer_v1.json      counts by type and language
"""

import csv
import gzip
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime

from src.corpus.seeds import batches
from src.corpus.wikimedia import ROOT, Wikimedia, qid, values

RAW_DIR = ROOT / "data/raw/gazetteer_v1"
OUT = ROOT / "data/gazetteer/gazetteer_v1.jsonl.gz"
REPORT = ROOT / "results/linking/gazetteer_v1.json"
INDIA = "Q668"
LANGS = ("en", "hi", "kn")
# Administrative place classes (state and union territory as in seeds.py's StateResolver;
# the others as found by the neighbour probe).
PLACE_CLASSES = {
    "Q12443800": "state of India",
    "Q467745": "union territory",
    "Q1149652": "district of India",
    "Q515": "city",
    "Q1549591": "big city",
    "Q3957": "town",
    "Q58339518": "town in India",
}
ENTITY_LISTS = ["seeds_v1.csv", "asi_remainder_v1.csv", "neighbours_v1.csv", "class_list_v1.csv"]
STRATUM_TYPE = {
    "dynasty": "polity",
    "style": "style",
    "person": "person",
    "culture": "culture",
    "topup_culture": "culture",
}
NEIGHBOUR_TYPE = {
    "dynasty": "polity",
    "historical country": "polity",
    "princely state": "polity",
    "architectural style": "style",
    "person": "person",
    "raga": "culture",
    "thaat": "culture",
    "festival": "culture",
    "music festival": "culture",
    "music genre": "culture",
    "type of dance": "culture",
}


def entry_type(row: dict) -> str:
    if row["stratum"] == "neighbour":
        return NEIGHBOUR_TYPE.get(row["sub_stratum"].split(",")[0], "site")
    if row["stratum"] in STRATUM_TYPE:
        return STRATUM_TYPE[row["stratum"]]
    return "site" if row["doc_type"] == "site" else row["doc_type"]


def main() -> None:
    wm = Wikimedia(RAW_DIR)
    types: dict[str, str] = {}
    for name in ENTITY_LISTS:
        for r in csv.DictReader(open(ROOT / "data/seeds" / name, encoding="utf-8")):
            if r.get("status", "selected") == "selected":
                types.setdefault(r["qid"], entry_type(r))

    for c in PLACE_CLASSES:
        for r in wm.sparql(f"places_{c}", f"SELECT DISTINCT ?i WHERE {{ ?i wdt:P31 wd:{c} ; wdt:P17 wd:{INDIA} . }}"):
            types.setdefault(qid(r["i"]), "place")
    types.setdefault(INDIA, "place")

    names: dict[str, dict[str, set]] = {q: {lang: set() for lang in LANGS} for q in types}
    labels: dict[str, dict[str, str]] = {}
    for n, chunk in batches(sorted(types), 150):
        rows = wm.sparql(
            f"names_{n:03d}",
            # The client keeps only values, so the language is selected as its own variable.
            f"""SELECT ?i ?l ?lang ?kind WHERE {{ VALUES ?i {{ {values(chunk)} }}
                {{ ?i rdfs:label ?l BIND("label" AS ?kind) }} UNION {{ ?i skos:altLabel ?l BIND("alias" AS ?kind) }}
                BIND(LANG(?l) AS ?lang) FILTER(?lang IN ("en", "hi", "kn")) }}""",
        )
        for r in rows:
            names[qid(r["i"])][r["lang"]].add(r["l"])
            if r["kind"] == "label":
                labels.setdefault(qid(r["i"]), {})[r["lang"]] = r["l"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for q in sorted(types):
            entry = {
                "qid": q,
                "type": types[q],
                "label": labels.get(q, {}),
                "names": {lang: sorted(names[q].get(lang, ())) for lang in LANGS},
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "date": datetime.now(UTC).date().isoformat(),
        "commit": commit,
        "entries": len(types),
        "by_type": dict(Counter(types.values()).most_common()),
        "entries_with_names": {lang: sum(1 for q in types if names[q].get(lang)) for lang in LANGS},
        "names": {lang: sum(len(names[q].get(lang, ())) for q in types) for lang in LANGS},
        "requests": {"fetched": wm.fetched, "reused": wm.reused},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
