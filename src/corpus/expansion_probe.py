"""Count-only probe for growing the corpus beyond the entity lists (after decision 0015).

Nothing is collected. Counts for:
- museum objects collected by place (India) rather than by seed name: Met, Cleveland,
  Art Institute of Chicago (public-domain or CC0 only);
- a class-based Wikidata list: Indian items of heritage building and site classes with
  a Wikipedia article in English, Hindi or Kannada, and how many are not already in the
  three entity lists, and how many have a Hindi or Kannada article but no English one;
- public-domain reference series on the Internet Archive (dated before 1950).

Usage:
    uv run python -m src.corpus.expansion_probe

Outputs:
    results/probes/expansion_v1.json
    results/probes/expansion_v1.md
"""

import csv
import json
import subprocess
from datetime import date

from src.corpus.collection_probe import CLEVELAND, IA, IA_FREE, MET
from src.corpus.wikimedia import ROOT, Wikimedia, qid

RAW_DIR = ROOT / "data/raw/expansion_probe_v1"
OUT_DIR = ROOT / "results/probes"
CHICAGO = "https://api.artic.edu/api/v1/artworks/search"
LISTS = ["seeds_v1.csv", "asi_remainder_v1.csv", "neighbours_v1.csv"]
INDIA = "Q668"

# Heritage classes, taken from the classes the neighbour probe found (results/probes/neighbours_v1.json).
CLASSES = {
    "Q842402": "Hindu temple",
    "Q44539": "temple",
    "Q58621988": "temple complex",
    "Q2613100": "Jain temple",
    "Q32815": "mosque",
    "Q16970": "church building",
    "Q1785071": "fort",
    "Q57821": "fortification",
    "Q57831": "fortress",
    "Q16560": "palace",
    "Q381885": "tomb",
    "Q162875": "mausoleum",
    "Q1473950": "stepwell",
    "Q839954": "archaeological site",
}
IA_SERIES = {
    "Archaeological Survey of India reports": 'title:("Archaeological Survey of India")',
    "Imperial Gazetteer of India": 'title:("Imperial Gazetteer of India")',
    "District gazetteers": 'title:("District Gazetteer") AND subject:(India)',
    "Epigraphia Indica": 'title:("Epigraphia Indica")',
    "Indian Antiquary": 'title:("Indian Antiquary")',
}


def class_items(wm: Wikimedia, c: str) -> dict[str, set]:
    """Indian items of one class with their Wikipedia languages among en, hi, kn."""
    rows = wm.sparql(
        f"class_{c}",
        f"""SELECT ?i ?wiki WHERE {{ ?i wdt:P31 wd:{c} ; wdt:P17 wd:{INDIA} .
            OPTIONAL {{ ?a schema:about ?i ; schema:isPartOf ?wiki .
                       VALUES ?wiki {{ <https://en.wikipedia.org/> <https://hi.wikipedia.org/> <https://kn.wikipedia.org/> }} }} }}""",
    )
    out: dict[str, set] = {}
    for r in rows:
        langs = out.setdefault(qid(r["i"]), set())
        if r.get("wiki"):
            langs.add(r["wiki"].split("//")[1].split(".")[0])
    return out


def main() -> None:
    wm = Wikimedia(RAW_DIR)
    report: dict = {"date": date.today().isoformat()}

    # Museums by place, public domain only. The Met's search needs a query term; "India"
    # with the geoLocation filter keeps objects whose place is India.
    met_geo = wm.get_json("met_geo_india", MET, {"q": "India", "geoLocation": "India", "isPublicDomain": "true"})
    met_dept = wm.get_json("met_dept6_india", MET, {"q": "India", "departmentId": 6, "isPublicDomain": "true"})
    cle = {
        c: (
            wm.get_json(f"cleveland_culture_{c}", CLEVELAND, {"culture": c, "cc0": "1", "limit": 1}).get("info") or {}
        ).get("total")
        for c in ("India", "South Asia")
    }
    # The Art Institute's API sits behind a bot check and asks clients to identify
    # themselves in this header; its search takes an Elasticsearch query as JSON.
    wm.session.headers["AIC-User-Agent"] = wm.session.headers["User-Agent"]
    chi_query = {
        "query": {"bool": {"must": [{"term": {"is_public_domain": True}}, {"match": {"place_of_origin": "India"}}]}},
        "fields": ["id"],
        "limit": 0,
    }
    chi = wm.get_json("chicago_india", CHICAGO, {"params": json.dumps(chi_query, sort_keys=True)})
    report["museums_by_place"] = {
        "met_geoLocation_India": met_geo.get("total"),
        "met_asian_art_India": met_dept.get("total"),
        "cleveland_culture": cle,
        "chicago_place_of_origin_India": (chi.get("pagination") or {}).get("total"),
    }

    # Class-based Wikidata list
    listed = set()
    for name in LISTS:
        listed |= {r["qid"] for r in csv.DictReader(open(ROOT / "data/seeds" / name, encoding="utf-8"))}
    items: dict[str, set] = {}
    per_class = {}
    for c, label in CLASSES.items():
        found = class_items(wm, c)
        per_class[label] = {
            "items": len(found),
            "with_article": sum(1 for v in found.values() if v),
            "en": sum(1 for v in found.values() if "en" in v),
            "hi": sum(1 for v in found.values() if "hi" in v),
            "kn": sum(1 for v in found.values() if "kn" in v),
        }
        for q, v in found.items():
            items.setdefault(q, set()).update(v)
    with_article = {q for q, v in items.items() if v}
    new = with_article - listed
    report["class_based_list"] = {
        "per_class": per_class,
        "distinct_items": len(items),
        "distinct_with_article": len(with_article),
        "not_in_entity_lists": len(new),
        "new_articles_by_language": {lang: sum(1 for q in new if lang in items[q]) for lang in ("en", "hi", "kn")},
        "new_with_hi_or_kn_but_no_en": sum(1 for q in new if "en" not in items[q]),
    }

    # Internet Archive reference series, public domain, dated before 1950
    series = {}
    for label, q in IA_SERIES.items():
        params = {
            "q": f"{q} AND mediatype:texts AND {IA_FREE} AND date:[1700-01-01 TO 1949-12-31]",
            "rows": 0,
            "output": "json",
        }
        series[label] = (
            wm.get_json(f"ia_series_{label[:20].replace(' ', '_')}", IA, params).get("response", {}).get("numFound")
        )
    report["internet_archive_series_before_1950"] = series

    report["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    report["requests"] = {"fetched": wm.fetched, "reused": wm.reused}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "expansion_v1.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")

    m, cl = report["museums_by_place"], report["class_based_list"]
    lines = [
        "# Expansion probe, v1",
        "",
        f"Generated by `src/corpus/expansion_probe.py` on {report['date']} at commit `{report['commit']}`. Counts only.",
        "",
        "## Museum objects by place (public domain or CC0)",
        "",
        "| Source | Objects |",
        "| --- | --- |",
        f"| Met, geoLocation India | {m['met_geoLocation_India']} |",
        f"| Met, Asian Art department, 'India' | {m['met_asian_art_India']} |",
        *[f"| Cleveland, culture '{c}' | {n} |" for c, n in m["cleveland_culture"].items()],
        f"| Art Institute of Chicago, place of origin India | {m['chicago_place_of_origin_India']} |",
        "",
        "Search totals are upper bounds before any filtering. The Met's search ignores its public-domain filter"
        " (seen in the seed-name collection, where 404 of about 1,000 objects looked at were not public domain),"
        " so its public-domain share is known only after each object is fetched.",
        "",
        "## Class-based Wikidata list (country India)",
        "",
        "| Class | Items | With an article | en | hi | kn |",
        "| --- | --- | --- | --- | --- | --- |",
        *[
            f"| {k} | {v['items']} | {v['with_article']} | {v['en']} | {v['hi']} | {v['kn']} |"
            for k, v in sorted(cl["per_class"].items(), key=lambda kv: -kv[1]["items"])
        ],
        "",
        f"Distinct items: {cl['distinct_items']}; with an en, hi or kn article: {cl['distinct_with_article']};"
        f" not already in the entity lists: {cl['not_in_entity_lists']}"
        f" (en {cl['new_articles_by_language']['en']}, hi {cl['new_articles_by_language']['hi']},"
        f" kn {cl['new_articles_by_language']['kn']}; Hindi or Kannada only: {cl['new_with_hi_or_kn_but_no_en']}).",
        "",
        "## Internet Archive reference series (public domain, dated 1700–1949)",
        "",
        "| Series | Items |",
        "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in series.items()],
        "",
    ]
    (OUT_DIR / "expansion_v1.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
