"""Count-only probe: how many entities lie one hop from the seeds (decision 0005).

Nothing is collected. For every selected seed, the probe asks Wikidata which
entities the seed links to, and which entities link to the seed, along the core
relations of decision 0006. It then checks each neighbour for the India scope
rule (decision 0001) and for a Wikipedia article in English, Hindi or Kannada.
The counts fix the size of the corpus expansion before any collector runs.

Usage:
    uv run python -m src.corpus.neighbour_probe

Outputs:
    results/probes/neighbours_v1.json   counts per relation, scope and language; hubs; drift sample
    results/probes/neighbours_v1.md     summary
"""

import csv
import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import date

from src.corpus.seeds import batches
from src.corpus.wikimedia import ROOT, Wikimedia, qid, values

RAW_DIR = ROOT / "data/raw/neighbours_probe_v1"
OUT_DIR = ROOT / "results/probes"
SEEDS = ROOT / "data/seeds/seeds_v1.csv"
RANDOM_SEED = 20261119
DRIFT_SAMPLE = 50  # neighbours listed for the author's scope-drift hand check
TOP_HUBS = 15
INDIA = "Q668"

# Core relations (decision 0006). Each is followed in both directions: "out" is
# seed -> neighbour, "in" is neighbour -> seed.
RELATIONS = {
    "P131": "located in the administrative entity",
    "P276": "location",
    "P361": "part of",
    "P527": "has part",
    "P88": "commissioned by",
    "P84": "architect",
    "P112": "founded by",
    "P53": "family (dynasty)",
    "P36": "capital",
    "P149": "architectural style",
    "P136": "genre",
    "P135": "movement",
    "P1365": "replaces",
    "P1366": "replaced by",
    "P800": "notable work",
    "P170": "creator",
}
DIRECTIONS = ("out", "in")
LANGS = ("en", "hi", "kn")


def edge_query(chunk: list[str], direction: str) -> str:
    props = " ".join(f"wdt:{p}" for p in RELATIONS)
    triple = "?s ?p ?n ." if direction == "out" else "?n ?p ?s ."
    return f"""SELECT DISTINCT ?s ?p ?n WHERE {{ VALUES ?s {{ {values(chunk)} }} VALUES ?p {{ {props} }}
        {triple} FILTER(isIRI(?n)) FILTER(STRSTARTS(STR(?n), "http://www.wikidata.org/entity/Q")) }}"""


def attribute_query(chunk: list[str]) -> str:
    wiki = " ".join(
        f"BIND(EXISTS {{ ?a_{lang} schema:about ?n ; schema:isPartOf <https://{lang}.wikipedia.org/> }} AS ?{lang})"
        for lang in LANGS
    )
    return f"""SELECT ?n ?nLabel ?india ?country ?human {" ".join("?" + lang for lang in LANGS)} WHERE {{
        VALUES ?n {{ {values(chunk)} }}
        BIND(EXISTS {{ ?n wdt:P17 wd:{INDIA} }} AS ?india)
        BIND(EXISTS {{ ?n wdt:P17 ?c }} AS ?country)
        BIND(EXISTS {{ ?n wdt:P31 wd:Q5 }} AS ?human)
        {wiki}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}"""


def truthy(value: str) -> bool:
    return value in ("true", "1")


def scope_of(attrs: dict) -> str:
    """India scope class (decision 0001): country India; no country at all, which
    people, dynasties and styles usually lack and which needs a rule; or another country."""
    if attrs["india"]:
        return "india"
    if not attrs["country"]:
        return "no_country"
    return "other_country"


def summarise(edges: list[tuple[str, str, str, str]], attrs: dict[str, dict], seeds: set[str]) -> dict:
    """Counts from (seed, property, neighbour, direction) edges and per-neighbour attributes.

    Seeds are never counted as neighbours: an edge between two seeds adds a graph
    link, not a new document.
    """
    neighbours: dict[str, set] = defaultdict(set)  # neighbour -> seeds it links to
    by_relation: dict[str, set] = defaultdict(set)
    for s, p, n, d in edges:
        if n in seeds:
            continue
        neighbours[n].add(s)
        by_relation[f"{p} {d}"].add(n)

    def profile(qids) -> dict:
        qids = [q for q in qids if q in attrs]
        scopes = Counter(scope_of(attrs[q]) for q in qids)
        in_scope = [q for q in qids if scope_of(attrs[q]) != "other_country"]
        return {
            "entities": len(qids),
            "scope": dict(sorted(scopes.items())),
            "humans": sum(1 for q in qids if attrs[q]["human"]),
            "in_scope_with_article": {lang: sum(1 for q in in_scope if attrs[q][lang]) for lang in LANGS},
            "in_scope_with_any_article": sum(1 for q in in_scope if any(attrs[q][lang] for lang in LANGS)),
            "india_with_any_article": sum(
                1 for q in qids if attrs[q]["india"] and any(attrs[q][lang] for lang in LANGS)
            ),
        }

    hubs = sorted(neighbours.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:TOP_HUBS]
    return {
        "all": profile(neighbours),
        "by_relation": {
            key: {"label": f"{RELATIONS[key.split()[0]]} ({key.split()[1]})", **profile(qs)}
            for key, qs in sorted(by_relation.items(), key=lambda kv: -len(kv[1]))
        },
        "hubs": [{"qid": q, "label": attrs.get(q, {}).get("label", ""), "seeds_linked": len(ss)} for q, ss in hubs],
    }


def main() -> None:
    rows = [r for r in csv.DictReader(open(SEEDS, encoding="utf-8")) if r["status"] == "selected"]
    seeds = {r["qid"] for r in rows}
    wm = Wikimedia(RAW_DIR)

    edges = []
    for direction in DIRECTIONS:
        # Small batches: the "in" direction can return thousands of rows per seed.
        for n, chunk in batches(sorted(seeds), 50):
            for r in wm.sparql(f"edges_{direction}_{n:02d}", edge_query(chunk, direction)):
                edges.append((qid(r["s"]), qid(r["p"]), qid(r["n"]), direction))

    found = sorted({n for _, _, n, _ in edges} - seeds)
    attrs = {}
    for n, chunk in batches(found, 200):
        for r in wm.sparql(f"attrs_{n:03d}", attribute_query(chunk)):
            attrs[qid(r["n"])] = {
                "label": r.get("nLabel", ""),
                "india": truthy(r["india"]),
                "country": truthy(r["country"]),
                "human": truthy(r["human"]),
                **{lang: truthy(r[lang]) for lang in LANGS},
            }

    summary = summarise(edges, attrs, seeds)
    candidates = sorted(
        q for q, a in attrs.items() if scope_of(a) != "other_country" and any(a[lang] for lang in LANGS)
    )
    # The Wikidata classes (P31) of the candidates, so the rule for which neighbours
    # become documents is set from the classes actually present, not guessed.
    classes: dict[str, list[str]] = defaultdict(list)
    class_labels: dict[str, str] = {}
    for n, chunk in batches(candidates, 200):
        for r in wm.sparql(
            f"classes_{n:03d}",
            f"""SELECT ?n ?c ?cLabel WHERE {{ VALUES ?n {{ {values(chunk)} }} ?n wdt:P31 ?c .
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}""",
        ):
            classes[qid(r["n"])].append(qid(r["c"]))
            class_labels[qid(r["c"])] = r.get("cLabel", "")
    class_counts = Counter(c for cs in classes.values() for c in set(cs))
    summary["candidate_classes"] = [
        {"qid": c, "label": class_labels[c], "candidates": k} for c, k in class_counts.most_common()
    ]
    summary["candidates"] = {q: {"label": attrs[q]["label"], "classes": classes.get(q, [])} for q in candidates}

    rng = random.Random(RANDOM_SEED)
    drift = [
        {"qid": q, "label": attrs[q]["label"], "scope": scope_of(attrs[q])}
        for q in rng.sample(candidates, min(DRIFT_SAMPLE, len(candidates)))
    ]

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "date": date.today().isoformat(),
        "commit": commit,
        "seeds": len(seeds),
        "relations": RELATIONS,
        "edges": len(edges),
        "edges_between_seeds": sum(1 for _, _, n, _ in edges if n in seeds),
        "requests": {"fetched": wm.fetched, "reused": wm.reused},
        **summary,
        "drift_sample": drift,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "neighbours_v1.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")

    a = summary["all"]
    lines = [
        "# Neighbour probe, v1",
        "",
        f"Generated by `src/corpus/neighbour_probe.py` on {report['date']} at commit `{commit}`."
        f" Count only: entities one hop from the {len(seeds)} selected seeds along {len(RELATIONS)}"
        " core relations, in both directions. Seeds are not counted as neighbours.",
        "",
        f"- Edges found: {report['edges']} ({report['edges_between_seeds']} between two seeds).",
        f"- Neighbours: {a['entities']}; scope {a['scope']}; humans {a['humans']}.",
        f"- In scope (country India, or no country) with an article: any language {a['in_scope_with_any_article']};"
        f" en {a['in_scope_with_article']['en']}, hi {a['in_scope_with_article']['hi']},"
        f" kn {a['in_scope_with_article']['kn']}.",
        f"- Country India with an article: {a['india_with_any_article']}.",
        "",
        "| Relation | Neighbours | India | No country | Other country | In scope with an article (any / en / hi / kn) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, s in summary["by_relation"].items():
        w = s["in_scope_with_article"]
        lines.append(
            f"| {key} {s['label']} | {s['entities']} | {s['scope'].get('india', 0)} | {s['scope'].get('no_country', 0)}"
            f" | {s['scope'].get('other_country', 0)} | {s['in_scope_with_any_article']} / {w['en']} / {w['hi']} / {w['kn']} |"
        )
    lines += [
        "",
        "A neighbour reached by several relations is counted once in the totals and once per relation in the table.",
        "",
        f"## Most-linked neighbours (hub candidates, top {TOP_HUBS})",
        "",
        "| Neighbour | Seeds linked |",
        "| --- | --- |",
        *[f"| {h['label']} ({h['qid']}) | {h['seeds_linked']} |" for h in summary["hubs"]],
        "",
        f"## Scope-drift sample ({len(drift)} in-scope neighbours with an article, fixed seed {RANDOM_SEED})",
        "",
        *[f"- {d['label']} ({d['qid']}, {d['scope']})" for d in drift],
        "",
    ]
    (OUT_DIR / "neighbours_v1.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
