"""Which one-hop neighbours become document entities (decision 0015).

The neighbour probe (decision 0005) lists every in-scope neighbour with a Wikipedia
article. Its scope-drift sample showed districts, cities, universities and modern
musicians among them, so a neighbour becomes a document entity only when its
Wikidata class (P31, followed up P279) falls under a heritage root class below,
none of its direct classes is in the excluded list, and, for a person, the seed
date rule holds (born, died or active before 1900). Places such as districts and
cities stay graph nodes.

The root and excluded classes were chosen from the classes the probe found
(`results/probes/neighbours_v1.json`, `candidate_classes`).

Usage:
    uv run python -m src.corpus.neighbour_select

Outputs:
    data/seeds/neighbours_v1.csv          entity list in the seed-list format, read by the collectors
    results/probes/neighbour_select_v1.md  counts by document type and root class; drift sample
"""

import csv
import json
import random
from collections import Counter, defaultdict

from src.corpus.seeds import (
    PEOPLE_BORN_BEFORE,
    PEOPLE_SCAN_FILE,
    STYLE_SCAN_FILE,
    batches,
    born_before,
    date_basis,
    life_years,
    scan_removed,
    year_of,
)
from src.corpus.wikimedia import ROOT, Wikimedia, qid, values

RAW_DIR = ROOT / "data/raw/neighbours_probe_v1"
PROBE = ROOT / "results/probes/neighbours_v1.json"
OUT_CSV = ROOT / "data/seeds/neighbours_v1.csv"
ASI_REMAINDER = ROOT / "data/seeds/asi_remainder_v1.csv"  # run src.corpus.asi_remainder first
OUT_MD = ROOT / "results/probes/neighbour_select_v1.md"
RANDOM_SEED = 20261119
DRIFT_SAMPLE = 50
HUMAN = "Q5"

# Heritage root classes, in priority order: the first root a neighbour falls under
# decides its document type.
ROOTS = {
    "Q1640824": ("inscription", "inscription"),
    "Q87167": ("manuscript", "manuscript_text"),
    "Q384515": ("treatise", "manuscript_text"),
    "Q7725634": ("literary work", "manuscript_text"),  # only with a date before 1900, see DATED_ROOTS
    "Q3305213": ("painting", "object"),
    "Q860861": ("sculpture", "object"),
    "Q220659": ("archaeological artefact", "object"),
    "Q839954": ("archaeological site", "site"),
    "Q811979": ("built structure", "site"),
    "Q35509": ("cave", "site"),
    "Q1107656": ("garden", "site"),
    "Q15661340": ("ancient city", "site"),
    "Q420962": ("heritage railway", "site"),
    "Q216926": ("raga", "culture"),
    "Q2379243": ("thaat", "culture"),
    "Q132241": ("festival", "culture"),
    "Q868557": ("music festival", "culture"),
    "Q188451": ("music genre", "culture"),
    "Q107357104": ("type of dance", "culture"),
    "Q164950": ("dynasty", "person_dynasty_style"),
    "Q3024240": ("historical country", "person_dynasty_style"),
    "Q1336152": ("princely state", "person_dynasty_style"),
    "Q32880": ("architectural style", "person_dynasty_style"),
}

# Direct classes that rule a neighbour out even when it falls under a root (modern
# infrastructure under "built structure", and educational and research institutions).
EXCLUDED = {
    "Q55488": "railway station",
    "Q22808404": "station located on surface",
    "Q27686": "hotel",
    "Q16917": "hospital",
    "Q682943": "cricket field",
    "Q3918": "university",
    "Q189004": "college",
    "Q3914": "school",
    "Q7075": "library",
    "Q31855": "research institute",
    "Q4671277": "academic institution",
    "Q1663017": "engineering school",
    "Q131389368": "state private university",
    "Q7603893": "state public university",
    "Q3520135": "deemed university",
    "Q3351682": "central university",
    "Q1248784": "airport",
    "Q220505": "film festival",
    "Q23902005": "literary festival",
    "Q11424": "film",
    "Q13406463": "Wikimedia list article",
    "Q383092": "art academy",
    "Q7540126": "headquarters",
}

# Roots that also include modern works; a neighbour counts under them only with an
# inception or publication date before the people date limit.
DATED_ROOTS = {"Q7725634"}


def classify(
    classes: list[str], roots_found: set[str], years: dict | None, work_year: int | None = None
) -> tuple[str | None, str]:
    """The document type for one neighbour and the reason, or None and why it is left out."""
    if HUMAN in classes:
        if years and born_before(years, PEOPLE_BORN_BEFORE):
            return "person_dynasty_style", f"person, {date_basis(years)}"
        return None, "person after the date limit or undated"
    excluded = [EXCLUDED[c] for c in classes if c in EXCLUDED]
    if excluded:
        return None, f"excluded class: {excluded[0]}"
    for root, (label, doc_type) in ROOTS.items():
        if root in roots_found:
            if root in DATED_ROOTS and (work_year is None or work_year >= PEOPLE_BORN_BEFORE):
                continue
            return doc_type, label
    return None, "no heritage class (place, organisation, work or other)"


def main() -> None:
    probe = json.load(open(PROBE, encoding="utf-8"))
    # Hand-scan removals of the seed list apply to neighbours too (author decisions).
    removed = scan_removed(PEOPLE_SCAN_FILE) | scan_removed(STYLE_SCAN_FILE)
    # ASI monuments among the neighbours are already in the ASI remainder list.
    asi = {r["qid"] for r in csv.DictReader(open(ASI_REMAINDER, encoding="utf-8"))}
    candidates = {q: c for q, c in probe["candidates"].items() if q not in removed and q not in asi}
    wm = Wikimedia(RAW_DIR)

    roots: dict[str, set] = defaultdict(set)
    root_values = " ".join(f"wd:{r}" for r in ROOTS)
    for n, chunk in batches(sorted(candidates), 100):
        for r in wm.sparql(
            f"select_roots_{n:03d}",
            f"""SELECT DISTINCT ?n ?root WHERE {{ VALUES ?n {{ {values(chunk)} }} VALUES ?root {{ {root_values} }}
                ?n wdt:P31/wdt:P279* ?root . }}""",
        ):
            roots[qid(r["n"])].add(qid(r["root"]))

    people = [q for q, c in candidates.items() if HUMAN in c["classes"]]
    years = life_years(wm, people)

    dated = sorted(q for q, rs in roots.items() if rs & DATED_ROOTS)
    work_years: dict[str, int] = {}
    for n, chunk in batches(dated, 100):
        for r in wm.sparql(
            f"select_work_dates_{n:03d}",
            f"""SELECT ?n ?d WHERE {{ VALUES ?n {{ {values(chunk)} }}
                {{ ?n wdt:P571 ?d }} UNION {{ ?n wdt:P577 ?d }} }}""",
        ):
            y = year_of(r.get("d"))
            if y is not None:
                work_years[qid(r["n"])] = min(y, work_years.get(qid(r["n"]), y))

    kept, reasons = [], Counter()
    for q in sorted(candidates):
        doc_type, reason = classify(candidates[q]["classes"], roots.get(q, set()), years.get(q), work_years.get(q))
        reasons[reason if doc_type is None else "kept: " + reason.split(",")[0]] += 1
        if doc_type:
            kept.append({"qid": q, "label": candidates[q]["label"], "doc_type": doc_type, "basis": reason})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "label", "stratum", "sub_stratum", "doc_type", "status", "flags"])
        for k in kept:
            w.writerow(
                [k["qid"], k["label"], "neighbour", k["basis"], k["doc_type"], "selected", "one hop from a seed"]
            )

    drift = random.Random(RANDOM_SEED).sample(kept, min(DRIFT_SAMPLE, len(kept)))
    by_type = Counter(k["doc_type"] for k in kept)
    lines = [
        "# Neighbour selection, v1",
        "",
        f"Generated by `src/corpus/neighbour_select.py` from `results/probes/neighbours_v1.json`"
        f" (probe of {probe['date']}, commit `{probe['commit']}`). {len(candidates)} in-scope neighbours with an"
        f" article were checked ({len(probe['candidates']) - len(candidates)} more were removed by the seed hand scans"
        " or are already in the ASI remainder list);"
        f" {len(kept)} become document entities.",
        "",
        "| Document type | Entities |",
        "| --- | --- |",
        *[f"| {t} | {n} |" for t, n in by_type.most_common()],
        "",
        "| Outcome | Neighbours |",
        "| --- | --- |",
        *[f"| {r} | {n} |" for r, n in reasons.most_common()],
        "",
        f"## Drift sample ({len(drift)} kept entities, fixed seed {RANDOM_SEED}), for the author's hand check",
        "",
        *[f"- {d['label']} ({d['qid']}, {d['doc_type']}, {d['basis']})" for d in drift],
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
