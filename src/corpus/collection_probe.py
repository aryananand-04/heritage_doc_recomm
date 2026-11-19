"""Count-only probe: how many museum and archive records each source holds for the seeds.

Nothing is downloaded beyond search counts and, for a fixed sample of seeds, the
first page of results used to measure precision. The output lets the author set
per-entity caps before any collection.

Sources (specification, section 3): the Met and Cleveland open access (objects),
Wellcome Collection (manuscripts, archives and pictures) and the Internet Archive
(public-domain texts).

Usage:
    uv run python -m src.corpus.collection_probe

Outputs:
    results/probes/collections_v1.json   per-seed counts, precision sample, caps applied
    results/probes/collections_v1.md     summary
"""

import csv
import json
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import requests

from src.corpus.wikimedia import ROOT, Wikimedia

RAW_DIR = ROOT / "data/raw/collections_probe_v1"
OUT_DIR = ROOT / "results/probes"
SEEDS = ROOT / "data/seeds/seeds_v1.csv"
RANDOM_SEED = 20261118
SAMPLE_SEEDS = 30  # seeds whose first page of results is checked for precision
SAMPLE_DEPTH = 5  # results checked per seed and source

# ASI monuments are left out: museum and archive records are about dynasties, styles,
# crafts, cultural forms, people and major sites, not individual minor monuments.
PROBE_STRATA = ("world_heritage", "tentative_list", "dynasty", "culture", "style", "person", "topup_culture")

# Proposed per-seed caps (for the author to approve), with the document type each source gives.
CAPS = {
    "met": (5, "object"),
    "cleveland": (5, "object"),
    "wellcome_manuscripts": (3, "manuscript_text"),
    "wellcome_pictures": (2, "object"),
    "internet_archive": (2, "manuscript_text"),
}

MET = "https://collectionapi.metmuseum.org/public/collection/v1.1/search"
MET_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
CLEVELAND = "https://openaccess-api.clevelandart.org/api/artworks/"
WELLCOME = "https://api.wellcomecollection.org/catalogue/v2/works"
IA = "https://archive.org/advancedsearch.php"
IA_FREE = "(possible-copyright-status:NOT_IN_COPYRIGHT OR licenseurl:*publicdomain* OR licenseurl:*zero*)"


def term(label: str, stratum: str = "") -> str:
    """Search term for a seed.

    The label loses a trailing parenthesis ('Sua (folk music)' -> 'Sua'). For dynasties
    and historical states the generic word goes too, because collections describe objects
    as 'Chola period' or 'Mughal', not 'Chola dynasty'.
    """
    t = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    if stratum == "dynasty":
        # "Sultanate" stays: without it, "Delhi Sultanate" would match anything about Delhi.
        stripped = re.sub(r"^(the\s+)?kingdom of\s+", "", t, flags=re.I)
        stripped = re.sub(r"\s+(dynasty|empire|kingdom)\b", "", stripped, flags=re.I).strip(" ,")
        t = stripped or t
    return t


def mentions(text: str, search: str) -> bool:
    return re.search(r"\b" + re.escape(search.lower()) + r"\b", (text or "").lower()) is not None


# -------------------------------------------------------------- per source


def probe_met(wm: Wikimedia, seed: dict, sample: bool) -> dict:
    data = wm.get_json(f"met_{seed['qid']}", MET, {"q": seed["term"], "isPublicDomain": "true", "limit": SAMPLE_DEPTH})
    out = {"met": data.get("total") or 0}
    if sample:
        texts = []
        for oid in (data.get("objectIDs") or [])[:SAMPLE_DEPTH]:
            try:
                o = wm.get_json(f"met_object_{oid}", MET_OBJECT.format(oid), {})
            except requests.HTTPError:
                continue  # withdrawn objects answer 404; they are not counted in the sample
            texts.append(" ".join(str(o.get(k, "")) for k in ("title", "culture", "period", "dynasty", "objectName")))
        out["met_sample"] = [mentions(t, seed["term"]) for t in texts]
    return out


def probe_cleveland(wm: Wikimedia, seed: dict, sample: bool) -> dict:
    data = wm.get_json(f"cleveland_{seed['qid']}", CLEVELAND, {"q": seed["term"], "cc0": "1", "limit": SAMPLE_DEPTH})
    out = {"cleveland": data.get("info", {}).get("total") or 0}
    if sample:
        out["cleveland_sample"] = [
            mentions(
                " ".join(str(x) for x in (d.get("title"), d.get("culture"), d.get("type"), d.get("description"))),
                seed["term"],
            )
            for d in (data.get("data") or [])[:SAMPLE_DEPTH]
        ]
    return out


def probe_wellcome(wm: Wikimedia, seed: dict, sample: bool) -> dict:
    data = wm.get_json(
        f"wellcome_{seed['qid']}",
        WELLCOME,
        {"query": f'"{seed["term"]}"', "pageSize": SAMPLE_DEPTH, "aggregations": "workType"},
    )
    buckets = {b["data"]["id"]: b["count"] for b in data.get("aggregations", {}).get("workType", {}).get("buckets", [])}
    out = {
        "wellcome_manuscripts": buckets.get("b", 0) + buckets.get("h", 0),
        "wellcome_pictures": buckets.get("k", 0),
        "wellcome_all": data.get("totalResults") or 0,
    }
    if sample:
        out["wellcome_sample"] = [
            mentions(w.get("title", ""), seed["term"]) for w in data.get("results", [])[:SAMPLE_DEPTH]
        ]
    return out


def probe_ia(wm: Wikimedia, seed: dict, sample: bool) -> dict:
    t = seed["term"].replace('"', "")
    params = {
        "q": f'(title:("{t}") OR subject:("{t}")) AND mediatype:texts AND {IA_FREE}',
        "fl[]": "identifier,title,subject",
        "rows": SAMPLE_DEPTH,
        "output": "json",
    }
    data = wm.get_json(f"ia_{seed['qid']}", IA, params)
    out = {"internet_archive": data.get("response", {}).get("numFound") or 0}
    if sample:
        out["internet_archive_sample"] = [
            mentions(f"{d.get('title', '')} {d.get('subject', '')}", seed["term"])
            for d in data.get("response", {}).get("docs", [])[:SAMPLE_DEPTH]
        ]
    return out


SOURCES = {"met": probe_met, "cleveland": probe_cleveland, "wellcome": probe_wellcome, "internet_archive": probe_ia}


def run_source(name: str, seeds: list[dict], sample_ids: set) -> dict:
    # One client per thread: sources are probed in parallel, each at one request per second.
    wm = Wikimedia(RAW_DIR)
    results = {s["qid"]: SOURCES[name](wm, s, s["qid"] in sample_ids) for s in seeds}
    return {"results": results, "fetched": wm.fetched, "reused": wm.reused}


# -------------------------------------------------------------------- main


def main() -> None:
    rows = [r for r in csv.DictReader(open(SEEDS, encoding="utf-8")) if r["status"] == "selected"]
    probe = [r for r in rows if r["stratum"] in PROBE_STRATA]
    for r in probe:
        r["term"] = term(r["label"], r["stratum"])
    rng = random.Random(RANDOM_SEED)
    sample_ids = set(rng.sample(sorted(r["qid"] for r in probe), SAMPLE_SEEDS))

    with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
        futures = {name: pool.submit(run_source, name, probe, sample_ids) for name in SOURCES}
        by_source = {name: f.result() for name, f in futures.items()}

    per_seed = {
        r["qid"]: {"label": r["label"], "stratum": r["stratum"], "sub_stratum": r["sub_stratum"]} for r in probe
    }
    for res in by_source.values():
        for q, counts in res["results"].items():
            per_seed[q].update(counts)

    summary = {}
    for key, (cap, doc_type) in CAPS.items():
        hits = [v.get(key, 0) for v in per_seed.values()]
        checked = [
            x
            for v in per_seed.values()
            for x in v.get(f"{key.split('_')[0] if key.startswith('wellcome') else key}_sample", [])
        ]
        summary[key] = {
            "doc_type": doc_type,
            "seeds_with_hits": sum(1 for h in hits if h),
            "raw_hits": sum(hits),
            "cap_per_seed": cap,
            "capped_upper_bound": sum(min(h, cap) for h in hits),
            "precision_sample": {"checked": len(checked), "mention_seed": sum(checked)},
        }

    wiki = {lang: sum(1 for r in rows if r[lang] == "yes") for lang in ("en", "hi", "kn")}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "date": date.today().isoformat(),
        "commit": commit,
        "seeds_probed": len(probe),
        "strata_probed": PROBE_STRATA,
        "sample_seeds": sorted(sample_ids),
        "wikipedia_articles_available": wiki,
        "summary": summary,
        "requests": {n: {"fetched": r["fetched"], "reused": r["reused"]} for n, r in by_source.items()},
        "per_seed": per_seed,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "collections_v1.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Collection probe, v1",
        "",
        f"Generated by `src/corpus/collection_probe.py` on {report['date']} at commit `{commit}`."
        f" Search counts only, for {len(probe)} seeds (all strata except ASI monuments)."
        " Raw hits are upper bounds: label searches also match unrelated records.",
        "",
        "| Source | Gives | Seeds with hits | Raw hits | Cap per seed | Capped upper bound | Precision sample (top results mentioning the seed) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, s in summary.items():
        p = s["precision_sample"]
        lines.append(
            f"| {key} | {s['doc_type']} | {s['seeds_with_hits']} | {s['raw_hits']} | {s['cap_per_seed']}"
            f" | {s['capped_upper_bound']} | {p['mention_seed']} of {p['checked']} |"
        )
    lines += [
        "",
        f"Wikipedia articles available for all {len(rows)} seeds: English {wiki['en']}, Hindi {wiki['hi']},"
        f" Kannada {wiki['kn']}.",
        "",
        f"Precision sample: {SAMPLE_SEEDS} seeds drawn with a fixed seed ({RANDOM_SEED}); the first {SAMPLE_DEPTH}"
        " results per source checked for the seed's name in the title or descriptive fields.",
        "",
    ]
    (OUT_DIR / "collections_v1.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
