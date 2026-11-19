"""Met collector: public-domain objects that name a seed (decisions 0013, 0015).

For each non-ASI seed, the Met search (the same request the collection probe made,
so its cached answer is reused) gives object IDs; the first 20 objects are fetched,
and up to the cap are kept, in rank order, when the object is public domain and its
own descriptive fields name the seed. An object found for several seeds becomes one
record linked to all of them.

Met records have no descriptive prose. A record's text is its own fields (title,
object type, culture, period, dynasty, medium, date, tags) written as plain lines,
nothing added. Tags that carry a Wikidata ID become entity links.

Usage:
    uv run python -m src.corpus.met_collector --sample 20
    uv run python -m src.corpus.met_collector

Outputs:
    data/corpus_v1/staging/met_seeds_v1.jsonl.gz    validated records (not committed)
    results/collection/met_seeds_v1.json             counts: seeds, objects looked at, kept, skipped by reason
"""

import argparse
import csv
import gzip
import json
import random
import subprocess
from collections import Counter
from datetime import UTC, datetime

import requests

from src.corpus import matching
from src.corpus.collection_probe import MET, MET_OBJECT, PROBE_STRATA, SAMPLE_DEPTH, term
from src.corpus.records import Record, sha256
from src.corpus.wikimedia import ROOT, Wikimedia

RAW_DIR = ROOT / "data/raw/collections_probe_v1"  # shared with the probe, so its cached answers are reused
SEEDS = ROOT / "data/seeds/seeds_v1.csv"
STAGING_DIR = ROOT / "data/corpus_v1/staging"
REPORT_DIR = ROOT / "results/collection"
CAP = 10
LOOK_AT = 20
LICENCE = "https://creativecommons.org/publicdomain/zero/1.0/"
RANDOM_SEED = 20261119

# Fields written into the text, in this order, with the label each line gets.
TEXT_FIELDS = [
    ("objectName", "Object"),
    ("culture", "Culture"),
    ("period", "Period"),
    ("dynasty", "Dynasty"),
    ("reign", "Reign"),
    ("objectDate", "Date"),
    ("artistDisplayName", "Artist"),
    ("medium", "Medium"),
    ("classification", "Classification"),
    ("country", "Country"),
    ("region", "Region"),
    ("city", "City"),
]
# Fields checked for the seed's name (the descriptive ones, not medium or credit line).
NAME_FIELDS = ("title", "culture", "period", "dynasty", "reign", "objectName", "country", "region", "city")


def tag_qids(obj: dict) -> list[str]:
    out = []
    for t in obj.get("tags") or []:
        url = t.get("Wikidata_URL") or ""
        q = url.rsplit("/", 1)[-1]
        if q.startswith("Q") and q[1:].isdigit():
            out.append(q)
    return out


def object_text(obj: dict) -> str:
    lines = [obj.get("title", "").strip()]
    for key, label in TEXT_FIELDS:
        value = str(obj.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    tags = [t.get("term", "") for t in obj.get("tags") or [] if t.get("term")]
    if tags:
        lines.append("Tags: " + ", ".join(tags))
    return "\n".join(line for line in lines if line)


INDIA_FIELDS = ("culture", "country", "region")


def names_seed(obj: dict, search: str) -> bool:
    """The shared matching rule (src/corpus/matching.py) on the object's descriptive
    fields and tags, with India required in its culture, country or region."""
    fields = " ".join(str(obj.get(k) or "") for k in NAME_FIELDS)
    tags = " ".join(t.get("term", "") for t in obj.get("tags") or [])
    places = " ".join(str(obj.get(k) or "") for k in INDIA_FIELDS)
    return matching.names_seed(f"{fields} {tags}", search) and matching.names_india(places)


def year(value) -> int | None:
    return int(value) if isinstance(value, int) else None


def to_record(obj: dict, seeds: list[str], raw_ref: str, fetched_at: datetime) -> Record:
    text = object_text(obj)
    entity_ids = list(dict.fromkeys(seeds + tag_qids(obj)))
    start, end = year(obj.get("objectBeginDate")), year(obj.get("objectEndDate"))
    if start is not None and end is not None and start > end:
        start, end = end, start
    return Record(
        id=f"met:{obj['objectID']}",
        title=obj.get("title") or obj.get("objectName") or f"Met object {obj['objectID']}",
        text=text,
        language="en",
        doc_type="object",
        source="met",
        url=obj["objectURL"],
        licence=LICENCE,
        entity_ids=entity_ids,
        period_start=start,
        period_end=end,
        fetched_at=fetched_at,
        sha256=sha256(text),
        raw_ref=raw_ref,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=int, help="only a fixed random sample of this many seeds")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(SEEDS, encoding="utf-8")) if r["status"] == "selected"]
    seeds = sorted((r for r in rows if r["stratum"] in PROBE_STRATA), key=lambda r: r["qid"])
    if args.sample:
        seeds = sorted(random.Random(RANDOM_SEED).sample(seeds, args.sample), key=lambda r: r["qid"])
    wm = Wikimedia(RAW_DIR)

    found: dict[int, dict] = {}  # objectID -> {"obj", "seeds", "raw_ref", "fetched_at"}
    outcomes, kept_per_seed = Counter(), Counter()
    for s in seeds:
        search = term(s["label"], s["stratum"])
        # The probe's exact request, so the cached answer is reused.
        data = wm.get_json(f"met_{s['qid']}", MET, {"q": search, "isPublicDomain": "true", "limit": SAMPLE_DEPTH})
        for oid in (data.get("objectIDs") or [])[:LOOK_AT]:
            if kept_per_seed[s["qid"]] >= CAP:
                break
            name = f"met_object_{oid}"
            try:
                obj = wm.get_json(name, MET_OBJECT.format(oid), {})
            except requests.HTTPError:
                outcomes["withdrawn (404)"] += 1
                continue
            if not obj.get("isPublicDomain"):
                outcomes["not public domain"] += 1
            elif not names_seed(obj, search):
                outcomes["does not name the seed"] += 1
            elif s["stratum"] == "dynasty" and not matching.names_dynasty(object_text(obj), search, s["label"]):
                outcomes["dynasty name without a dynasty word"] += 1
            else:
                outcomes["kept"] += 1
                kept_per_seed[s["qid"]] += 1
                path = wm.raw_path(name, {"url": MET_OBJECT.format(oid)})
                entry = found.setdefault(
                    oid,
                    {
                        "obj": obj,
                        "seeds": [],
                        "raw_ref": str(path.relative_to(ROOT)),
                        "fetched_at": datetime.fromtimestamp(path.stat().st_mtime, UTC),
                    },
                )
                entry["seeds"].append(s["qid"])

    records = [to_record(e["obj"], e["seeds"], e["raw_ref"], e["fetched_at"]) for e in found.values()]
    stem = "met_seeds_v1" + ("_sample" if args.sample else "")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(STAGING_DIR / f"{stem}.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "date": datetime.now(UTC).date().isoformat(),
        "commit": commit,
        "seeds": len(seeds),
        "sample": args.sample,
        "cap": CAP,
        "looked_at_per_seed": LOOK_AT,
        "outcomes": dict(outcomes.most_common()),
        "seeds_with_records": sum(1 for n in kept_per_seed.values() if n),
        "records": len(records),
        "records_linked_to_several_seeds": sum(1 for e in found.values() if len(e["seeds"]) > 1),
        "requests": {"fetched": wm.fetched, "reused": wm.reused},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
