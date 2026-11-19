"""Met collector by place: every public-domain object whose place is India (decision 0017).

Object IDs come from two searches, both paged 500 at a time: geoLocation India, and
the Asian Art department searched for "India". Each object is fetched (the same
cached request the seed-name collector uses) and kept when the object itself says
it is public domain (the search ignores that filter) and its culture, country or
region names India. Text and record follow `met_collector.to_record`.

Entity links at collection are layer 1 only (decision 0017): the Wikidata IDs the
Met gives on tags and artists. Layers 2 and 3 are added by the linking step. An
object with no layer 1 link is linked to India (Q668) for now.

Usage:
    uv run python -m src.corpus.met_place_collector --sample 50
    uv run python -m src.corpus.met_place_collector

Outputs:
    data/corpus_v1/staging/met_place_v1.jsonl.gz
    results/collection/met_place_v1.json
"""

import argparse
import gzip
import json
import random
import subprocess
from collections import Counter
from datetime import UTC, datetime

import requests

from src.corpus import matching
from src.corpus.collection_probe import MET, MET_OBJECT
from src.corpus.met_collector import INDIA_FIELDS, RAW_DIR, object_text, tag_qids, to_record
from src.corpus.wikimedia import ROOT, Wikimedia

STAGING_DIR = ROOT / "data/corpus_v1/staging"
REPORT_DIR = ROOT / "results/collection"
INDIA = "Q668"
PAGE = 500
RANDOM_SEED = 20261119
SEARCHES = {
    "geoLocation India": {"q": "India", "geoLocation": "India"},
    "Asian Art department": {"q": "India", "departmentId": 6},
}


def artist_qids(obj: dict) -> list[str]:
    url = obj.get("artistWikidata_URL") or ""
    q = url.rstrip("/").rsplit("/", 1)[-1]
    return [q] if q.startswith("Q") and q[1:].isdigit() else []


def object_ids(wm: Wikimedia) -> tuple[list[int], dict]:
    ids, totals = [], {}
    for label, params in SEARCHES.items():
        offset, total = 0, None
        while total is None or offset < total:
            page = {**params, "limit": PAGE, "offset": offset}
            data = wm.get_json(f"met_place_{label[:12].replace(' ', '_')}_{offset}", MET, page)
            total = data.get("total") or 0
            got = data.get("objectIDs") or []
            ids.extend(got)
            if not got:
                break
            offset += PAGE
        totals[label] = total
    return sorted(set(ids)), totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=int, help="only a fixed random sample of this many objects")
    args = ap.parse_args()

    wm = Wikimedia(RAW_DIR)
    ids, totals = object_ids(wm)
    distinct = len(ids)
    if args.sample:
        ids = sorted(random.Random(RANDOM_SEED).sample(ids, args.sample))

    records, outcomes, layer1 = [], Counter(), 0
    for oid in ids:
        name = f"met_object_{oid}"
        try:
            obj = wm.get_json(name, MET_OBJECT.format(oid), {})
        except requests.HTTPError:
            outcomes["withdrawn (404)"] += 1
            continue
        places = " ".join(str(obj.get(k) or "") for k in INDIA_FIELDS)
        if not obj.get("isPublicDomain"):
            outcomes["not public domain"] += 1
        elif not matching.names_india(places):
            outcomes["place does not name India"] += 1
        elif not object_text(obj):
            outcomes["no text"] += 1
        else:
            links = list(dict.fromkeys(tag_qids(obj) + artist_qids(obj)))
            layer1 += bool(links)
            path = wm.raw_path(name, {"url": MET_OBJECT.format(oid)})
            fetched_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            rec = to_record(obj, links or [INDIA], str(path.relative_to(ROOT)), fetched_at)
            records.append(rec)
            outcomes["kept"] += 1

    stem = "met_place_v1" + ("_sample" if args.sample else "")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(STAGING_DIR / f"{stem}.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "date": datetime.now(UTC).date().isoformat(),
        "commit": commit,
        "search_totals": totals,
        "distinct_object_ids": distinct,
        "sample": args.sample,
        "outcomes": dict(outcomes.most_common()),
        "records": len(records),
        "records_with_layer1_links": layer1,
        "requests": {"fetched": wm.fetched, "reused": wm.reused},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
