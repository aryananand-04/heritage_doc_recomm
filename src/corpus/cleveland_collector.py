"""Cleveland Museum of Art collector: CC0 objects that name a seed (decisions 0013, 0015).

For each non-ASI seed, the open-access search (CC0 records only) returns the first
20 results; up to the cap are kept, in rank order, when the record is CC0, its
descriptive fields name the seed and its culture names India (src/corpus/matching.py).
An object found for several seeds becomes one record linked to all of them.

The text is the record's own title, summary line (tombstone), description and
"did you know" note, with markup removed and nothing added. The object's own
Wikidata ID, when the museum gives one, becomes an entity link.

Usage:
    uv run python -m src.corpus.cleveland_collector --sample 20
    uv run python -m src.corpus.cleveland_collector

Outputs:
    data/corpus_v1/staging/cleveland_seeds_v1.jsonl.gz    validated records (not committed)
    results/collection/cleveland_seeds_v1.json             counts: seeds, records looked at, kept, skipped by reason
"""

import argparse
import csv
import gzip
import json
import random
import subprocess
from collections import Counter
from datetime import UTC, datetime

from src.corpus import matching
from src.corpus.collection_probe import CLEVELAND, PROBE_STRATA, term
from src.corpus.records import Record, sha256
from src.corpus.wikimedia import ROOT, Wikimedia

RAW_DIR = ROOT / "data/raw/corpus_v1/cleveland"
SEEDS = ROOT / "data/seeds/seeds_v1.csv"
STAGING_DIR = ROOT / "data/corpus_v1/staging"
REPORT_DIR = ROOT / "results/collection"
CAP = 5
LOOK_AT = 20
LICENCE = "https://creativecommons.org/publicdomain/zero/1.0/"
RANDOM_SEED = 20261119


def cultures(obj: dict) -> str:
    c = obj.get("culture") or []
    return "; ".join(c) if isinstance(c, list) else str(c)


def names_seed(obj: dict, search: str) -> bool:
    """All words of the seed's name in the short identifying fields, or the exact name
    in the description. Long descriptions contain common words such as "sun" and
    "temple" by chance, so words scattered through them are not enough."""
    short = " ".join(matching.plain(obj.get(k)) for k in ("title", "tombstone", "type")) + " " + cultures(obj)
    in_short = matching.names_seed(short, search)
    in_description = matching.has_word(matching.plain(obj.get("description")), search.lower())
    return (in_short or in_description) and matching.names_india(cultures(obj))


def object_text(obj: dict) -> str:
    parts = [matching.plain(obj.get(k)) for k in ("title", "tombstone", "description", "did_you_know")]
    return "\n".join(p for p in dict.fromkeys(parts) if p)


def own_qids(obj: dict) -> list[str]:
    urls = (obj.get("external_resources") or {}).get("wikidata") or []
    qids = [u.rstrip("/").rsplit("/", 1)[-1] for u in urls]
    return [q for q in qids if q.startswith("Q") and q[1:].isdigit()]


def year(value) -> int | None:
    return int(value) if isinstance(value, int) else None


def to_record(obj: dict, seeds: list[str], raw_ref: str, fetched_at: datetime) -> Record:
    text = object_text(obj)
    start, end = year(obj.get("creation_date_earliest")), year(obj.get("creation_date_latest"))
    if start is not None and end is not None and start > end:
        start, end = end, start
    return Record(
        id=f"cleveland:{obj['id']}",
        title=matching.plain(obj.get("title")) or f"Cleveland object {obj['accession_number']}",
        text=text,
        language="en",
        doc_type="object",
        source="cleveland",
        url=obj["url"],
        licence=LICENCE,
        entity_ids=list(dict.fromkeys(seeds + own_qids(obj))),
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

    found: dict[int, dict] = {}
    outcomes, kept_per_seed = Counter(), Counter()
    for s in seeds:
        search = term(s["label"], s["stratum"])
        name = f"cleveland_{s['qid']}"
        params = {"q": search, "cc0": "1", "limit": LOOK_AT}
        data = wm.get_json(name, CLEVELAND, params)
        path = wm.raw_path(name, {"url": CLEVELAND, **params})
        for obj in data.get("data") or []:
            if kept_per_seed[s["qid"]] >= CAP:
                break
            if obj.get("share_license_status") != "CC0":
                outcomes["not CC0"] += 1
            elif not names_seed(obj, search):
                outcomes["does not name the seed or India"] += 1
            elif s["stratum"] == "dynasty" and not matching.names_dynasty(object_text(obj), search, s["label"]):
                outcomes["dynasty name without a dynasty word"] += 1
            elif not object_text(obj):
                outcomes["no text"] += 1
            else:
                outcomes["kept"] += 1
                kept_per_seed[s["qid"]] += 1
                entry = found.setdefault(
                    obj["id"],
                    {
                        "obj": obj,
                        "seeds": [],
                        "raw_ref": f"{path.relative_to(ROOT)}#id={obj['id']}",
                        "fetched_at": datetime.fromtimestamp(path.stat().st_mtime, UTC),
                    },
                )
                entry["seeds"].append(s["qid"])

    records = [to_record(e["obj"], e["seeds"], e["raw_ref"], e["fetched_at"]) for e in found.values()]
    stem = "cleveland_seeds_v1" + ("_sample" if args.sample else "")
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
