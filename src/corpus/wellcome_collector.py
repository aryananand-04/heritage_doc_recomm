"""Wellcome Collection collector: manuscripts and pictures that name a seed (decisions 0013, 0015).

For each non-ASI seed, the catalogue is searched twice, once for manuscripts and
archives (work types b and h) and once for pictures (k), first 20 results each; up
to the cap per type are kept, in rank order, when the work names the seed and points
to India (src/corpus/matching.py, or an Indian language).

The text is Wellcome's own catalogue description of the work: title, physical
description, notes (copyright and access notes left out) and subjects. Wellcome
states that data it creates is CC0; the digitised items themselves often carry other
licences (frequently CC BY-NC), and they are neither stored nor redistributed.

Usage:
    uv run python -m src.corpus.wellcome_collector --sample 20
    uv run python -m src.corpus.wellcome_collector

Outputs:
    data/corpus_v1/staging/wellcome_seeds_v1.jsonl.gz    validated records (not committed)
    results/collection/wellcome_seeds_v1.json             counts by work type and outcome
"""

import argparse
import csv
import gzip
import json
import random
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime

from src.corpus import matching
from src.corpus.collection_probe import PROBE_STRATA, WELLCOME, term
from src.corpus.records import Record, sha256
from src.corpus.wikimedia import ROOT, Wikimedia

RAW_DIR = ROOT / "data/raw/corpus_v1/wellcome"
SEEDS = ROOT / "data/seeds/seeds_v1.csv"
STAGING_DIR = ROOT / "data/corpus_v1/staging"
REPORT_DIR = ROOT / "results/collection"
LOOK_AT = 20
LICENCE = "https://creativecommons.org/publicdomain/zero/1.0/"  # Wellcome-created catalogue data
RANDOM_SEED = 20261119
INCLUDE = "subjects,notes,production,languages,genres,contributors"

# Work types searched, with the cap per seed and the document type each gives.
KINDS = {
    "manuscripts": ("b,h", 5, "manuscript_text"),
    "pictures": ("k", 3, "object"),
}
# Catalogue language codes of Indian languages; a work in one of them points to India.
INDIC = {"san", "hin", "tam", "kan", "tel", "ben", "mar", "guj", "pan", "ori", "mal", "pra", "pli", "urd", "asm", "new"}
SKIP_NOTES = {"copyright-note", "terms-of-use", "access-note", "reproduction-note", "location-of-original"}


def subjects(work: dict) -> list[str]:
    return [s.get("label", "") for s in work.get("subjects") or [] if s.get("label")]


def places(work: dict) -> list[str]:
    return [p.get("label", "") for e in work.get("production") or [] for p in e.get("places") or []]


def notes(work: dict) -> list[str]:
    return [
        matching.plain(c)
        for n in work.get("notes") or []
        if (n.get("noteType") or {}).get("id") not in SKIP_NOTES
        for c in n.get("contents") or []
    ]


def languages(work: dict) -> set[str]:
    return {lang.get("id", "") for lang in work.get("languages") or []}


def names_seed(work: dict, search: str) -> bool:
    """All words of the name in the title, subjects or places, or the exact name in the
    notes; and India named in those fields, or the work in an Indian language."""
    short = " ".join([matching.plain(work.get("title")), *subjects(work), *places(work)])
    long = " ".join(notes(work))
    named = matching.names_seed(short, search) or matching.has_word(long, search.lower())
    india = matching.names_india(f"{short} {long}") or bool(languages(work) & INDIC)
    return named and india


def work_text(work: dict) -> str:
    parts = [matching.plain(work.get("title")), matching.plain(work.get("physicalDescription")), *notes(work)]
    if subjects(work):
        parts.append("Subjects: " + "; ".join(subjects(work)))
    if places(work):
        parts.append("Places: " + "; ".join(places(work)))
    return "\n".join(p for p in dict.fromkeys(parts) if p)


def years(work: dict) -> tuple[int | None, int | None]:
    found = [
        int(y)
        for e in work.get("production") or []
        for d in e.get("dates") or []
        for y in re.findall(r"\b(1[0-9]{3}|[5-9][0-9]{2})\b", d.get("label", ""))
    ]
    return (min(found), max(found)) if found else (None, None)


def to_record(work: dict, seeds: list[str], doc_type: str, raw_ref: str, fetched_at: datetime) -> Record:
    text = work_text(work)
    start, end = years(work)
    return Record(
        id=f"wellcome:{work['id']}",
        title=matching.plain(work.get("title")) or f"Wellcome work {work['id']}",
        text=text,
        language="en",  # the catalogue description is in English, whatever the work's own language
        doc_type=doc_type,
        source="wellcome",
        url=f"https://wellcomecollection.org/works/{work['id']}",
        licence=LICENCE,
        entity_ids=seeds,
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

    found: dict[str, dict] = {}
    outcomes = Counter()
    for s in seeds:
        search = term(s["label"], s["stratum"])
        for kind, (types, cap, doc_type) in KINDS.items():
            name = f"wellcome_{kind}_{s['qid']}"
            params = {"query": f'"{search}"', "workType": types, "pageSize": LOOK_AT, "include": INCLUDE}
            data = wm.get_json(name, WELLCOME, params)
            path = wm.raw_path(name, {"url": WELLCOME, **params})
            kept = 0
            for work in data.get("results") or []:
                if kept >= cap:
                    break
                if not names_seed(work, search):
                    outcomes[f"{kind}: does not name the seed or India"] += 1
                    continue
                if s["stratum"] == "dynasty" and not matching.names_dynasty(work_text(work), search, s["label"]):
                    outcomes[f"{kind}: dynasty name without a dynasty word"] += 1
                    continue
                outcomes[f"{kind}: kept"] += 1
                kept += 1
                entry = found.setdefault(
                    work["id"],
                    {
                        "work": work,
                        "doc_type": doc_type,
                        "seeds": [],
                        "raw_ref": f"{path.relative_to(ROOT)}#id={work['id']}",
                        "fetched_at": datetime.fromtimestamp(path.stat().st_mtime, UTC),
                    },
                )
                entry["seeds"].append(s["qid"])

    records = [to_record(e["work"], e["seeds"], e["doc_type"], e["raw_ref"], e["fetched_at"]) for e in found.values()]
    stem = "wellcome_seeds_v1" + ("_sample" if args.sample else "")
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
        "caps": {k: v[1] for k, v in KINDS.items()},
        "looked_at_per_seed_and_type": LOOK_AT,
        "outcomes": dict(sorted(outcomes.items())),
        "records": len(records),
        "records_by_doc_type": dict(Counter(r.doc_type for r in records)),
        "records_linked_to_several_seeds": sum(1 for e in found.values() if len(e["seeds"]) > 1),
        "requests": {"fetched": wm.fetched, "reused": wm.reused},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
