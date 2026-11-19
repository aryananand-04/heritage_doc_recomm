"""Internet Archive reference series: Imperial Gazetteer of India volumes (decisions 0016, 0017).

The expansion probe counted 118 public-domain items titled "Imperial Gazetteer of
India" dated 1700–1949 (`results/probes/expansion_v1.md`). Each item goes through
the same rules as the seed-name collector (public-domain licence on the item, OCR
text of at least 500 characters, English, Hindi or Kannada by script, dated before
1950). The volumes describe places across British India, so each is linked to the
Indian places its title names (gazetteer, place entries) and to India. Many volumes
were scanned more than once; repeat scans are found by text overlap and one copy,
the longest, is kept.

Usage:
    uv run python -m src.corpus.ia_series_collector

Outputs:
    data/corpus_v1/staging/internet_archive_series_v1.jsonl.gz
    results/collection/internet_archive_series_v1.json
"""

import gzip
import itertools
import json
import re
import statistics
import subprocess
from collections import Counter
from datetime import UTC, datetime
from urllib.parse import quote

from src.corpus import internet_archive_collector as ia
from src.corpus import matching
from src.corpus.link_objects import Gazetteer
from src.corpus.records import Record, sha256
from src.corpus.wikimedia import ROOT, Wikimedia

STAGING = ROOT / "data/corpus_v1/staging/internet_archive_series_v1.jsonl.gz"
REPORT = ROOT / "results/collection/internet_archive_series_v1.json"
INDIA = "Q668"
SERIES = 'title:("Imperial Gazetteer of India")'
ROWS = 500
# Repeat scans of one volume: two texts whose 5-word shingles in a fixed window overlap by
# more than half (of the smaller set) are copies; one copy, the longest, is kept.
WINDOW = (200_000, 260_000)
DUPLICATE_OVERLAP = 0.5


def shingles(text: str) -> set[str]:
    words = re.findall(r"[a-z]{3,}", text[WINDOW[0] : WINDOW[1]].lower())
    return {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}


def drop_repeat_scans(records: list[Record]) -> tuple[list[Record], list[list[str]]]:
    """One record per group of repeat scans (the longest text); the groups that lost copies."""
    sets = [shingles(r.text) for r in records]
    parent = list(range(len(records)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(len(records)), 2):
        a, b = sets[i], sets[j]
        if a and b and len(a & b) / min(len(a), len(b)) > DUPLICATE_OVERLAP:
            parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(root(i), []).append(i)
    kept, dropped = [], []
    for members in groups.values():
        members.sort(key=lambda i: (-len(records[i].text), records[i].id))
        kept.append(records[members[0]])
        if len(members) > 1:
            dropped.append([records[i].id for i in members])
    return sorted(kept, key=lambda r: r.id), dropped


def main() -> None:
    wm = Wikimedia(ia.RAW_DIR)
    gaz = Gazetteer()
    params = {
        "q": f"{SERIES} AND mediatype:texts AND {ia.IA_FREE} AND date:[1700-01-01 TO 1949-12-31]",
        "fl[]": ia.FIELDS,
        "rows": ROWS,
        "output": "json",
    }
    docs = wm.get_json("ia_series_imperial_gazetteer", ia.IA, params).get("response", {}).get("docs", [])

    records, outcomes = [], Counter()
    for doc in docs:
        ident = doc["identifier"]
        y = ia.item_year(doc)
        if y is None or y >= ia.DATED_BEFORE:
            outcomes["undated or dated 1950 or later"] += 1
            continue
        meta = wm.get_json(f"ia_meta_{ident}", f"https://archive.org/metadata/{ident}", {})
        lic, name = ia.licence(meta.get("metadata") or {}), ia.ocr_file(meta)
        if lic is None:
            outcomes["no public-domain licence on the item"] += 1
            continue
        if name is None:
            outcomes["no OCR text"] += 1
            continue
        url = f"https://archive.org/download/{ident}/{quote(name)}"
        text = ia.clean(wm.get_text(f"ia_text_{ident}", url))
        if len(text) < ia.MIN_CHARS:
            outcomes["OCR text too short"] += 1
            continue
        lang = ia.language(doc, text)
        if lang is None:
            outcomes["text not in English, Hindi or Kannada"] += 1
            continue
        title = matching.plain(" ".join(ia.as_list(doc.get("title")))) or ident
        places = [c["qid"] for _, cands in gaz.find(title, {"place", "polity"}) if len(cands) == 1 for c in cands]
        path = wm.raw_path(f"ia_text_{ident}", {"url": url}, ".txt.gz")
        records.append(
            Record(
                id=f"internet_archive:{ident}",
                title=title,
                text=text,
                language=lang,
                doc_type="manuscript_text",
                source="internet_archive",
                url=f"https://archive.org/details/{ident}",
                licence=lic,
                entity_ids=list(dict.fromkeys([q for q in places if q != INDIA] + [INDIA])),
                period_start=y,
                period_end=y,
                fetched_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
                sha256=sha256(text),
                raw_ref=str(path.relative_to(ROOT)),
            )
        )
        outcomes["passed the rules"] += 1

    records, duplicate_groups = drop_repeat_scans(records)
    outcomes["repeat scan of a kept volume"] = sum(len(g) - 1 for g in duplicate_groups)
    with gzip.open(STAGING, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    lengths = [len(r.text) for r in records]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "date": datetime.now(UTC).date().isoformat(),
        "commit": commit,
        "series": SERIES,
        "items_found": len(docs),
        "outcomes": dict(outcomes.most_common()),
        "records": len(records),
        "records_linked_to_a_place": sum(1 for r in records if len(r.entity_ids) > 1),
        "repeat_scan_groups": duplicate_groups,
        "text_length_chars": {"median": statistics.median(lengths), "max": max(lengths)} if lengths else {},
        "requests": {"fetched": wm.fetched, "reused": wm.reused},
    }
    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
