"""Build the corpus snapshot from the staging files (decision 0004).

Every staging file written by a collector is read, every record is validated
again against the schema (the build stops on the first invalid one and names it),
and a document reached from more than one entity list (a Wikipedia page that is
both a seed's and a neighbour's article) becomes one record whose entity links are
the union of both, in order. Records are sorted by id, so the snapshot is
byte-identical for the same staging files.

Usage:
    uv run python -m src.corpus.build_snapshot

Outputs:
    data/corpus_v1/records.jsonl.gz    the snapshot (gzip with a fixed timestamp, so it is reproducible)
    data/corpus_v1/manifest.json       snapshot SHA-256 and size; per-source counts; every raw file with its SHA-256
"""

import gzip
import hashlib
import io
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime

from src.corpus.records import Record
from src.corpus.wikimedia import ROOT

STAGING_DIR = ROOT / "data/corpus_v1/staging"
RAW_DIR = ROOT / "data/raw"
OUT = ROOT / "data/corpus_v1/records.jsonl.gz"
MANIFEST = ROOT / "data/corpus_v1/manifest.json"
# Staging files that make up the snapshot, in merge order (samples are left out).
SOURCES = [
    "wikipedia_seeds_v1",
    "wikipedia_asi_remainder_v1",
    "wikipedia_neighbours_v1",
    "met_seeds_v1",
    "cleveland_seeds_v1",
    "wellcome_seeds_v1",
    "internet_archive_seeds_v1",
]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def merge(records: list[Record]) -> tuple[list[Record], int]:
    """One record per id; entity links of duplicates are joined in order. Returns the
    records sorted by id and the number of duplicates merged."""
    by_id: dict[str, Record] = {}
    merged = 0
    for r in records:
        if r.id in by_id:
            first = by_id[r.id]
            if first.sha256 != r.sha256:
                raise SystemExit(f"{r.id}: two staging records with different text")
            by_id[r.id] = first.model_copy(update={"entity_ids": list(dict.fromkeys(first.entity_ids + r.entity_ids))})
            merged += 1
        else:
            by_id[r.id] = r
    return [by_id[k] for k in sorted(by_id)], merged


def main() -> None:
    records, per_file = [], {}
    for name in SOURCES:
        path = STAGING_DIR / f"{name}.jsonl.gz"
        if not path.exists():
            raise SystemExit(f"Missing staging file {path.relative_to(ROOT)}; run its collector first")
        n = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    records.append(Record.model_validate_json(line))
                except Exception as e:
                    raise SystemExit(f"{path.name} line {i}: invalid record: {e}") from e
                n += 1
        per_file[name] = n

    records, merged = merge(records)

    # gzip with mtime 0 and no file name, so the same records give the same bytes.
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        for r in records:
            gz.write((r.model_dump_json() + "\n").encode("utf-8"))
    OUT.write_bytes(buf.getvalue())

    raw = sorted(p for p in RAW_DIR.rglob("*.gz") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "snapshot": {
            "path": str(OUT.relative_to(ROOT)),
            "sha256": file_sha256(OUT),
            "bytes": OUT.stat().st_size,
            "records": len(records),
        },
        "staging_records": per_file,
        "duplicates_merged": merged,
        "records_by_source": dict(Counter(r.source for r in records).most_common()),
        "records_by_language": dict(Counter(r.language for r in records).most_common()),
        "records_by_doc_type": dict(Counter(r.doc_type for r in records).most_common()),
        "raw_files": {
            "count": len(raw),
            "bytes": sum(p.stat().st_size for p in raw),
            "files": {str(p.relative_to(ROOT)): file_sha256(p) for p in raw},
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {k: v for k, v in manifest.items() if k != "raw_files"}
    summary["raw_files"] = {k: manifest["raw_files"][k] for k in ("count", "bytes")}
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
