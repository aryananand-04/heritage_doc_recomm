"""Art Institute of Chicago collector by place: public-domain artworks from India (decision 0017).

Pages through the API search for public-domain artworks with place of origin India,
100 a page. The text is the record's own fields (title, artist and place line, date,
medium, classification, style, subjects and terms) and its description. The API
states that the description field is CC BY 4.0 and all other data CC0, so a record
that uses a description carries CC BY 4.0, and one without carries CC0. Entity links
are added by the linking step; until then each record is linked to India (Q668).

Usage:
    uv run python -m src.corpus.chicago_collector

Outputs:
    data/corpus_v1/staging/artic_place_v1.jsonl.gz
    results/collection/artic_place_v1.json
"""

import gzip
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime

from src.corpus import matching
from src.corpus.records import Record, sha256
from src.corpus.wikimedia import ROOT, Wikimedia

RAW_DIR = ROOT / "data/raw/corpus_v1/artic"
STAGING_DIR = ROOT / "data/corpus_v1/staging"
REPORT_DIR = ROOT / "results/collection"
SEARCH = "https://api.artic.edu/api/v1/artworks/search"
INDIA = "Q668"
PAGE = 100
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"
FIELDS = [
    "id", "title", "artist_display", "date_display", "date_start", "date_end", "place_of_origin",
    "medium_display", "classification_title", "style_title", "subject_titles", "term_titles",
    "description", "is_public_domain",
]  # fmt: skip


def query(page: int) -> dict:
    return {
        "query": {"bool": {"must": [{"term": {"is_public_domain": True}}, {"match": {"place_of_origin": "India"}}]}},
        "fields": FIELDS,
        "limit": PAGE,
        "page": page,
    }


def artwork_text(a: dict) -> tuple[str, bool]:
    """The record's text and whether it uses the CC BY description."""
    lines = [matching.plain(a.get("title"))]
    for key, label in (
        ("artist_display", "Made by"),
        ("date_display", "Date"),
        ("medium_display", "Medium"),
        ("classification_title", "Classification"),
        ("style_title", "Style"),
    ):
        value = matching.plain(a.get(key))
        if value:
            lines.append(f"{label}: {value}")
    tags = [t for t in (a.get("subject_titles") or []) + (a.get("term_titles") or []) if t]
    if tags:
        lines.append("Subjects: " + ", ".join(dict.fromkeys(tags)))
    description = matching.plain(a.get("description"))
    if description:
        lines.append(description)
    return "\n".join(line for line in lines if line), bool(description)


def to_record(a: dict, raw_ref: str, fetched_at: datetime) -> Record:
    text, uses_description = artwork_text(a)
    start, end = a.get("date_start"), a.get("date_end")
    start = start if isinstance(start, int) else None
    end = end if isinstance(end, int) else None
    if start is not None and end is not None and start > end:
        start, end = end, start
    return Record(
        id=f"artic:{a['id']}",
        title=matching.plain(a.get("title")) or f"Art Institute of Chicago artwork {a['id']}",
        text=text,
        language="en",
        doc_type="object",
        source="artic",
        url=f"https://www.artic.edu/artworks/{a['id']}",
        licence=CC_BY if uses_description else CC0,
        entity_ids=[INDIA],
        period_start=start,
        period_end=end,
        fetched_at=fetched_at,
        sha256=sha256(text),
        raw_ref=raw_ref,
    )


def main() -> None:
    wm = Wikimedia(RAW_DIR)
    # The API sits behind a bot check and asks clients to identify themselves in this header.
    wm.session.headers["AIC-User-Agent"] = wm.session.headers["User-Agent"]
    found, outcomes, page, pages, licence_text = {}, Counter(), 1, None, ""
    while pages is None or page <= pages:
        name = f"artic_india_{page:03d}"
        params = {"params": json.dumps(query(page), sort_keys=True)}
        data = wm.get_json(name, SEARCH, params)
        pages = (data.get("pagination") or {}).get("total_pages") or 0
        licence_text = (data.get("info") or {}).get("license_text", licence_text)
        path = wm.raw_path(name, {"url": SEARCH, **params})
        fetched_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        for a in data.get("data") or []:
            if not a.get("is_public_domain"):
                outcomes["not public domain"] += 1
            elif not matching.names_india(str(a.get("place_of_origin") or "")):
                outcomes["place does not name India"] += 1
            elif a["id"] not in found:
                found[a["id"]] = to_record(a, f"{path.relative_to(ROOT)}#id={a['id']}", fetched_at)
                outcomes["kept"] += 1
        page += 1

    records = [found[k] for k in sorted(found)]
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(STAGING_DIR / "artic_place_v1.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "date": datetime.now(UTC).date().isoformat(),
        "commit": commit,
        "api_licence_text": licence_text,
        "outcomes": dict(outcomes.most_common()),
        "records": len(records),
        "records_by_licence": dict(Counter(r.licence for r in records)),
        "requests": {"fetched": wm.fetched, "reused": wm.reused},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "artic_place_v1.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
