"""Internet Archive collector: public-domain texts that name a seed (decisions 0013, 0015, 0016).

For each non-ASI seed, the advanced search (public-domain and CC0 texts only, the
probe's filter) returns the first 20 items; up to the cap are kept, in rank order,
when the title or subjects name the seed, India is named (src/corpus/matching.py),
the item is in English, Hindi or Kannada and it has OCR text of at least 500
characters. The record's text is that OCR text with whitespace normalised; the
uploader's description is never used as text (decision 0016).

Usage:
    uv run python -m src.corpus.internet_archive_collector --sample 20
    uv run python -m src.corpus.internet_archive_collector

Outputs:
    data/corpus_v1/staging/internet_archive_seeds_v1.jsonl.gz    validated records (not committed)
    results/collection/internet_archive_seeds_v1.json             counts by outcome; text lengths
"""

import argparse
import csv
import gzip
import json
import random
import re
import statistics
import subprocess
from collections import Counter
from datetime import UTC, datetime
from urllib.parse import quote

from src.corpus import matching
from src.corpus.collection_probe import IA, IA_FREE, PROBE_STRATA, term
from src.corpus.records import Record, sha256
from src.corpus.wikimedia import ROOT, Wikimedia

RAW_DIR = ROOT / "data/raw/corpus_v1/internet_archive"
SEEDS = ROOT / "data/seeds/seeds_v1.csv"
STAGING_DIR = ROOT / "data/corpus_v1/staging"
REPORT_DIR = ROOT / "results/collection"
CAP = 5
LOOK_AT = 20
MIN_CHARS = 500
PD_MARK = "https://creativecommons.org/publicdomain/mark/1.0/"
RANDOM_SEED = 20261119
FIELDS = [
    "identifier",
    "title",
    "subject",
    "description",
    "language",
    "licenseurl",
    "possible-copyright-status",
    "date",
]
LANGS = {"eng": "en", "english": "en", "en": "en", "hin": "hi", "hindi": "hi", "hi": "hi", "kan": "kn", "kannada": "kn"}
# Indian languages as the archive writes them; an item in one of them points to India.
INDIC = {
    "hin", "hindi", "san", "sanskrit", "kan", "kannada", "tam", "tamil", "tel", "telugu", "ben", "bengali",
    "mar", "marathi", "guj", "gujarati", "pan", "punjabi", "ori", "oriya", "mal", "malayalam", "urd", "urdu",
    "pra", "prakrit", "pli", "pali", "asm", "assamese",
}  # fmt: skip
# Items dated from this year on are left out: an uploader's public-domain mark is not safe
# for modern Indian works (copyright runs 60 years), and they are records such as
# parliamentary debates rather than heritage texts.
DATED_BEFORE = 1950


def as_list(value) -> list[str]:
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def metadata_language(doc: dict) -> str | None:
    """The corpus language the item's metadata names, or None if it names none or several."""
    found = {LANGS.get(v.strip().lower()) for v in as_list(doc.get("language"))}
    found.discard(None)
    return found.pop() if len(found) == 1 else None


def script_language(text: str) -> str | None:
    """The corpus language from the OCR text's script: Latin is English, Kannada script is
    Kannada, Devanagari is Hindi only when the metadata also names Hindi (Devanagari is
    also used for Sanskrit and Marathi). None when no script holds most of the letters."""
    latin = len(re.findall(r"[A-Za-z]", text))
    deva = len(re.findall(r"[\u0900-\u097F]", text))
    kann = len(re.findall(r"[\u0C80-\u0CFF]", text))
    total = sum(1 for ch in text if ch.isalpha())  # every script counts, so Urdu or Tamil text fails the share
    if not total:
        return None
    if latin / total > 0.8:
        return "en"
    if kann / total > 0.8:
        return "kn"
    if deva / total > 0.8:
        return "hi?"
    return None


def language(doc: dict, text: str) -> str | None:
    """The text's script decides (archive language metadata is often wrong: an English
    edition tagged Hindi, Urdu books tagged Hindi); the metadata only tells Hindi apart
    from other languages written in Devanagari."""
    by_script = script_language(text)
    if by_script == "hi?":
        return "hi" if metadata_language(doc) == "hi" else None
    return by_script


def indic(doc: dict) -> bool:
    return any(v.strip().lower() in INDIC for v in as_list(doc.get("language")))


def names_seed(doc: dict, search: str) -> bool:
    short = " ".join(as_list(doc.get("title")) + as_list(doc.get("subject")))
    places = f"{short} {matching.plain(' '.join(as_list(doc.get('description'))))}"
    return matching.names_seed(short, search) and (matching.names_india(places) or indic(doc))


def licence(meta: dict) -> str | None:
    url = str(meta.get("licenseurl") or "").strip()
    if url.startswith("http") and ("publicdomain" in url or "zero" in url):
        return url.replace("http://", "https://", 1)
    if str(meta.get("possible-copyright-status") or "").upper() == "NOT_IN_COPYRIGHT":
        return PD_MARK
    return None


def clean(text: str) -> str:
    text = re.sub(r"[ \t\f\v]+", " ", text.replace("\r", ""))
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


def ocr_file(meta: dict) -> str | None:
    for f in meta.get("files") or []:
        if f.get("format") == "DjVuTXT":
            return f["name"]
    return None


def year(value) -> int | None:
    m = re.match(r"^\s*(\d{3,4})", str(value or ""))
    return int(m.group(1)) if m else None


def item_year(doc: dict) -> int | None:
    """The item's date, or failing that the latest plausible year in its title (Digital
    Library of India uploads often carry the year only there: "... 1953 London").
    Undated items are left out: without a year their public-domain mark cannot be checked."""
    y = year(doc.get("date"))
    if y is not None:
        return y
    title = " ".join(as_list(doc.get("title")))
    found = [int(m) for m in re.findall(r"\b(1[4-9]\d\d|20[0-2]\d)\b", title)]
    return max(found) if found else None


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
    outcomes, kept_per_seed = Counter(), Counter()
    for s in seeds:
        search = term(s["label"], s["stratum"])
        t = search.replace('"', "")
        params = {
            "q": f'(title:("{t}") OR subject:("{t}")) AND mediatype:texts AND {IA_FREE}',
            "fl[]": FIELDS,
            "rows": LOOK_AT,
            "output": "json",
        }
        data = wm.get_json(f"ia_search_{s['qid']}", IA, params)
        for doc in data.get("response", {}).get("docs", []):
            if kept_per_seed[s["qid"]] >= CAP:
                break
            ident = doc["identifier"]
            if ident in found:  # already kept for another seed: link it, no second download
                found[ident]["seeds"].append(s["qid"])
                kept_per_seed[s["qid"]] += 1
                outcomes["kept (also for another seed)"] += 1
                continue
            if not names_seed(doc, search):
                outcomes["does not name the seed or India"] += 1
                continue
            described = " ".join(
                as_list(doc.get("title")) + as_list(doc.get("subject")) + as_list(doc.get("description"))
            )
            if s["stratum"] == "dynasty" and not matching.names_dynasty(matching.plain(described), search, s["label"]):
                outcomes["dynasty name without a dynasty word"] += 1
                continue
            y = item_year(doc)
            if y is None:
                outcomes["undated (public-domain status cannot be checked)"] += 1
                continue
            if y >= DATED_BEFORE:
                outcomes[f"dated {DATED_BEFORE} or later"] += 1
                continue
            meta_url = f"https://archive.org/metadata/{ident}"
            meta = wm.get_json(f"ia_meta_{ident}", meta_url, {})
            lic = licence(meta.get("metadata") or {})
            name = ocr_file(meta)
            if lic is None:
                outcomes["no public-domain licence on the item"] += 1
                continue
            if name is None:
                outcomes["no OCR text"] += 1
                continue
            text_url = f"https://archive.org/download/{ident}/{quote(name)}"
            text = clean(wm.get_text(f"ia_text_{ident}", text_url))
            if len(text) < MIN_CHARS:
                outcomes[f"OCR text under {MIN_CHARS} characters"] += 1
                continue
            lang = language(doc, text)
            if lang is None:
                outcomes["text not in English, Hindi or Kannada"] += 1
                continue
            path = wm.raw_path(f"ia_text_{ident}", {"url": text_url}, ".txt.gz")
            outcomes["kept"] += 1
            kept_per_seed[s["qid"]] += 1
            found[ident] = {
                "doc": doc,
                "text": text,
                "language": lang,
                "licence": lic,
                "seeds": [s["qid"]],
                "raw_ref": str(path.relative_to(ROOT)),
                "fetched_at": datetime.fromtimestamp(path.stat().st_mtime, UTC),
            }

    records = []
    for ident, e in found.items():
        y = item_year(e["doc"])
        records.append(
            Record(
                id=f"internet_archive:{ident}",
                title=matching.plain(" ".join(as_list(e["doc"].get("title")))) or ident,
                text=e["text"],
                language=e["language"],
                doc_type="manuscript_text",
                source="internet_archive",
                url=f"https://archive.org/details/{ident}",
                licence=e["licence"],
                entity_ids=list(dict.fromkeys(e["seeds"])),
                period_start=y,
                period_end=y,
                fetched_at=e["fetched_at"],
                sha256=sha256(e["text"]),
                raw_ref=e["raw_ref"],
            )
        )
    stem = "internet_archive_seeds_v1" + ("_sample" if args.sample else "")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(STAGING_DIR / f"{stem}.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    lengths = [len(r.text) for r in records]
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
        "records_by_language": dict(Counter(r.language for r in records)),
        "text_length_chars": {"median": statistics.median(lengths), "min": min(lengths), "max": max(lengths)}
        if lengths
        else {},
        "requests": {"fetched": wm.fetched, "reused": wm.reused},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
