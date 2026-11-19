"""Wikipedia collector: one plain-text article per entity and language (decisions 0003, 0013).

For every entity in the list, the English, Hindi and Kannada article titles come
from Wikidata sitelinks. Each article is fetched with the TextExtracts API in the
same request as its revision ID and canonical URL, and the raw response is saved
untouched. Cleaning into records (decision 0004) is a separate step that reads only
the saved responses.

Usage:
    uv run python -m src.corpus.wikipedia_collector --sample 20    # try a fixed sample first
    uv run python -m src.corpus.wikipedia_collector                # every selected seed
    uv run python -m src.corpus.wikipedia_collector --entities data/seeds/neighbours_v1.csv

Outputs:
    data/corpus_v1/staging/wikipedia_<entity list>.jsonl.gz   validated records (not committed; the snapshot is)
    results/collection/wikipedia_<entity list>.json          counts: requested, kept, skipped by reason, lengths
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
from pathlib import Path

from src.corpus.records import Record, sha256
from src.corpus.seeds import batches
from src.corpus.wikimedia import ROOT, Wikimedia, qid, values

RAW_DIR = ROOT / "data/raw/corpus_v1/wikipedia"
STAGING_DIR = ROOT / "data/corpus_v1/staging"
REPORT_DIR = ROOT / "results/collection"
SEEDS = ROOT / "data/seeds/seeds_v1.csv"
LANGS = ("en", "hi", "kn")
LICENCE = "https://creativecommons.org/licenses/by-sa/4.0/"
RANDOM_SEED = 20261119

# Trailing sections that hold links and citations rather than article text. Their
# headings survive TextExtracts with little or no body; everything from the first
# of them to the end is dropped.
TAIL_SECTIONS = {
    "en": {"see also", "references", "notes", "external links", "further reading", "bibliography", "sources"},
    "hi": {
        "इन्हें भी देखें",
        "यह भी देखें",
        "ये भी देखें",
        "सन्दर्भ",
        "संदर्भ",
        "सन्दर्भ ग्रन्थ",
        "सन्दर्भ ग्रंथ",
        "बाहरी कड़ियाँ",
        "बाहरी कडियाँ",
        "बाह्य कड़ियाँ",
        "टिप्पणी",
        "टिप्पणियाँ",
        "टीका टिप्पणी",
        "टीका-टिप्पणी",
        "स्रोत",
        "आगे पढ़ें",
        "यह सभी देखें",
        "इन्हें भी देखे",
    },
    "kn": {
        "ಉಲ್ಲೇಖ",
        "ಉಲ್ಲೇಖಗಳು",
        "ಆಕರ",
        "ಆಕರಗಳು",
        "ಬಾಹ್ಯ ಕೊಂಡಿ",
        "ಬಾಹ್ಯ ಕೊಂಡಿಗಳು",
        "ಹೊರಗಿನ ಕೊಂಡಿಗಳು",
        "ಇದನ್ನೂ ನೋಡಿ",
        "ಇವನ್ನೂ ನೋಡಿ",
        "ಇದನ್ನು ನೋಡಿ",
        "ಟಿಪ್ಪಣಿಗಳು",
        "ಹೆಚ್ಚಿನ ಓದಿಗೆ",
        "ಮತ್ತಷ್ಟು ಓದುವಿಕೆ",
        "ಬಾಹ್ಯ ಸಂಪರ್ಕಗಳು",
        "ಬಾಹ್ಯ ಸಂಪರ್ಕ",
        "ಸಹ ನೋಡಿ",
        "ನೋಡಿ",
        "ಆಧಾರ",
        "ಗ್ರಂಥಸೂಚಿ",
        "ಹೆಚ್ಚಿಗೆ ಓದಲು",
    },
}
HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$")


def article_params(title: str) -> dict:
    return {
        "action": "query",
        "prop": "extracts|revisions|info|pageprops",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "rvprop": "ids|timestamp",
        "inprop": "url",
        "ppprop": "disambiguation",
        "redirects": "1",
        "titles": title,
    }


def clean_text(text: str, lang: str) -> str:
    """Cut the trailing link and citation sections, drop empty sections, tidy blank lines."""
    lines = text.splitlines()
    tail = TAIL_SECTIONS[lang]
    for i, line in enumerate(lines):
        m = HEADING.match(line)
        if m and m.group(2).strip().lower() in tail:
            lines = lines[:i]
            break
    kept: list[str] = []
    for line in lines:
        m = HEADING.match(line)
        # A heading directly followed by another heading of the same or higher level
        # has no body (common after TextExtracts drops tables); drop it.
        if kept and HEADING.match(kept[-1]) and m and len(m.group(1)) <= len(HEADING.match(kept[-1]).group(1)):
            kept.pop()
        kept.append(line.rstrip())
    while kept and (not kept[-1].strip() or HEADING.match(kept[-1])):
        kept.pop()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def to_record(entity: dict, lang: str, response: dict, raw_ref: str, fetched_at: datetime) -> tuple[Record | None, str]:
    """A record from one saved response, or None and the reason it was skipped."""
    pages = response.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or pages[0].get("invalid"):
        return None, "missing"
    page = pages[0]
    if "disambiguation" in page.get("pageprops", {}):
        return None, "disambiguation"
    text = clean_text(page.get("extract") or "", lang)
    if not text:
        return None, "empty"
    rev = (page.get("revisions") or [{}])[0].get("revid")
    record = Record(
        id=f"wikipedia:{lang}:{page['pageid']}",
        title=page["title"],
        text=text,
        language=lang,
        doc_type=entity["doc_type"],
        source="wikipedia",
        url=page["fullurl"],
        licence=LICENCE,
        entity_ids=[entity["qid"]],
        fetched_at=fetched_at,
        sha256=sha256(text),
        raw_ref=f"{raw_ref}#revid={rev}",
    )
    return record, "kept"


def sitelink_titles(wm: Wikimedia, qids: list[str]) -> dict[tuple[str, str], str]:
    pairs = " ".join(f'(<https://{lang}.wikipedia.org/> "{lang}")' for lang in LANGS)
    out = {}
    for n, chunk in batches(sorted(qids)):
        rows = wm.sparql(
            f"sitelinks_{n:02d}",
            f"""SELECT ?i ?lang ?title WHERE {{ VALUES ?i {{ {values(chunk)} }} VALUES (?wiki ?lang) {{ {pairs} }}
                ?a schema:about ?i ; schema:isPartOf ?wiki ; schema:name ?title . }}""",
        )
        out |= {(qid(r["i"]), r["lang"]): r["title"] for r in rows}
    return out


def load_entities(path: Path, sample: int | None) -> list[dict]:
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["status"] == "selected"]
    if sample:
        rows = random.Random(RANDOM_SEED).sample(sorted(rows, key=lambda r: r["qid"]), sample)
    return sorted(rows, key=lambda r: r["qid"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entities", type=Path, default=SEEDS, help="CSV with qid, label, doc_type and status columns")
    ap.add_argument("--sample", type=int, help="fetch only a fixed random sample of this many entities")
    args = ap.parse_args()
    args.entities = args.entities.resolve()  # relative paths from the command line

    entities = load_entities(args.entities, args.sample)
    wm = Wikimedia(RAW_DIR)
    titles = sitelink_titles(wm, [e["qid"] for e in entities])

    records, outcomes = [], Counter()
    seen_ids: set[str] = set()
    for e in entities:
        for lang in LANGS:
            title = titles.get((e["qid"], lang))
            if not title:
                continue
            name = f"{lang}_{e['qid']}"
            params = article_params(title)
            response = wm.wiki(name, params, lang)
            path = wm.raw_path(name, wm.wiki_request(params, lang))
            fetched_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            record, outcome = to_record(e, lang, response, str(path.relative_to(ROOT)), fetched_at)
            if record and record.id in seen_ids:
                outcome, record = "duplicate_page", None  # two entities whose sitelinks resolve to one page
            outcomes[(lang, outcome)] += 1
            if record:
                seen_ids.add(record.id)
                records.append(record)

    # One staging file and report per entity list (seeds, ASI remainder, neighbours).
    stem = f"wikipedia_{args.entities.stem}" + ("_sample" if args.sample else "")
    out = STAGING_DIR / f"{stem}.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    lengths = {lang: [len(r.text) for r in records if r.language == lang] for lang in LANGS}
    # The last heading of each article, so tail sections missing from TAIL_SECTIONS show up.
    last_headings = Counter()
    for r in records:
        heads = [m.group(2) for m in map(HEADING.match, r.text.splitlines()) if m]
        if heads:
            last_headings[(r.language, heads[-1].lower())] += 1
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "date": datetime.now(UTC).date().isoformat(),
        "commit": commit,
        "entities_file": str(args.entities.relative_to(ROOT)),
        "entities": len(entities),
        "sample": args.sample,
        "articles_with_sitelink": {lang: sum(1 for (q, lg) in titles if lg == lang) for lang in LANGS},
        "outcomes": {f"{lang} {o}": n for (lang, o), n in sorted(outcomes.items())},
        "records": len(records),
        "records_by_doc_type": dict(Counter(r.doc_type for r in records).most_common()),
        "text_length_chars": {
            lang: {"median": statistics.median(v), "min": min(v), "max": max(v)} for lang, v in lengths.items() if v
        },
        "most_common_last_headings": {
            lang: [[h, n] for (lg, h), n in last_headings.most_common() if lg == lang][:25] for lang in LANGS
        },
        "requests": {"fetched": wm.fetched, "reused": wm.reused},
        "output": str(out.relative_to(ROOT)),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
