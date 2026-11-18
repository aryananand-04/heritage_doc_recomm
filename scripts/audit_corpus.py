"""Read-only audit of the live corpus before the rebuild.

Answers three questions about data/classified/classified_documents.json
without changing it:

1. Which documents are in scope? Each Wikipedia URL is mapped to its
   Wikidata item, and the item's country (P17) decides a proposed label
   (see docs/decisions/0001-corpus-scope.md):
       keep    P17 includes India (Q668)
       drop    P17 names a present-day country, and none is India
       review  no Wikidata item, no P17 (people, concepts), or P17 names
               only historical states such as the Mughal Empire
2. Which documents are duplicates or near-duplicates? Same URL, same
   normalised title, or heavily overlapping entity sets. The full text is
   not in the repo, so MinHash on text waits for the rebuild.
3. How many synthetic and "Unknown"-source records remain? Reported for
   the live corpus and for the retired legacy file, so the cleaning
   report can state both.

Outputs (DATE defaults to today):
    results/audit/raw/DATE/*.json.gz         API responses, saved untouched (gzipped)
    results/audit/corpus_audit_DATE.json     counts, duplicate groups, run info
    results/audit/indian_check_DATE.csv      one row per document for the hand check

Raw responses are reused when they already exist for DATE, so a rerun with
the same --date reproduces the outputs without touching the network.

Usage:
    uv run python scripts/audit_corpus.py
    uv run python scripts/audit_corpus.py --date 2026-11-17

Requirements (.env):
    CONTACT_EMAIL   included in the User-Agent, as Wikimedia asks
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import date
from itertools import combinations
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/classified/classified_documents.json"
LEGACY_CORPUS = ROOT / "legacy/data/balanced/balanced_documents.json"
OUT_DIR = ROOT / "results/audit"

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
REPO_URL = "https://github.com/aryananand-04/heritage_doc_recomm"

INDIA = "Q668"
BATCH = 50  # the most titles or ids both APIs accept per request for normal clients
PAUSE_SECONDS = 1.0

# Near-duplicate rule on entity sets. Small sets overlap by chance, so a
# pair is only compared when both documents have enough entities.
ENTITY_JACCARD = 0.8
MIN_ENTITIES = 5


# ---------------------------------------------------------------- helpers


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    # Only tracked files count: the outputs this script writes are untracked
    # until committed and would otherwise always mark the run as dirty.
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def batches(items: list, size: int = BATCH):
    for i in range(0, len(items), size):
        yield i // size, items[i : i + size]


def title_from_url(url: str) -> str | None:
    """'https://en.wikipedia.org/wiki/Taj_Mahal' -> 'Taj Mahal'."""
    parsed = urlparse(url or "")
    if "wikipedia.org" not in parsed.netloc or not parsed.path.startswith("/wiki/"):
        return None
    return unquote(parsed.path[len("/wiki/") :]).replace("_", " ")


def normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", title.casefold())).strip()


def entity_set(doc: dict) -> set[str]:
    entities = doc.get("entities") or {}
    return {f"{kind}:{name.casefold().strip()}" for kind, names in entities.items() for name in names or []}


# ---------------------------------------------------------- cached fetches


class CachedClient:
    """GETs JSON, saving each raw response so the run can be replayed offline."""

    def __init__(self, raw_dir: Path, contact_email: str | None):
        self.raw_dir = raw_dir
        self.contact_email = contact_email
        self.fetched = 0
        self.reused = 0

    def get(self, name: str, url: str, params: dict) -> dict:
        # Gzipped: Wikidata entities are large, and these files are committed.
        path = self.raw_dir / f"{name}.json.gz"
        if path.exists():
            self.reused += 1
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)

        if not self.contact_email:
            raise SystemExit(
                f"No cached response at {path.relative_to(ROOT)} and CONTACT_EMAIL is not set in .env; "
                "set it to fetch from Wikimedia."
            )
        headers = {"User-Agent": f"Heritage-Doc-Audit/0.1 ({REPO_URL}; contact: {self.contact_email})"}
        time.sleep(PAUSE_SECONDS)
        r = requests.get(url, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(r.text)
        self.fetched += 1
        return r.json()


def lookup_qids(client: CachedClient, titles: list[str]) -> dict[str, str | None]:
    """Map each Wikipedia title to its Wikidata QID, following redirects."""
    qids: dict[str, str | None] = {}
    for n, chunk in batches(titles):
        data = client.get(
            f"wikipedia_pageprops_{n:02d}",
            WIKIPEDIA_API,
            {
                "action": "query",
                "prop": "pageprops",
                "ppprop": "wikibase_item",
                "titles": "|".join(chunk),
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            },
        )
        query = data.get("query", {})
        # The API rewrites titles in two steps (normalising, then redirects);
        # follow both so every input title can be traced to the page returned.
        renamed = {x["from"]: x["to"] for x in query.get("normalized", [])}
        renamed.update({x["from"]: x["to"] for x in query.get("redirects", [])})
        by_title = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in query.get("pages", [])}
        for title in chunk:
            final = title
            for _ in range(3):  # normalised -> redirect target, at most a couple of hops
                final = renamed.get(final, final)
            qids[title] = by_title.get(final)
    return qids


def claim_ids(entity: dict, prop: str) -> list[str]:
    ids = []
    for claim in entity.get("claims", {}).get(prop, []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and "id" in value:
            ids.append(value["id"])
    return ids


def fetch_entities(client: CachedClient, prefix: str, ids: list[str], props: str) -> dict[str, dict]:
    entities: dict[str, dict] = {}
    for n, chunk in batches(sorted(ids)):
        data = client.get(
            f"{prefix}_{n:02d}",
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": props,
                "languages": "en",
                "format": "json",
            },
        )
        entities.update(data.get("entities", {}))
    return entities


def en_label(entity: dict | None) -> str:
    return ((entity or {}).get("labels", {}).get("en") or {}).get("value", "")


# ---------------------------------------------------------------- the audit


def propose_label(qid: str | None, countries: list[str], historical: set[str]) -> tuple[str, str]:
    if not qid:
        return "review", "no Wikidata item for this article"
    if not countries:
        return "review", "Wikidata item has no country (P17)"
    if INDIA in countries:
        if len(countries) > 1:
            return "keep", f"India is one of {len(countries)} countries"
        return "keep", "country is India"
    # A former state (Mughal Empire, British Raj) says nothing about where the
    # subject is today, so it cannot justify a drop on its own.
    if all(c in historical for c in countries):
        return "review", "country (P17) lists only historical states"
    return "drop", "country is not India"


def duplicate_groups(docs: list[dict]) -> tuple[list[dict], dict[int, int]]:
    """Group documents linked by any duplicate rule (union-find over pairs)."""
    parent = list(range(len(docs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reasons: dict[tuple[int, int], list[str]] = {}

    def link(i: int, j: int, reason: str) -> None:
        reasons.setdefault((min(i, j), max(i, j)), []).append(reason)
        parent[find(i)] = find(j)

    for key, rule in (
        (lambda d: (d.get("url") or "").strip(), "same URL"),
        (lambda d: normalise_title(d.get("title") or ""), "same title"),
    ):
        seen: dict[str, int] = {}
        for i, doc in enumerate(docs):
            k = key(doc)
            if not k:
                continue
            if k in seen:
                link(seen[k], i, rule)
            else:
                seen[k] = i

    sets = [entity_set(d) for d in docs]
    for i, j in combinations(range(len(docs)), 2):
        a, b = sets[i], sets[j]
        if len(a) < MIN_ENTITIES or len(b) < MIN_ENTITIES:
            continue
        jaccard = len(a & b) / len(a | b)
        if jaccard >= ENTITY_JACCARD:
            link(i, j, f"entity Jaccard {jaccard:.2f}")

    members: dict[int, list[int]] = {}
    for i in range(len(docs)):
        members.setdefault(find(i), []).append(i)

    groups, group_of = [], {}
    for idx in (m for m in members.values() if len(m) > 1):
        gid = len(groups) + 1
        for i in idx:
            group_of[i] = gid
        groups.append(
            {
                "group": gid,
                "documents": [{"file_name": docs[i].get("file_name"), "title": docs[i].get("title")} for i in idx],
                "links": [
                    {"a": docs[a].get("file_name"), "b": docs[b].get("file_name"), "reasons": r}
                    for (a, b), r in sorted(reasons.items())
                    if a in idx and b in idx
                ],
            }
        )
    return groups, group_of


def source_counts(docs: list[dict]) -> dict:
    sources = Counter(d.get("source") or "(missing)" for d in docs)
    return {
        "total": len(docs),
        "by_source": dict(sources.most_common()),
        "synthetic": sum(n for s, n in sources.items() if "(Synthetic)" in s),
        "unknown_source": sum(n for s, n in sources.items() if s.startswith("Unknown") and "(Synthetic)" not in s),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--date", default=date.today().isoformat(), help="run date; reuses raw responses saved under it"
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = CachedClient(OUT_DIR / "raw" / args.date, os.environ.get("CONTACT_EMAIL", "").strip() or None)

    docs = json.loads(CORPUS.read_text(encoding="utf-8"))
    print(f"Corpus: {len(docs)} documents from {CORPUS.relative_to(ROOT)}")

    titles = [title_from_url(d.get("url", "")) for d in docs]
    unique_titles = sorted({t for t in titles if t})
    print(f"Looking up {len(unique_titles)} Wikipedia titles...")
    qids = lookup_qids(client, unique_titles)

    found = sorted({q for q in qids.values() if q})
    items = fetch_entities(client, "wikidata_items", found, "labels|claims")
    countries = {q: claim_ids(items.get(q, {}), "P17") for q in found}
    instance_of = {q: claim_ids(items.get(q, {}), "P31") for q in found}

    # A country with a dissolved date (P576) is a historical state. One SPARQL
    # query asks only for that property; full country entities run to tens of MB.
    country_ids = sorted({c for ids in countries.values() for c in ids})
    sparql = "SELECT DISTINCT ?c WHERE { VALUES ?c { %s } ?c wdt:P576 ?end . }" % " ".join(
        f"wd:{c}" for c in country_ids
    )
    dissolved = client.get("wikidata_dissolved_countries", WIKIDATA_SPARQL, {"query": sparql, "format": "json"})
    historical = {b["c"]["value"].rsplit("/", 1)[-1] for b in dissolved["results"]["bindings"]}

    # Labels for the countries and types, so the hand check reads names, not ids.
    label_ids = sorted({x for ids in (*countries.values(), *instance_of.values()) for x in ids})
    label_entities = fetch_entities(client, "wikidata_labels", label_ids, "labels")

    def names(ids: list[str]) -> str:
        return "; ".join(f"{en_label(label_entities.get(x)) or '?'} ({x})" for x in ids)

    groups, group_of = duplicate_groups(docs)

    rows = []
    for i, (doc, title) in enumerate(zip(docs, titles)):
        qid = qids.get(title) if title else None
        doc_countries = countries.get(qid, []) if qid else []
        label, reason = propose_label(qid, doc_countries, historical)
        rows.append(
            {
                "file_name": doc.get("file_name", ""),
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "qid": qid or "",
                "wikidata_label": en_label(items.get(qid)) if qid else "",
                "instance_of": names(instance_of.get(qid, [])) if qid else "",
                "countries": names(doc_countries),
                "proposed_label": label,
                "reason": reason,
                "duplicate_group": group_of.get(i, ""),
                "final_label": "",
                "note": "",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"indian_check_{args.date}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = Counter(r["proposed_label"] for r in rows)
    report = {
        "date": args.date,
        "git": git_state(),
        "inputs": {
            str(CORPUS.relative_to(ROOT)): sha256_of(CORPUS),
            str(LEGACY_CORPUS.relative_to(ROOT)): sha256_of(LEGACY_CORPUS),
        },
        "rules": {
            "scope": "docs/decisions/0001-corpus-scope.md",
            "india_qid": INDIA,
            "entity_jaccard_threshold": ENTITY_JACCARD,
            "min_entities_for_jaccard": MIN_ENTITIES,
        },
        "live_corpus": {
            **source_counts(docs),
            "without_wikipedia_url": sum(1 for t in titles if not t),
            "without_wikidata_item": sum(1 for r in rows if not r["qid"]),
            "proposed_labels": {k: labels.get(k, 0) for k in ("keep", "drop", "review")},
            "duplicate_groups": len(groups),
            "documents_in_duplicate_groups": len(group_of),
        },
        "legacy_corpus": source_counts(json.loads(LEGACY_CORPUS.read_text(encoding="utf-8"))),
        "duplicate_groups": groups,
        "api": {"requests_fetched": client.fetched, "responses_reused": client.reused},
    }
    report_path = OUT_DIR / f"corpus_audit_{args.date}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    live = report["live_corpus"]
    print(f"Proposed labels: {live['proposed_labels']}")
    print(f"Duplicate groups: {live['duplicate_groups']} ({live['documents_in_duplicate_groups']} documents)")
    print(
        f"Legacy file: {report['legacy_corpus']['synthetic']} synthetic, "
        f"{report['legacy_corpus']['unknown_source']} Unknown-source"
    )
    print(f"Wrote {report_path.relative_to(ROOT)} and {csv_path.relative_to(ROOT)}")
    print("Next: fill final_label in the CSV for every drop and review row, and a sample of keep rows.")


if __name__ == "__main__":
    main()
