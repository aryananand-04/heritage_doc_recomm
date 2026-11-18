"""Polite, cached access to Wikidata SPARQL and the Wikipedia API.

Every response is saved untouched (gzipped) under a run directory before it is
used, and reused on the next run, so a build can be replayed offline and the
raw data archived with the release (decision 0004).
"""

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SPARQL_URL = "https://query.wikidata.org/sparql"
REPO_URL = "https://github.com/aryananand-04/heritage_doc_recomm"


class Wikimedia:
    def __init__(self, raw_dir: Path, pause: float = 1.0):
        load_dotenv(ROOT / ".env")
        self.raw_dir = raw_dir
        self.pause = pause
        self.fetched = 0
        self.reused = 0
        email = os.environ.get("CONTACT_EMAIL", "").strip()
        self.session = requests.Session()
        # Wikimedia asks automated clients to identify themselves with a contact.
        self.session.headers["User-Agent"] = f"HeritageRec-Corpus/1.0 ({REPO_URL}; contact: {email or 'unset'})"
        self.can_fetch = bool(email)

    def _cached(self, name: str, request: dict, fetch) -> dict:
        # The file name carries a hash of the exact request, so a changed query
        # never silently reuses an old answer.
        digest = hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()[:10]
        path = self.raw_dir / f"{name}_{digest}.json.gz"
        if path.exists():
            self.reused += 1
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        if not self.can_fetch:
            raise SystemExit(f"No cached response {path.name} and CONTACT_EMAIL is not set in .env")
        time.sleep(self.pause)
        for attempt in range(4):
            r = fetch()
            # The query service answers 429 when busy and 5xx when overloaded; back off and retry.
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(int(r.headers.get("Retry-After", 10 * (attempt + 1))))
                continue
            r.raise_for_status()
            break
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(r.text)
        self.fetched += 1
        return r.json()

    def sparql(self, name: str, query: str) -> list[dict]:
        """Run a SPARQL query and return its bindings as plain {var: value} dicts."""
        data = self._cached(
            name,
            {"sparql": query},
            lambda: self.session.post(
                SPARQL_URL,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=300,
            ),
        )
        return [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]

    def wiki(self, name: str, params: dict, lang: str = "en") -> dict:
        """Call the MediaWiki action API of the given language edition."""
        url = f"https://{lang}.wikipedia.org/w/api.php"
        full = {"format": "json", "formatversion": "2", **params}
        return self._cached(name, {"url": url, **full}, lambda: self.session.get(url, params=full, timeout=120))


def qid(uri: str) -> str:
    """'http://www.wikidata.org/entity/Q668' -> 'Q668'."""
    return uri.rsplit("/", 1)[-1]


def values(qids) -> str:
    """A SPARQL VALUES body for a list of QIDs."""
    return " ".join(f"wd:{q}" for q in qids)
