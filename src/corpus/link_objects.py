"""Link museum objects to Wikidata entities in three layers (decision 0017).

For each object record in the museum staging files, the raw museum record is read
again (through the record's raw_ref) and its fields are matched against the
gazetteer, each field only against the entity types it can name:

- layer 1, links the museum gives (already on the record at collection);
- layer 2, structured fields: place fields against places and polities; period and
  dynasty fields against polities; object-type fields against culture forms and
  styles;
- layer 3, title and description: any type, as "mentions" with a confidence, kept
  in the links file only (the mentions edge layer, decision 0006), never in a
  record's entity links.

A name that belongs to several entities of the allowed types (Patan, Bijapur) is
ambiguous: it is written to the links file with confidence 1/n and kept out of the
record's entity links. Several polities sharing a core name (Chola dynasty, Chola
Empire) are all linked, since they are items for the same polity.

Usage:
    uv run python -m src.corpus.link_objects

Outputs:
    data/corpus_v1/staging/<source>_linked.jsonl.gz   records with layer 1 and 2 entity links
    data/corpus_v1/links_v1.jsonl.gz                  every link: doc, qid, layer, field, name, confidence
    results/linking/object_links_v1.json               counts per source and layer
    results/linking/link_check_sample_v1.csv           200 links, stratified by layer, for the author's check
"""

import csv
import gzip
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime

from src.corpus import matching
from src.corpus.records import Record
from src.corpus.wikimedia import ROOT

GAZETTEER = ROOT / "data/gazetteer/gazetteer_v1.jsonl.gz"
STAGING_DIR = ROOT / "data/corpus_v1/staging"
LINKS = ROOT / "data/corpus_v1/links_v1.jsonl.gz"
REPORT_DIR = ROOT / "results/linking"
INDIA = "Q668"
RANDOM_SEED = 20261119
SAMPLE_SIZE = 200
MAX_WORDS = 6
INPUTS = ["met_seeds_v1", "met_place_v1", "cleveland_seeds_v1", "cleveland_place_v1", "artic_place_v1"]

PLACE_TYPES = {"place", "polity"}
POLITY_TYPES = {"polity"}
FORM_TYPES = {"culture", "style"}
ALL_TYPES = {"place", "polity", "culture", "style", "person", "site"}

POLITY_PREFIX = re.compile(r"^(?:kingdom|empire|state) of (?:the )?", re.I)
POLITY_SUFFIX = re.compile(r"\s+(?:dynasty|empire|kingdom|state)$", re.I)
PLACE_SUFFIX = re.compile(r"\s+district$", re.I)
# Short forms that are ordinary words ("South district" -> "South", "Slave dynasty" -> "Slave").
COMMON_WORDS = {"south", "north", "east", "west", "central", "new", "old", "upper", "lower", "slave", "great", "little"}


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.lower()))


def core_names(name: str, etype: str) -> list[str]:
    """Short forms museums use: 'Mughal Empire' -> 'Mughal', 'Kingdom of Marwar' -> 'Marwar',
    'Kangra district' -> 'Kangra'. 'Sultanate' is kept ('Delhi Sultanate' is not Delhi)."""
    out = []
    if etype == "polity":
        short = POLITY_SUFFIX.sub("", POLITY_PREFIX.sub("", name)).strip()
        if short and short.lower() != name.lower():
            out.append(short)
    if etype == "place":
        short = PLACE_SUFFIX.sub("", name).strip()
        if short.lower() != name.lower():
            out.append(short)
    return [s for s in out if s.lower() not in COMMON_WORDS]


class Gazetteer:
    def __init__(self, path=GAZETTEER):
        self.index: dict[str, list[dict]] = defaultdict(list)
        self.label: dict[str, str] = {}
        self.type: dict[str, str] = {}
        # Qualifiers of all an entity's names: 'Pala dynasty (Kamarupa)' makes every name of
        # that entity, unqualified aliases included, require 'Kamarupa' in the text.
        self.qualifiers: dict[str, set] = defaultdict(set)
        for line in gzip.open(path, "rt", encoding="utf-8"):
            e = json.loads(line)
            names = e["names"]["en"]
            self.label[e["qid"]] = e.get("label", {}).get("en") or (names[0] if names else e["qid"])
            self.type[e["qid"]] = e["type"]
            for name in names:
                if matching.qualifier(name):
                    self.qualifiers[e["qid"]].add(matching.qualifier(name))
            for name in names:
                qual = matching.qualifier(name)
                base = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
                for n, derived in [(base, False)] + [(c, True) for c in core_names(base, e["type"])]:
                    key = normalise(n)
                    if len(key) >= 3:
                        self.index[key].append(
                            {"qid": e["qid"], "type": e["type"], "derived": derived, "qualifier": qual}
                        )

    def find(self, text: str, types: set[str], derived: bool = True) -> list[tuple[str, list[dict]]]:
        """Longest names first, without overlap: (matched name, candidate entries of the allowed types)."""
        words = normalise(text).split()
        taken = [False] * len(words)
        found = []
        for size in range(min(MAX_WORDS, len(words)), 0, -1):
            for i in range(len(words) - size + 1):
                if any(taken[i : i + size]):
                    continue
                key = " ".join(words[i : i + size])
                cands = [c for c in self.index.get(key, ()) if c["type"] in types and (derived or not c["derived"])]
                # A full name beats a short form made from another entity's name
                # ("Lucknow" the city over "Lucknow" from "Lucknow district").
                if any(not c["derived"] for c in cands):
                    cands = [c for c in cands if not c["derived"]]
                cands = [
                    c
                    for c in cands
                    if not self.qualifiers.get(c["qid"])
                    or any(matching.names_seed(text, q) for q in self.qualifiers[c["qid"]])
                ]
                if cands:
                    uniq = {c["qid"]: c for c in cands}
                    found.append((key, list(uniq.values())))
                    taken[i : i + size] = [True] * size
        return found


def met_fields(o: dict) -> dict[str, tuple[str, set, bool]]:
    """field -> (text, allowed types, needs a polity word next to a derived polity name)."""
    place = ", ".join(
        str(o.get(k) or "") for k in ("culture", "country", "region", "subregion", "state", "county", "city")
    )
    # The Met's culture field names periods too ("Indian, Mughal"), so it is read as a period
    # field as well, where it is matched only against polities.
    period = ", ".join(str(o.get(k) or "") for k in ("period", "dynasty", "reign", "culture"))
    form = ", ".join(str(o.get(k) or "") for k in ("objectName", "classification"))
    return {
        "place": (place, PLACE_TYPES, True),
        "period": (period, POLITY_TYPES, False),
        "object type": (form, FORM_TYPES, False),
    }


def met_free(o: dict) -> str:
    return " ".join([str(o.get("title") or "")] + [t.get("term", "") for t in o.get("tags") or []])


def cleveland_fields(o: dict) -> dict[str, tuple[str, set, bool]]:
    culture = "; ".join(o.get("culture") or []) if isinstance(o.get("culture"), list) else str(o.get("culture") or "")
    return {
        "place": (culture, PLACE_TYPES, True),
        "period": (f"{culture}. {matching.plain(o.get('tombstone'))}", POLITY_TYPES, True),
        "object type": (str(o.get("type") or ""), FORM_TYPES, False),
    }


def cleveland_free(o: dict) -> str:
    return f"{matching.plain(o.get('title'))} {matching.plain(o.get('description'))}"


def artic_fields(a: dict) -> dict[str, tuple[str, set, bool]]:
    artist = matching.plain(a.get("artist_display"))
    return {
        "place": (f"{a.get('place_of_origin') or ''}, {artist}", PLACE_TYPES, True),
        "period": (f"{a.get('style_title') or ''}. {artist}", POLITY_TYPES, True),
        "object type": (f"{a.get('classification_title') or ''}, {a.get('style_title') or ''}", FORM_TYPES, False),
    }


def artic_free(a: dict) -> str:
    tags = " ".join((a.get("subject_titles") or []) + (a.get("term_titles") or []))
    return f"{matching.plain(a.get('title'))} {matching.plain(a.get('description'))} {tags}"


SOURCES = {
    "met": (met_fields, met_free),
    "cleveland": (cleveland_fields, cleveland_free),
    "artic": (artic_fields, artic_free),
}


class RawReader:
    """The raw museum record behind a staging record, from its raw_ref (path, or path#id=N)."""

    def __init__(self):
        self.cache: dict[str, object] = {}

    def get(self, raw_ref: str) -> dict:
        path, _, frag = raw_ref.partition("#")
        if path not in self.cache:
            self.cache[path] = json.load(gzip.open(ROOT / path, "rt", encoding="utf-8"))
        data = self.cache[path]
        if not frag.startswith("id="):
            return data
        wanted = frag.split("=", 1)[1]
        return next(o for o in data.get("data") or [] if str(o.get("id")) == wanted)


def link_record(rec: Record, raw: dict, gaz: Gazetteer, fields_fn, free_fn) -> tuple[list[str], list[dict]]:
    links = [
        {"doc": rec.id, "qid": q, "layer": 1, "field": "museum link", "name": "", "confidence": 1.0}
        for q in rec.entity_ids
        if q != INDIA
    ]
    core = [q for q in rec.entity_ids if q != INDIA]
    for field, (text, types, needs_word) in fields_fn(raw).items():
        for name, cands in gaz.find(text, types):
            cands = [c for c in cands if not (needs_word and c["derived"] and c["type"] == "polity")
                     or matching.names_dynasty(text, name)]  # fmt: skip
            if not cands:
                continue
            same_polity = all(c["type"] == "polity" for c in cands) and len(cands) <= 3
            ambiguous = len(cands) > 1 and not same_polity
            cands = [c for c in cands if c["qid"] != INDIA]  # India is the default link, not a finding
            for c in cands:
                links.append(
                    {"doc": rec.id, "qid": c["qid"], "layer": 2, "field": field, "name": name,
                     "confidence": round(1 / len(cands), 3) if ambiguous else 1.0}
                )  # fmt: skip
                if not ambiguous and c["qid"] != INDIA:
                    core.append(c["qid"])
    already = {link["qid"] for link in links}
    for name, cands in gaz.find(free_fn(raw), ALL_TYPES, derived=False):
        if len(name.split()) == 1 and (len(name) < 5 or len(cands) > 1):
            continue  # single short or shared words in free text are too often something else
        for c in cands:
            if c["qid"] not in already and c["qid"] != INDIA:
                links.append(
                    {"doc": rec.id, "qid": c["qid"], "layer": 3, "field": "title or description", "name": name,
                     "confidence": round(1 / len(cands), 3)}
                )  # fmt: skip
    return list(dict.fromkeys(core)) or [INDIA], links


def main() -> None:
    gaz = Gazetteer()
    reader = RawReader()
    all_links, report = [], {}
    for stem in INPUTS:
        path = STAGING_DIR / f"{stem}.jsonl.gz"
        if not path.exists():
            report[stem] = "missing (collector not run yet)"
            continue
        recs = [Record.model_validate_json(line) for line in gzip.open(path, "rt", encoding="utf-8")]
        out, counts = [], Counter()
        for rec in recs:
            fields_fn, free_fn = SOURCES[rec.source]
            entity_ids, links = link_record(rec, reader.get(rec.raw_ref), gaz, fields_fn, free_fn)
            out.append(rec.model_copy(update={"entity_ids": entity_ids}))
            all_links += links
            layers = {link["layer"] for link in links}
            counts["records"] += 1
            counts["with layer 1"] += 1 in layers
            counts["with layer 2"] += 2 in layers
            counts["with layer 3"] += 3 in layers
            counts["linked to India only"] += entity_ids == [INDIA]
            for link in links:
                counts[f"links layer {link['layer']}"] += 1
                counts[f"ambiguous layer {link['layer']}"] += link["confidence"] < 1
        with gzip.open(STAGING_DIR / f"{stem}_linked.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in out:
                f.write(r.model_dump_json() + "\n")
        report[stem] = dict(counts)

    with gzip.open(LINKS, "wt", encoding="utf-8") as f:
        for link in all_links:
            f.write(json.dumps(link, ensure_ascii=False) + "\n")

    # The author's check: 200 links, split evenly across the three layers where possible.
    rng = random.Random(RANDOM_SEED)
    by_layer = defaultdict(list)
    for link in all_links:
        by_layer[link["layer"]].append(link)
    sample = []
    for layer in (1, 2, 3):
        pool = by_layer[layer]
        sample += rng.sample(pool, min(SAMPLE_SIZE // 3 + (layer == 1), len(pool)))
    titles = {}
    for stem in INPUTS:
        p = STAGING_DIR / f"{stem}.jsonl.gz"
        if p.exists():
            for line in gzip.open(p, "rt", encoding="utf-8"):
                r = json.loads(line)
                titles[r["id"]] = (r["title"], r["url"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "link_check_sample_v1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["doc", "title", "url", "layer", "field", "matched_name", "qid", "entity_label", "confidence", "correct"]
        )
        for s in sample:
            t, u = titles.get(s["doc"], ("", ""))
            w.writerow([s["doc"], t, u, s["layer"], s["field"], s["name"], s["qid"], gaz.label.get(s["qid"], ""), s["confidence"], ""])  # fmt: skip

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    summary = {
        "date": datetime.now(UTC).date().isoformat(),
        "commit": commit,
        "by_file": report,
        "links": len(all_links),
        "links_by_layer_and_type": dict(
            Counter(
                f"layer {link['layer']}: {gaz.type.get(link['qid'], 'not in gazetteer (museum link)')}"
                for link in all_links
            ).most_common()
        ),
        "check_sample": len(sample),
    }
    (REPORT_DIR / "object_links_v1.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
