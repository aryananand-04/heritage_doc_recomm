"""Cleveland collector by place: every CC0 object whose culture names India (decision 0017).

Pages through the open-access search for culture "India" and "South Asia" (CC0
only), 1,000 records a page; keeps a record when it is CC0, its culture names India
and it has text. Text and record follow `cleveland_collector.to_record`. Entity links
at collection are layer 1 only: the object's own Wikidata ID when Cleveland gives
one, otherwise India (Q668) until the linking step.

Usage:
    uv run python -m src.corpus.cleveland_place_collector

Outputs:
    data/corpus_v1/staging/cleveland_place_v1.jsonl.gz
    results/collection/cleveland_place_v1.json
"""

import gzip
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime

from src.corpus import matching
from src.corpus.cleveland_collector import RAW_DIR, cultures, object_text, own_qids, to_record
from src.corpus.collection_probe import CLEVELAND
from src.corpus.wikimedia import ROOT, Wikimedia

STAGING_DIR = ROOT / "data/corpus_v1/staging"
REPORT_DIR = ROOT / "results/collection"
INDIA = "Q668"
PAGE = 1000
CULTURES = ("India", "South Asia")


def main() -> None:
    wm = Wikimedia(RAW_DIR)
    found, outcomes, totals, layer1 = {}, Counter(), {}, 0
    for culture in CULTURES:
        skip, total = 0, None
        while total is None or skip < total:
            name = f"cleveland_place_{culture.replace(' ', '_')}_{skip}"
            params = {"culture": culture, "cc0": "1", "limit": PAGE, "skip": skip}
            data = wm.get_json(name, CLEVELAND, params)
            total = (data.get("info") or {}).get("total") or 0
            path = wm.raw_path(name, {"url": CLEVELAND, **params})
            fetched_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            rows = data.get("data") or []
            for obj in rows:
                if obj["id"] in found:
                    outcomes["already found under the other culture"] += 1
                elif obj.get("share_license_status") != "CC0":
                    outcomes["not CC0"] += 1
                elif not matching.names_india(cultures(obj)):
                    outcomes["culture does not name India"] += 1
                elif not object_text(obj):
                    outcomes["no text"] += 1
                else:
                    links = own_qids(obj)
                    layer1 += bool(links)
                    raw_ref = f"{path.relative_to(ROOT)}#id={obj['id']}"
                    found[obj["id"]] = to_record(obj, links or [INDIA], raw_ref, fetched_at)
                    outcomes["kept"] += 1
            if not rows:
                break
            skip += PAGE
        totals[culture] = total

    records = [found[k] for k in sorted(found)]
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(STAGING_DIR / "cleveland_place_v1.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "date": datetime.now(UTC).date().isoformat(),
        "commit": commit,
        "search_totals": totals,
        "outcomes": dict(outcomes.most_common()),
        "records": len(records),
        "records_with_layer1_links": layer1,
        "requests": {"fetched": wm.fetched, "reused": wm.reused},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cleveland_place_v1.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8"
    )
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
