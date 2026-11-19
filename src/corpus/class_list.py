"""Indian heritage buildings and sites by Wikidata class, as a fourth entity list (decision 0018).

Reads the expansion probe's cached class queries (so the list matches the probe's
counts exactly), keeps items with a Wikipedia article in English, Hindi or Kannada
that are not already in another entity list or removed by a hand scan, and adds
their English labels.

Usage:
    uv run python -m src.corpus.class_list

Output:
    data/seeds/class_list_v1.csv    entity list in the seed-list format, read by the collectors
"""

import csv
import random

from src.corpus.expansion_probe import CLASSES, RAW_DIR, class_items
from src.corpus.seeds import PEOPLE_SCAN_FILE, STYLE_SCAN_FILE, batches, scan_removed
from src.corpus.wikimedia import ROOT, Wikimedia, qid, values

LISTS = ["seeds_v1.csv", "asi_remainder_v1.csv", "neighbours_v1.csv"]
OUT_CSV = ROOT / "data/seeds/class_list_v1.csv"
SAMPLE = ROOT / "results/probes/class_list_sample_v1.md"
RANDOM_SEED = 20261119


def main() -> None:
    wm = Wikimedia(RAW_DIR)
    listed = scan_removed(PEOPLE_SCAN_FILE) | scan_removed(STYLE_SCAN_FILE)
    for name in LISTS:
        listed |= {r["qid"] for r in csv.DictReader(open(ROOT / "data/seeds" / name, encoding="utf-8"))}

    first_class: dict[str, str] = {}
    langs: dict[str, set] = {}
    for c, label in CLASSES.items():
        for q, ls in class_items(wm, c).items():
            if ls and q not in listed:
                first_class.setdefault(q, label)
                langs.setdefault(q, set()).update(ls)

    labels: dict[str, str] = {}
    for n, chunk in batches(sorted(first_class)):
        for r in wm.sparql(
            f"class_labels_{n:03d}",
            f"""SELECT ?i ?iLabel WHERE {{ VALUES ?i {{ {values(chunk)} }}
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,hi,kn". }} }}""",
        ):
            labels[qid(r["i"])] = r.get("iLabel", "")

    rows = sorted(first_class)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "label", "stratum", "sub_stratum", "doc_type", "status", "flags"])
        for q in rows:
            w.writerow(
                [q, labels.get(q, ""), "class_list", first_class[q], "site", "selected", " ".join(sorted(langs[q]))]
            )

    sample = random.Random(RANDOM_SEED).sample(rows, min(50, len(rows)))
    SAMPLE.write_text(
        "\n".join(
            [
                "# Class-based list, scope sample (v1)",
                "",
                f"{len(rows)} entities in `data/seeds/class_list_v1.csv`; 50 drawn with seed {RANDOM_SEED} for the"
                " author's hand check.",
                "",
                *[f"- {labels.get(q, '')} ({q}, {first_class[q]})" for q in sample],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"{len(rows)} entities; wrote {OUT_CSV.name} and {SAMPLE.name}")


if __name__ == "__main__":
    main()
