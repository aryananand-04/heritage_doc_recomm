"""The ASI monuments not drawn into the seed sample, as a further entity list (decision 0015).

Decision 0002 samples about 500 ASI monuments by state for the seed list. The
neighbour probe showed that one hop adds fewer heritage entities than the corpus
target assumed, so every other ASI monument with an English article becomes a
document entity too. It is a separate list, not a larger seed quota, so the seed
sample and every later random draw in `seeds.py` stay unchanged.

Usage:
    uv run python -m src.corpus.asi_remainder

Output:
    data/seeds/asi_remainder_v1.csv    entity list in the seed-list format, read by the collectors
"""

import csv

from src.corpus.seeds import RAW_DIR, asi_monuments
from src.corpus.wikimedia import ROOT, Wikimedia

SEEDS = ROOT / "data/seeds/seeds_v1.csv"
OUT_CSV = ROOT / "data/seeds/asi_remainder_v1.csv"


def main() -> None:
    seeds = {r["qid"] for r in csv.DictReader(open(SEEDS, encoding="utf-8"))}
    # The same cached query the seed build used, so the list matches it exactly.
    asi = asi_monuments(Wikimedia(RAW_DIR))
    rest = sorted((q, label) for q, label in asi.items() if q not in seeds)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "label", "stratum", "sub_stratum", "doc_type", "status", "flags"])
        for q, label in rest:
            w.writerow([q, label, "asi_remainder", "", "site", "selected", "ASI monument not in the seed sample"])
    print(f"{len(asi)} ASI monuments with an English article; {len(rest)} not in the seed list; wrote {OUT_CSV.name}")


if __name__ == "__main__":
    main()
