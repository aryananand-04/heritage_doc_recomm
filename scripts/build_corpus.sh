#!/usr/bin/env bash
# Build corpus v1 from scratch, or rebuild it from the saved raw responses.
#
# Every step caches each API response under data/raw/ before using it, so a second run
# makes no network requests and gives a byte-identical snapshot. A first run needs
# CONTACT_EMAIL in .env (sent in the User-Agent, as Wikimedia asks) and takes several hours.
#
# Usage: bash scripts/build_corpus.sh
set -euo pipefail
cd "$(dirname "$0")/.."

run() { echo "== $*"; uv run python -m "$@"; }

# 1. Entity lists (decisions 0002, 0015, 0018)
run src.corpus.seeds
run src.corpus.neighbour_probe
run src.corpus.asi_remainder
run src.corpus.neighbour_select
run src.corpus.expansion_probe
run src.corpus.class_list

# 2. Wikipedia, one article per entity and language (decision 0003)
for list in seeds_v1 asi_remainder_v1 neighbours_v1 class_list_v1; do
  run src.corpus.wikipedia_collector --entities "data/seeds/$list.csv"
done

# 3. Museums and libraries by seed name (decisions 0013, 0015, 0016)
run src.corpus.met_collector
run src.corpus.cleveland_collector
run src.corpus.wellcome_collector
run src.corpus.internet_archive_collector

# 4. Museum objects by place, and reference series (decisions 0016, 0017)
run src.corpus.met_place_collector
run src.corpus.cleveland_place_collector
run src.corpus.chicago_collector
run src.corpus.ia_series_collector

# 5. Gazetteer and object linking (decision 0017)
run src.corpus.gazetteer
run src.corpus.link_objects

# 6. Snapshot, manifest and data card (decision 0004)
run src.corpus.build_snapshot
run src.corpus.data_card
