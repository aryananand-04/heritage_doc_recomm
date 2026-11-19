# Data notes: what was found in the data and what was changed

This file records every hand decision and every rule added because of something
found in the data, from the audit of the old corpus to the corpus v1 build. Each
entry says what was found, an example, what changed, where the change is recorded,
and who decided it. Counts come from the files named; nothing here replaces them.

"Author" means the decision was made by the author. "During collection, pending
author review" means the change was made while building the corpus and is marked
in its file for the author to confirm or reverse.

Last updated: 19 November 2026.

## 1. Audit of the old corpus

| Found | Example | Change | Recorded in | Decided by |
| --- | --- | --- | --- | --- |
| The old live corpus had 303 documents, many not Indian | Euphrates (Turkey, Syria, Iraq) | Rebuilt from scratch; the old files stay in `legacy/` | `results/audit/cleaning_report_2026-11-18.md` | Author |
| 178 documents had a Wikidata country other than India | Colosseum, Versailles, Persepolis | Dropped | `results/audit/indian_check_2026-11-18.csv` | Rule, checked by author |
| 42 documents had no clear country on Wikidata, so no rule could decide | Julius Caesar and Ashoka both have no country | Each judged by hand: 27 dropped, 15 kept | same CSV, `author_checked = yes` | Author |
| A historical figure's tie to India runs through a historical polity, not a modern country | Tipu Sultan: Kingdom of Mysore; Ashoka: Maurya Empire | Kept, with the polity as the reason | same CSV, `note` column | Author |
| A source URL pointed to a disambiguation page | Palace of the Popes | Dropped | same CSV | Author |
| 7 Indian documents were duplicate copies of another document | Pattadakal, Brihadisvara Temple, Sanchi Stupa (twice), Golconda, Nalanda Mahavihara, Hampi | One copy kept per duplicate group | same CSV, `duplicate_group` | Rule |

## 2. Choosing the seed entities

| Found | Example | Change | Recorded in | Decided by |
| --- | --- | --- | --- | --- |
| The obvious dynasty rule ("type dynasty, country India") misses most major polities | Chola, Mughal, Gupta, Hoysala, Rashtrakuta, Vijayanagara are typed "historical country"; several have no country | Dynasties also taken through their capital (P36) in India | `docs/decisions/0002-seed-strata-and-sampling.md` | Author |
| Colonial administrative units came in as dynasties | British Raj, Company rule in India, Dominion of India, Ajmer-Merwara, Eastern States Union | 5 removed | `data/seeds/curated/dynasty_scan.csv` | Author |
| Wikidata covers culture thinly and inconsistently | Dance forms recorded as subclasses; intangible heritage lists mixed with Malaysian, Dutch and Iraqi lists; crafts with a geographical indication not recorded | Culture strata built from Wikipedia lists, every item checked on Wikidata | decision 0002 | Author |
| Culture candidates included parts and terms rather than forms | Alarippu (a piece within Bharatanatyam), Melattur style, Arohana and Janya raga (theory terms), Temple dance (a general concept), a person | 33 of 110 removed | `data/seeds/curated/culture_scan.csv` | Author |
| Legendary figures passed the people date rule | Janamejaya, Alli Raani | Removed | `data/seeds/curated/people_scan.csv` | Author |
| Six people dated 1850 or later are tied to seeded sites | Le Corbusier, George Wittet, Axel Haig, F. W. Stevens, Aga Khan III, Nripendra Narayan of Cooch Behar | Kept | same file | Author |
| Non-Indian people passed every automatic rule through one link to a seeded site | Johann Wolfgang von Goethe; Catherine of Valois (born 1401) | Removed | same file | During collection, pending author review |
| Global styles came in through real style statements on Indian buildings, but their articles are about other countries | Gothic, baroque, Victorian, brutalist, Manueline, Moroccan, Seljuk architecture | 15 removed as document seeds; they stay graph nodes | `data/seeds/curated/style_scan.csv` | During collection, pending author review |
| Some styles are global but centred on India | Buddhist architecture, Greco-Buddhist art, rock-cut architecture | Kept | same file | During collection, pending author review |
| One seed has no English label | `Q11934612` | Not changed yet; to trace | — | Open |
| Seed sampling depended on Python's set order, so the seed list changed between runs | — | Sampling made order-independent; the list is byte-identical across runs | `src/corpus/seeds.py`, `tests/test_seeds.py` | Fix |

After the people and style scans, the seeds went from 1,215 to 1,196.

## 3. Growing the entity lists

| Found | Example | Change | Recorded in | Decided by |
| --- | --- | --- | --- | --- |
| One link from the seeds reaches 4,329 entities, 2,000 in scope with an article, but in a sample of 50 only about 20 were heritage | Districts and cities through "located in"; modern musicians through "genre"; universities; a wildlife sanctuary; a book | Neighbours kept only under heritage classes, people by the date rule, institutions excluded; places stay graph nodes: 568 kept | decision 0015, `src/corpus/neighbour_select.py` | Author (rule), checked on samples |
| Heritage classes still let modern things through | Jaipur International Airport (a built structure), Jaipur International Film Festival (a festival) | Airports, film and literary festivals, films, list articles excluded | `src/corpus/neighbour_select.py` | During collection |
| Illuminated manuscripts were missed | Victoria and Albert Akbarnama | Manuscript, painting, sculpture and inscription classes added | same file | During collection |
| Some heritage works have no date on Wikidata | Tutinama, Tarikh-i Alfi | Left out by the "before 1900" rule for literary works | same file | Rule |
| A removed seed came back as a neighbour | Catherine of Valois | Seed hand-scan removals applied to all lists | same file | During collection |
| 109 neighbours were also ASI monuments | — | Kept once, in the ASI list | same file | During collection |
| The seed ASI sample (about 500) left out 469 ASI monuments with English articles | — | Added as their own list, so the seed sample and its random draws stay unchanged | `src/corpus/asi_remainder.py`, decision 0015 | During collection |
| Most Indian temples, forts, mosques and palaces with articles were in no list | 3,516 such entities (2,357 of the 4,569 with articles are Hindu temples) | Added as a class-based list | decision 0018, `src/corpus/class_list.py` | Author |

## 4. Wikipedia text

| Found | Example | Change | Recorded in | Decided by |
| --- | --- | --- | --- | --- |
| Reference and link sections survive text extraction as headings with little text | "See also", "References", "External links" | Cut from the first such heading to the end | `src/corpus/wikipedia_collector.py` | Rule |
| The Hindi and Kannada versions of those headings vary a lot | ಇದನ್ನೂ ನೋಡಿ, ಸಹ ನೋಡಿ, ಬಾಹ್ಯ ಸಂಪರ್ಕಗಳು, ಮತ್ತಷ್ಟು ಓದುವಿಕೆ; यह सभी देखें, सन्दर्भ | Lists extended from the most common last headings in the collected articles, which every run reports | same file; `most_common_last_headings` in `results/collection/` | Rule |
| A few seeds lead to disambiguation, empty or missing pages, or two seeds to one page | 2 English disambiguation pages, 3 shared pages | Skipped and counted | `results/collection/wikipedia_seeds_v1.json` | Rule |
| Some articles are stubs | Records of 21 characters | Not changed yet; a minimum length belongs to the cleaning step | — | Open |

## 5. Museum and library records found by seed name

| Found | Example | Change | Recorded in | Decided by |
| --- | --- | --- | --- | --- |
| Collection searches match the name anywhere | "Sun Temple" returns Baalbek and Rome; "Valley of Flowers National Park" returns Yosemite | Kept only when every word of the seed's name is in the record's identifying fields and the record's place names India | `src/corpus/matching.py` | Rule |
| Exact-phrase matching missed real objects | Chamba rumals catalogued as "Cover (Rumal)" with culture "India (Himachal Pradesh, Chamba)" | Words may appear in any order | same file | Rule |
| Long descriptions contain common words by chance | Cleveland: a Surya painting and a Navagraha frieze matched "Sun Temple" | Words must be in the short fields, or the exact name in the description | `src/corpus/cleveland_collector.py` | Rule |
| A dynasty's short name is often something else | "Nanda" matched paintings of Nanda, Krishna's foster-father; "Matsya" the fish avatar of Vishnu; Shunga-period railings for "Kadamba" and "Mitra" | A dynasty name must sit next to a polity word ("Nanda dynasty", "period of the Nandas", "Raja of Marwar", "Mughal India") | `src/corpus/matching.py` | Rule |
| A title directly before a name is usually a person | "Prince Sur Singh Rathor of Bikaner" matched the Sur Empire | Titles count only with "of" before the name | same file | Rule |
| Two dynasties share one name | Pala Empire (Bengal) and Pala dynasty (Kamarupa) | A name's bracketed qualifier must appear in the record | same file | Rule |
| The Met's search ignores its public-domain filter | 404 of about 1,000 objects looked at were not public domain | Each object's own flag is checked | `src/corpus/met_collector.py` | Rule |
| Wellcome's search without quotes matches any word | "Silk Road sites in India" returned a German medical book | Phrase search | `src/corpus/wellcome_collector.py` | Rule |
| Wellcome's Indian manuscripts are catalogued under Sanskrit titles, not seed names | A phrase search for "Mughal" returns one manuscript | Accepted: Wellcome gives pictures only (55) | `results/collection/wellcome_seeds_v1.json` | Accepted limit |
| Wellcome's digitised items often carry CC BY-NC | — | Only Wellcome's own catalogue text (CC0) is used, never the item | decision 0016 context, `src/corpus/wellcome_collector.py` | Rule |
| Short or common seed names match unrelated records | "Sua" matches Italian ("sua"); "Lotus Temple" (Delhi) matched the Golden Lotus tank in Madurai; "Khyal" a modern poem; "Gopala" an editor named Gopala Bhatta; "Francis Xavier" a nun named after him | Not fully fixable by rules; reported in the data card | `docs/data_card.md` | Accepted limit |

## 6. Internet Archive texts

| Found | Example | Change | Recorded in | Decided by |
| --- | --- | --- | --- | --- |
| No stated licence for uploader-written descriptions | — | The document is the public-domain OCR text, never the description | decision 0016 | During collection |
| Language metadata is often wrong | An English edition tagged Hindi; Urdu books tagged Hindi; parliament records tagged "English, Hindi" | The text's script decides the language; metadata only separates Hindi from other Devanagari languages | `src/corpus/internet_archive_collector.py` | Rule |
| Real Indian texts never mention "India" in their metadata | Matta Vilasa Prahasana (tagged Hindi), a Dictionary of Hindustani Classical Music (tagged Sanskrit) | An Indian-language tag also counts as pointing to India | same file | Rule |
| A surname matched a dynasty | Urdu books by an author named Varman for "Varman dynasty" | Dropped by the script rule (Arabic script); the dynasty rule applies too | same file | Rule |
| Modern government records are marked public domain by uploaders | Lok Sabha debates such as "Widening of Grand Trunk Road (2009)" | Items dated 1950 or later left out | same file | Rule |
| Many items have no date field; their titles often carry a modern year | "Rajput Painting by Basil Gray … 1948", "… Rowland 1953", "… Randhawa 1963" | The year is taken from the title when the date is missing; items with no year anywhere are left out (90), as their public-domain mark cannot be checked | same file | Rule |
| Sanskrit in Devanagari can be tagged Hindi | Yoga Vasishta (Sanskrit commentary) | Not fixable by script; reported | `docs/data_card.md` | Accepted limit |

## 7. Museum objects collected by place, and their entity links

| Found | Example | Change | Recorded in | Decided by |
| --- | --- | --- | --- | --- |
| Name searches found few objects (Met 42, Cleveland 124) | — | Objects collected by place (India), then linked to entities | decision 0017 | Author |
| The Met's search returns at most 500 IDs a page | 4,367 found, 100 returned by default | Paged with an offset | `src/corpus/met_place_collector.py` | Fix |
| The Art Institute of Chicago licenses its description field differently | Its API states description CC BY 4.0, all other fields CC0 | Records using a description carry CC BY 4.0 (28 of 347) | `src/corpus/chicago_collector.py` | Rule |
| Museum fields use short forms of names | "Mughal India", "Chola period", "Kotah" | Short forms derived from labels ("Mughal Empire" → "Mughal"); aliases used ("Mughal India" is an alias of Mughal Empire, "Banaras" of Varanasi) | `src/corpus/link_objects.py`, gazetteer | Rule |
| "Delhi Sultanate" shortened would become the city of Delhi | — | "Sultanate" is never removed | same file | Rule |
| Short forms of district names are ordinary words | "South district" → "South"; "Krishna district" turned the deity Krishna into a district; "Slave dynasty" → "slave" | Common-word short forms dropped; free text uses full names only | same file | Rule |
| A city and its district share a name | Lucknow and Lucknow district | A full name wins over a short form made from another entity's name | same file | Rule |
| Two different places share a name | Patan (two towns), Bijapur | Kept as candidates in the links file with confidence 1/n, not in the record's entity links | same file | Rule |
| A qualified entity also has unqualified aliases | Pala dynasty (Kamarupa) also has the alias "Pala dynasty" | The qualifier applies to all of an entity's names | same file | Rule |
| Most Chicago records give only "India" as their place | 239 of 347 linked to India only | Accepted; reported | `results/linking/object_links_v1.json` | Accepted limit |

Link precision per layer is measured on the author's check of
`results/linking/link_check_sample_v1.csv` (200 links).

## 8. Checks waiting for the author

- Scope of the neighbour sample: `results/probes/neighbour_select_v1.md` (50 entities).
- Scope of the class-based list sample: `results/probes/class_list_sample_v1.md` (50 entities).
- Link check: `results/linking/link_check_sample_v1.csv` (200 links, `correct` column).
- Hand-scan rows marked "pending author review" in `data/seeds/curated/people_scan.csv` and `style_scan.csv`.
