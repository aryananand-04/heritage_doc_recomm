# HeritageRec: system specification

Nov 19, 2026 · @azzz

HeritageRec takes a question about Indian heritage in English, Hindi or Kannada and returns ranked documents of six kinds (sites, objects, manuscripts and texts, inscriptions, cultural practices, people and dynasties), recommends related documents and entities, and explains every result through the knowledge graph. The research claim is tested on a public 200-query benchmark: does graph evidence improve retrieval over strong text baselines, and for which queries?

## 1. Research questions and hypotheses

Every result is reported whichever way it goes; the paper title follows the result.

| # | Question | Hypothesis (what we expect, not what we promise) |
| --- | --- | --- |
| RQ1 | Does graph evidence (Personalized PageRank, SimRank, meta-paths) improve ranking over BM25 and multilingual dense retrieval? | Graph helps most on entity-focused and complex queries; BM25 stays strong on simple keyword queries |
| RQ2 | Does combining graph and text signals beat each alone, and which signals matter? | Fusion or LambdaMART is best overall; ablation and SHAP show which signals carry it |
| RQ3 | Do graph signals help Hindi and Kannada queries find English documents? | Yes: Wikidata IDs are language-independent, so a linked entity bridges languages even when text overlap is zero |
| RQ4 | Are graph-path explanations faithful to the ranking? | Removing the cited path lowers the document's rank for most explained results |

The headline RQ1 result uses the Wikidata edge layer only (Section 4). The other edge layers are reported as additions on top of it.

## 2. Document types

Six types, so results cover the whole of heritage rather than only monuments. Every document enters through a Wikidata entity and carries a `doc_type`.

| Type | What it covers | Example | Main sources | Typical length |
| --- | --- | --- | --- | --- |
| Site | Monuments, temples, forts, caves, archaeological sites, historic towns | Hampi; Brihadisvara Temple | Wikipedia (en, hi, kn) | Long |
| Object | Sculptures, paintings, bronzes, coins, textiles, arms | A Chola Nataraja bronze; a Mughal album folio | Met, Cleveland open access | Short |
| Manuscript and text | Manuscripts, historical reports, gazetteers, classical texts | An ASI annual report; a Sanskrit medical manuscript | Wellcome, Internet Archive (public domain); Wikisource (proposed) | Short to very long |
| Inscription | Stone and copper-plate inscriptions | A Hoysala grant inscription | SIDDHAM (proposed; licence checked per record) | Short |
| Culture | Living and intangible heritage: performing arts, festivals, crafts, painting schools, culinary and textile traditions | Yakshagana; Kalamkari; Tanjore painting | Wikipedia | Medium |
| Person, dynasty, style | Rulers, patrons, architects, dynasties, architectural and art styles | Shah Jahan; Hoysala architecture | Wikipedia | Medium to long |

Document length differs a lot by type, so results are reported per document type as well as overall, and the data card reports the document-type mix of the corpus.

## 3. Corpus

Target: about 10,000 documents, Republic of India only (decision 0001), from the 1,196 seed entities (decision 0002, after the people and style hand scans) plus the entities one hop away from them along selected Wikidata relations (decision 0005). The probes of 19 November 2026 (decision 0015) set three entity lists: the 1,196 seeds, 469 further ASI monuments and 568 heritage neighbours. The corpus size is the number the collectors produce from them, likely below 10,000, and is never padded with out-of-scope documents.

| Seed stratum | Rule | Take |
| --- | --- | --- |
| World Heritage sites in India | Heritage designation | All |
| Tentative List sites | Heritage designation | All |
| Dynasties and historical states | Type dynasty with country India, plus any dynasty or state whose capital is in India (decision 0002) | All, listed for a hand scan |
| ASI monuments | ASI Monument ID, English article exists | About 500, stratified by state, minimum 5 per state |
| Cultural heritage | UNESCO Intangible Cultural Heritage elements of India, classical and folk performing arts, painting schools, crafts with geographical indication | Quotas in decision 0002 |
| Styles | Architectural styles and art schools used by the seeded sites and objects | All reached in one hop |
| People | Rule set at checkpoint 1 (decision 0002) | All that pass the rule |

**One-hop expansion (decision 0005).** Entities linked to a seed by the relations in Section 4 (for example built by, located in, member of dynasty, style) are added when they pass the India scope rule and have a Wikipedia article in English, Hindi or Kannada. A random sample of added entities is checked by hand for scope drift, and the drift rate is reported.

**Sources and caps (decision 0013).** Wikipedia (en, hi, kn) for every seed and added entity; Met, Cleveland, Wellcome and Internet Archive per seed, under per-source caps so famous places cannot flood the corpus. Wikisource and SIDDHAM, with per-record licence checks, are proposed: they are collected only if the corpus gates leave time.

All IDs are verified on Wikidata before use; counts come from probes, never estimates. Wikipedia text comes through the TextExtracts API (decision 0003). Each record follows one schema (decision 0004): id, title, text, language, doc\_type, source, url, licence URI, entity IDs, coordinates, period start and end, fetch date, SHA-256, raw-response pointer.

## 4. Knowledge graph

The graph holds only structured facts and entity links, never text similarity. The conference version mixed embedding-similarity and cluster edges into the graph, which made "graph versus text" impossible to separate; this version keeps them apart so RQ1 is a clean test.

| Node type | Examples | Source |
| --- | --- | --- |
| Document | Each corpus record | Corpus |
| Site | Hampi, Badami caves | Wikidata |
| Person | Shah Jahan, Krishnadevaraya | Wikidata |
| Dynasty or polity | Chola, Vijayanagara Empire | Wikidata |
| Place | Belur, Hassan district, Karnataka | Wikidata |
| Style or art form | Dravidian architecture, Kalamkari | Wikidata |
| Tradition | Yakshagana, Kumbh Mela | Wikidata |
| Object type or material | Bronze, palm-leaf manuscript | Getty AAT |
| Period | 12th century; Vijayanagara period | Derived from dates |

**Edge layers (decision 0006).** Every edge carries its layer, and each layer can be switched on or off, so the ablation shows which kind of structure helps.

| Layer | Edges | Source | Role |
| --- | --- | --- | --- |
| Core | about (document → main entity); located in, part of; built or commissioned by, architect; member of dynasty, capital; style, genre; in period | Wikidata statements, sitelinks, derived dates | Headline RQ1 result |
| Hyperlinks | Article → article it links to | Wikipedia link tables | Addition on top of core |
| Getty AAT | depicts, made of, object type | Museum records, Getty AAT | Addition on top of core |
| Geography | Coordinates-based containment and proximity between sites and places | Wikidata coordinates and admin areas | Addition on top of core |
| Mentions | Document → entity, with confidence | Entity linking on text (Section 5) | Addition on top of core |
| LLM triples | Relations extracted from text by an LLM | LLM extraction | Labelled ablation only, never in the headline result |

**Extent and hubs (decision 0007).** The core graph is the seeds plus one hop. Two hops are measured by a probe only, to report how fast the graph grows, and are not used. Very common hubs (India, Hinduism, a whole century) are down-weighted by inverse frequency, so two documents are not "related" just because both are Indian. Personalized PageRank runs on the full graph; SimRank runs only over each query's candidate set or as an approximation, because exact SimRank does not scale to the full graph.

Relation weights are tuned on the 40 dev queries only. The graph is saved as a versioned edge list, not a pickle, and its statistics (nodes and edges by type and layer, degrees, components) are generated into the data card.

## 5. Query understanding

1. Detect the language (English, Hindi, Kannada).
2. Link entity mentions to Wikidata IDs with a gazetteer built from Wikidata labels and aliases in all three languages, plus fuzzy matching. Measured on the dev queries, and on 50 hand-labelled documents (precision, recall and F1) with 200 random links checked by hand.
3. Classify the query type (simple keyword, entity-focused, concept-focused, complex), measured on dev.
4. Extract period and region cues (dynasty names, centuries, states).

## 6. Retrieval and ranking

A four-stage pipeline (decision 0008); each stage is also evaluated alone so its contribution is visible.

1. **Candidates (top 100 from each):** BM25 and BM25+RM3 with language-aware analysers; dense retrieval with MiniLM (old baseline), BGE-M3 and multilingual-e5; graph retrieval by Personalized PageRank from the query's linked entities. RotatE graph embeddings as a learned structural baseline are proposed.
2. **Fusion:** reciprocal rank fusion of the lists (no tuning needed).
3. **Reranking:** a multilingual cross-encoder (bge-reranker-v2-m3) on the top 50.
4. **Learning to rank:** LightGBM LambdaMART, 5-fold cross-validation split by query, over the features below.

**Two controls that make the claims fair.** Each RQ needs an opponent that removes the easy explanation.

| Control | What it is | Which claim it protects |
| --- | --- | --- |
| Entity-expanded BM25 | BM25 with the query expanded by the linked entities' labels and aliases (all three languages) | RQ1: the graph wins because of structure, not just because it knows the entities |
| Translate-then-search | Hindi and Kannada queries translated to English with IndicTrans2 (AI4Bharat, free), then BM25 and dense retrieval | RQ3: the graph wins cross-lingually even against a system that translates |

| Feature group | Features |
| --- | --- |
| Text | BM25 score, dense score, cross-encoder score, document length |
| Graph | PPR score, SimRank to top seed documents, meta-path counts (shares dynasty, site, style, place, tradition), entity overlap, Horn's Index of shared entities (RotatE score, if RotatE is accepted) |
| Query match | Period match, region match, document type matches the query's intent, language match |

A final, optional diversity step (maximal marginal relevance across document types) makes the top 10 include more than one type where the query allows; type coverage is reported separately so it never hides relevance.

## 7. Recommendations

The recommender is a hybrid of meta-path similarity, Personalized PageRank and dense similarity, fused, with the same diversity re-rank (decision 0009). There is no click data, so graph neural recommenders that learn from user interactions are not used.

| Recommendation | Input → output | How it is scored | Evaluated? |
| --- | --- | --- | --- |
| More like this | Document → documents | Meta-path similarity, PPR and dense similarity, fused, then diversity re-rank | Proposed: 40 seed documents, judged like queries |
| Across types | Site → its objects, texts and traditions | Graph paths through shared dynasty, site, style or tradition | Proposed, as part of the 40 seed documents (type coverage) |
| Related entities | Entity → entities | Personalized PageRank on the entity graph | Demo only |
| Nearby | Location → sites and their documents | Distance plus relevance | Demo only |

Example: from the Hampi page, recommendations include Vijayanagara coins in the Met, an ASI report on Hampi, Vijayanagara architecture, and Krishnadevaraya, each with its explanation.

## 8. Explanations

- **Path explanation:** the top one or two weighted graph paths from the query's entities to the document, in words: "Shares Krishnadevaraya and Vijayanagara architecture with your query."
- **Signal explanation:** the top three SHAP contributions from LambdaMART: "Ranked high mainly for graph proximity and a period match."
- **Fidelity:** remove the cited path, re-rank, and count how often the document drops. **Coverage:** share of top-10 results that get a path explanation.
- **User study (proposed, decision 0014):** 12–20 participants judge the relevance of results with and without explanations, with informed consent and responses stored under participant codes. Not in the first release unless accepted.

## 9. Evaluation and planned outputs

Benchmark: 200 queries (40 in Hindi or Kannada), LLM-generated and author-curated, LLM-judged with a fixed prompt, validated against the author's judgements on 15 test queries (weighted kappa reported), plus 40 seed documents for recommendation. The method is disclosed in the paper and the data card. Primary metric NDCG@10; also P@5, MAP, Recall@100, type coverage@10, period and region match, latency. Paired randomization tests with Holm correction and 95% bootstrap intervals.

**Making the benchmark hard to dispute**

- **Queries from information needs, not from the entity list.** The LLM is given short descriptions of who is searching and why ("a student comparing Hoysala and Chalukya temple styles"), not the seed entities, so queries are not tilted toward the graph. About 40 queries are written in real question styles (exam, tourist, researcher). The share of queries containing a linkable entity is reported.
- **LLM judge:** one open-weight model chosen by an options comparison, not used inside any compared system, with every judgement saved with its rationale.
- **Validation set:** 15 test queries, including 3–4 in Hindi or Kannada, judged by the author blind to labels and systems. Agreement is reported overall and per language.
- **Judge bias check:** agreement between the LLM and the author is reported per system, to show the judge does not favour one kind of system (for example dense retrieval).
- **Second human (strongly recommended):** one more person judges the same 15 queries; human–human agreement is the ceiling the LLM is compared against.
- **Proposed, not accepted (decision 0010):** a second LLM judge, and about 50 validation queries judged by two people.
- **Re-pooling:** documents surfaced by later systems (LambdaMART, reranker) are judged too; the number of unjudged documents in each top 10 is reported.
- **Frozen analysis plan:** before any test-set run, a dated file with hypotheses, primary metric (NDCG@10), tests, corrections and the query split is committed. Results are reported against it.

| Output | What it shows |
| --- | --- |
| Table: corpus | Documents by type, source, language, licence, state, period |
| Table: knowledge graph | Nodes and edges by type and layer, degree, components |
| Table: main results | Every system on the 160 test queries, with intervals and significance |
| Table: by query type and language | Where graph helps and where it does not (RQ1, RQ3) |
| Table: edge layers | Core graph versus each added layer (RQ1) |
| Table: ablation and features | Drop in NDCG@10 per removed signal; SHAP importance (RQ2) |
| Table: recommendations | More-like-this quality and type coverage on 40 seed documents |
| Table: explanations | Fidelity and coverage (RQ4) |
| Table: judge agreement | LLM versus author (and versus the second person, if any): weighted kappa and raw agreement |
| Figure: architecture | The four-stage pipeline |
| Figure: graph schema | Node and edge types |
| Figure: worked example | One query, its results, and their explanations |

**Release (decision 0011).** Code, corpus snapshot, queries, qrels and run files archived on Zenodo with a DOI; the benchmark also on Hugging Face Datasets, with a loader that returns ranx-compatible qrels and runs. A Hugging Face Spaces demo with a graph explorer is proposed (decision 0012).

What makes the work remarkable is the design, not a promised number: six document types in one graph, cross-lingual retrieval through language-independent entities, a graph kept separate from text so the comparison is clean, explanations whose faithfulness is measured, and a public benchmark with a DOI.

## 10. Roadmap

Two phases. The corpus and graph are built first, in weekly gates until mid-January 2027; evaluation and release follow, with a lighter scope. The interim checkpoint on 30 November 2026 is unchanged. A block does not start until the previous gate is passed.

**Phase 1: corpus and graph (weekly gates to mid-January)**

| Week | Dates | Work | Gate |
| --- | --- | --- | --- |
| 1 | 19–22 Nov | Per-source caps and borderline people settled (decision 0013); one-hop neighbour probe (0005); Wikipedia collector tried on a small sample | Caps accepted; neighbour counts on file |
| 2 | 23–30 Nov | Wikipedia en/hi/kn for the seeds; Met, Cleveland, Wellcome, Internet Archive first pass with caps; snapshot and data card; core Wikidata graph; all 200 queries generated, curated and split (test sealed); LLM judge chosen; BM25, BM25+RM3, MiniLM, BGE-M3, multilingual-e5 on dev; dev pools judged | **Interim checkpoint (30 Nov):** snapshot with checksum, data card, graph statistics, baseline numbers on the 40 dev queries, all labelled interim |
| 3 | 1–6 Dec | Wikipedia for the one-hop neighbours; hand check of a random sample for scope drift | Corpus near the ~10K target; drift rate on file |
| 4 | 7–13 Dec | Full museum and library runs (Met, Cleveland, Wellcome, Internet Archive) with licence filtering | Per-source fetched, kept, dropped and capped counts on file |
| 5 | 14–20 Dec | Cleaning: near-duplicates (MinHash), document types, periods, coordinates; schema validation | Cleaning report; every record passes validation |
| 6 | 21–27 Dec | Edge layers: core Wikidata, hyperlinks, Getty AAT, geography; hub down-weighting (decisions 0006, 0007) | Edge list with a layer column; statistics per layer |
| 7 | 28 Dec–3 Jan | Gazetteer and entity linking (mentions layer), measured on 50 hand-labelled documents and 200 checked links; two-hop probe; LLM-triples ablation layer | Entity-linking precision, recall and F1; two-hop counts on file |
| 8 | 4–10 Jan | Quality pass: full data card (document-type mix, licences, states, periods, unfilled places), entity and edge tables, spot checks by the author | Data card complete; author's spot checks recorded |
| 9 | 11–15 Jan | Freeze: snapshot v1 with manifest and checksums; graph v1; corpus tag | Corpus and graph frozen and tagged; one command rebuilds them from the raw responses |

**Phase 2: evaluation and release (after the corpus freeze, lighter scope)**

| Block | Work | Gate |
| --- | --- | --- |
| E1. Systems | All must-ship systems on the frozen corpus: BM25, BM25+RM3, MiniLM, BGE-M3, e5, PPR, SimRank, meta-paths, both controls, fusion, cross-encoder, LambdaMART | Every system runs on dev |
| E2. Benchmark | Analysis plan committed first; test pools; LLM judging; the author's 15-query validation (and a second person, if available); agreement report | Qrels in TREC format; the kappa; `docs/plan/analysis_plan.md` committed |
| E3. Results | Full grid on the 160 test queries; significance, intervals, per-type and per-language results; ablation; LambdaMART 5-fold and SHAP | Every result file in `results/paper_v1/`, each with its query count |
| E4. Release | Re-run from a fresh clone; tag v1.0; Zenodo DOI; Hugging Face dataset and loader (decision 0011); results sheet; preprint draft to co-authors | One command reproduces every number |

Phase 2 dates are set at the corpus freeze. The interim dev pools are judged again on the frozen corpus, because the corpus grows after 30 November.

**Must ship versus proposed**

| Must ship | Proposed (only if accepted and time allows) |
| --- | --- |
| Corpus of about 10K documents, data card, knowledge graph with separate edge layers | SIDDHAM and Wikisource collectors |
| BM25, BM25+RM3, dense, graph retrieval, both controls, fusion, reranker, LambdaMART | RotatE; the 4B reranker |
| Benchmark: LLM-generated, author-curated queries; LLM judge; the author's 15-query validation; judge-bias check; frozen analysis plan | A second LLM judge; about 50 validation queries judged by two people |
| RQ1–RQ3 results with statistics and ablation; release with DOI and loader | Recommendation evaluation on 40 seed documents, explanation fidelity (RQ4), diversity step; explanation user study; Spaces demo and graph explorer |

After the release: arXiv preprint once the co-authors agree, then the paper submissions. Never drop the human validation set, and never invent or estimate a number.
