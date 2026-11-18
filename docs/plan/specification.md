# HeritageRec: system specification

Nov 18, 2026 · @azzz

HeritageRec takes a question about Indian heritage in English, Hindi or Kannada and returns ranked documents of six kinds (sites, objects, manuscripts and texts, inscriptions, cultural practices, people and dynasties), recommends related documents and entities, and explains every result through the knowledge graph. The research claim is tested on a public 200-query benchmark: does graph evidence improve retrieval over strong text baselines, and for which queries?

## 1. Research questions and hypotheses

Every result is reported whichever way it goes; the paper title follows the result.

| # | Question | Hypothesis (what we expect, not what we promise) |
| --- | --- | --- |
| RQ1 | Does graph evidence (Personalized PageRank, SimRank, meta-paths) improve ranking over BM25 and multilingual dense retrieval? | Graph helps most on entity-focused and complex queries; BM25 stays strong on simple keyword queries |
| RQ2 | Does combining graph and text signals beat each alone, and which signals matter? | Fusion or LambdaMART is best overall; ablation and SHAP show which signals carry it |
| RQ3 | Do graph signals help Hindi and Kannada queries find English documents? | Yes: Wikidata IDs are language-independent, so a linked entity bridges languages even when text overlap is zero |
| RQ4 | Are graph-path explanations faithful to the ranking? | Removing the cited path lowers the document's rank for most explained results |

## 2. Document types

Six types, so results cover the whole of heritage rather than only monuments. Every document enters through a Wikidata entity and carries a `doc_type`.

| Type | What it covers | Example | Main sources | Typical length |
| --- | --- | --- | --- | --- |
| Site | Monuments, temples, forts, caves, archaeological sites, historic towns | Hampi; Brihadisvara Temple | Wikipedia (en, hi, kn) | Long |
| Object | Sculptures, paintings, bronzes, coins, textiles, arms | A Chola Nataraja bronze; a Mughal album folio | Met, Cleveland open access | Short |
| Manuscript and text | Manuscripts, historical reports, gazetteers, classical texts | An ASI annual report; a Sanskrit medical manuscript | Wellcome, Internet Archive (public domain), Wikisource | Short to very long |
| Inscription | Stone and copper-plate inscriptions | A Hoysala grant inscription | SIDDHAM (optional) | Short |
| Culture | Living and intangible heritage: performing arts, festivals, crafts, painting schools, culinary and textile traditions | Yakshagana; Kalamkari; Tanjore painting | Wikipedia | Medium |
| Person, dynasty, style | Rulers, patrons, architects, dynasties, architectural and art styles | Shah Jahan; Hoysala architecture | Wikipedia | Medium to long |

The culture type needs its own seed strata (Section 3). Document length differs a lot by type, so results are reported per document type as well as overall.

## 3. Corpus

Target: 1,500–3,000 documents from about 800–900 seed entities, Republic of India only (decision 0001).

| Seed stratum | Rule | Take |
| --- | --- | --- |
| World Heritage sites in India | Heritage designation | All (57 counted) |
| Tentative List sites | Heritage designation | All (61 counted) |
| Dynasties and historical states | Type dynasty with country India, plus any dynasty or state whose capital is in India (decision 0002) | All, listed for a hand scan |
| ASI monuments | ASI Monument ID, English article exists | About 500, stratified by state, minimum 5 per state |
| Cultural heritage | UNESCO Intangible Cultural Heritage elements of India, classical and folk performing arts, painting schools, crafts with geographical indication | All with an English article, capped at about 150 |
| Styles | Architectural styles and art schools used by the seeded sites and objects | All reached in one hop |

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

| Edge | From → to | Source |
| --- | --- | --- |
| about | Document → its main entity | Sitelink or museum record |
| mentions | Document → entity, with confidence | Entity linking on text |
| located in, part of | Site → place, site → complex | Wikidata |
| built or commissioned by, architect | Site → person | Wikidata |
| member of dynasty, capital | Person → dynasty, dynasty → place | Wikidata |
| style, genre | Site or object → style or art form | Wikidata |
| depicts, made of | Object → entity, object → material | Museum record, Getty AAT |
| in period | Entity or document → period | Derived |

Very common hubs (India, Hinduism, a whole century) are down-weighted by inverse frequency, so two documents are not "related" just because both are Indian. Relation weights are tuned on the 40 dev queries only. The graph is saved as a versioned edge list, not a pickle, and its statistics (nodes and edges by type, degrees, components) are generated into the data card.

## 5. Query understanding

1. Detect the language (English, Hindi, Kannada).
2. Link entity mentions to Wikidata IDs with a gazetteer built from Wikidata labels and aliases in all three languages, plus fuzzy matching. Measured on the dev queries.
3. Classify the query type (simple keyword, entity-focused, concept-focused, complex), measured on dev.
4. Extract period and region cues (dynasty names, centuries, states).

## 6. Retrieval and ranking

A four-stage pipeline; each stage is also evaluated alone so its contribution is visible.

1. **Candidates (top 100 from each):** BM25 and BM25+RM3 with language-aware analysers; dense retrieval with BGE-M3 and multilingual-e5; graph retrieval by Personalized PageRank from the query's linked entities.
2. **Fusion:** reciprocal rank fusion of the three lists (no tuning needed).
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
| Graph | PPR score, SimRank to top seed documents, meta-path counts (shares dynasty, site, style, place, tradition), entity overlap, Horn's Index of shared entities |
| Query match | Period match, region match, document type matches the query's intent, language match |

A final, optional diversity step (maximal marginal relevance across document types) makes the top 10 include more than one type where the query allows; type coverage is reported separately so it never hides relevance.

## 7. Recommendations

| Recommendation | Input → output | How it is scored | Evaluated? |
| --- | --- | --- | --- |
| More like this | Document → documents | Meta-path similarity, SimRank and dense similarity, fused | Yes: 40 seed documents, judged like queries |
| Across types | Site → its objects, texts and traditions | Graph paths through shared dynasty, site, style or tradition | Yes, as part of the 40 seed documents (type coverage) |
| Related entities | Entity → entities | Personalized PageRank on the entity graph | Demo only |
| Nearby | Location → sites and their documents | Distance plus relevance | Demo only |

Example: from the Hampi page, recommendations include Vijayanagara coins in the Met, an ASI report on Hampi, Vijayanagara architecture, and Krishnadevaraya, each with its explanation.

## 8. Explanations

- **Path explanation:** the top one or two weighted graph paths from the query's entities to the document, in words: "Shares Krishnadevaraya and Vijayanagara architecture with your query."
- **Signal explanation:** the top three SHAP contributions from LambdaMART: "Ranked high mainly for graph proximity and a period match."
- **Fidelity:** remove the cited path, re-rank, and count how often the document drops. **Coverage:** share of top-10 results that get a path explanation.

## 9. Evaluation and planned outputs

Benchmark: 200 queries (40 in Hindi or Kannada), LLM-judged with a fixed prompt, validated against the author's judgements on 15 queries (weighted kappa reported), plus 40 seed documents for recommendation. Primary metric NDCG@10; also P@5, MAP, Recall@100, type coverage@10, period and region match, latency. Paired randomization tests with Holm correction and 95% bootstrap intervals.

**Making the benchmark hard to dispute**

- **Queries from information needs, not from the entity list.** The LLM is given short descriptions of who is searching and why ("a student comparing Hoysala and Chalukya temple styles"), not the seed entities, so queries are not tilted toward the graph. About 40 queries are written in real question styles (exam, tourist, researcher). The share of queries containing a linkable entity is reported.
- **Validation set:** 15 test queries, including 3–4 in Hindi or Kannada, judged by the author blind to labels and systems. Agreement is reported overall and per language.
- **Judge bias check:** agreement between the LLM and the author is reported per system, to show the judge does not favour one kind of system (for example dense retrieval).
- **Second human (strongly recommended):** one more person judges the same 15 queries; human–human agreement is the ceiling the LLM is compared against.
- **Re-pooling:** documents surfaced by later systems (LambdaMART, reranker) are judged too; the number of unjudged documents in each top 10 is reported.
- **Frozen analysis plan:** before any test-set run, a dated file with hypotheses, primary metric (NDCG@10), tests, corrections and the query split is committed. Results are reported against it.

| Output | What it shows |
| --- | --- |
| Table: corpus | Documents by type, source, language, licence, state, period |
| Table: knowledge graph | Nodes and edges by type, degree, components |
| Table: main results | Every system on the 160 test queries, with intervals and significance |
| Table: by query type and language | Where graph helps and where it does not (RQ1, RQ3) |
| Table: ablation and features | Drop in NDCG@10 per removed signal; SHAP importance (RQ2) |
| Table: recommendations | More-like-this quality and type coverage on 40 seed documents |
| Table: explanations | Fidelity and coverage (RQ4) |
| Table: judge agreement | LLM versus author: weighted kappa and raw agreement |
| Figure: architecture | The four-stage pipeline |
| Figure: graph schema | Node and edge types |
| Figure: worked example | One query, its results, and their explanations |

What makes the work remarkable is the design, not a promised number: six document types in one graph, cross-lingual retrieval through language-independent entities, a graph kept separate from text so the comparison is clean, explanations whose faithfulness is measured, and a public benchmark with a DOI.

## 10. Build order

This follows the fast-track roadmap in the HeritageRec 11/10 plan tab.

1. Corpus and data card (seeds, collectors, schema, snapshot).
2. Knowledge graph and gazetteer (from Wikidata, version-safe format).
3. Query understanding (linking, type, cues), measured on dev.
4. Retrieval stages and fusion; reranker; LambdaMART.
5. Recommendations and explanations.
6. Benchmark (queries, pools, LLM judge, human validation).
7. Full grid, statistics, release, results sheet.

**Must ship versus stretch.** The two-week fast track ships the left column; the right column follows if time allows, or goes into the IJDL version.

| Must ship | Stretch |
| --- | --- |
| Corpus, data card, knowledge graph | Inscription documents (SIDDHAM), Wikisource |
| BM25, BM25+RM3, dense, graph retrieval, both controls, fusion, reranker, LambdaMART | RotatE graph embeddings, 4B reranker |
| Benchmark with validation, judge-bias check, frozen analysis plan | Second human annotator (strongly recommended) |
| RQ1–RQ3 results with statistics and ablation | Recommendation evaluation on 40 seed documents, explanation fidelity (RQ4), diversity step |
