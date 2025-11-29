# CLAUDE.md - AI Assistant Guide for Heritage Document Recommendation System

> **Last Updated:** 2025-11-29
> **Project:** Multi-Source Heritage Document Recommendation System using Knowledge Graphs, Autoencoders, and Graph-Based Ranking

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Development Workflow](#development-workflow)
4. [Key Conventions](#key-conventions)
5. [Technical Architecture](#technical-architecture)
6. [Data Pipeline](#data-pipeline)
7. [Dependencies & Environment](#dependencies--environment)
8. [Common Tasks](#common-tasks)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

### Purpose
This system collects heritage documents from multiple authoritative sources, constructs a rich Knowledge Graph, and provides intelligent recommendations based on semantic similarity and graph topology.

### Key Technologies
- **NLP & ML:** NLTK, scikit-learn, sentence-transformers, PyTorch
- **Knowledge Graph:** NetworkX, WordNet (Lesk similarity)
- **Deep Learning:** Autoencoder for document classification
- **Data Collection:** BeautifulSoup, requests, Wikipedia API
- **Visualization:** matplotlib

### Academic Context
- Final year project intended for conference submission
- Author: Akchhya Singh (akchhya1108@gmail.com)
- Status: 🚧 In Development | Version 1.0.0

---

## 📁 Codebase Structure

### Directory Layout

```
heritage_doc_recomm/
├── .git/                          # Git repository
├── .gitignore                     # Ignore patterns for data, models, venv
├── README.md                      # User-facing documentation
├── CLAUDE.md                      # This file - AI assistant guide
├── requirements.txt               # Python dependencies
│
├── src/                           # Source code (sequential pipeline)
│   ├── 1a_collect_wikipedia.py      # Wikipedia heritage scraper
│   ├── 1b_collect_unesco.py         # UNESCO World Heritage Sites scraper
│   ├── 1c_collect_indian_heritage.py # Indian monuments scraper
│   ├── 1d_collect_archives.py       # Archive.org scraper
│   ├── 1_collect_all_sources.py    # Main orchestrator for all scrapers
│   ├── clean_data.py                # Data cleaning & preprocessing (legacy)
│   ├── 2_extract_metadata.py       # NLP-based metadata extraction
│   ├── 3_generate_embeddings.py    # Sentence transformer embeddings
│   ├── 4_train_autoencoder.py      # Autoencoder training & clustering
│   └── 5_build_knowledge_graph.py  # KG construction with Lesk similarity
│
├── data/                          # Data storage (gitignored)
│   ├── raw/                         # Raw scraped documents (JSON)
│   ├── cleaned/                     # Cleaned text documents
│   ├── metadata/                    # Extracted metadata (enriched)
│   ├── embeddings/                  # Document embeddings (npy files)
│   └── classified/                  # Clustered documents with labels
│
├── models/                        # Trained models (gitignored)
│   └── autoencoder/                 # Autoencoder model (.pth)
│
└── knowledge_graph/               # KG data (gitignored except stats)
    ├── heritage_kg.gpickle          # NetworkX graph (pickle)
    ├── heritage_kg.gml              # Graph (GML format)
    ├── kg_statistics.json           # Graph statistics (tracked in git)
    └── kg_visualization.png         # Sample visualization
```

### File Naming Convention

Files follow a **sequential numbering system** that reflects the data pipeline order:
- `1x_*.py` - Data collection from different sources
- `2_*.py` - Data cleaning and metadata extraction
- `3_*.py` - Embedding generation
- `4_*.py` - Classification with autoencoder
- `5_*.py` - Knowledge graph construction
- `6_*.py` - (Future) KG integration with external sources
- `7_*.py` - (Future) Ranking algorithms (SimRank)
- `8_*.py` - (Future) Firework optimization algorithm
- `9_*.py` - (Future) Query processing and recommendations

---

## 🔄 Development Workflow

### **MANDATORY WORKFLOW FOR ALL TASKS**

When working on any task in this repository, you MUST follow these steps:

#### 1. **Think & Plan**
   - **Read the codebase** thoroughly to understand relevant files
   - **Identify the root cause** of any bugs or issues
   - **Create a plan** and write it to `tasks/todo.md`
   - The plan should include a **checklist of todo items** that can be marked as complete

#### 2. **Check In Before Starting**
   - **Present the plan to the user** for verification
   - **Wait for approval** before beginning work
   - This prevents wasted effort on incorrect approaches

#### 3. **Execute with Transparency**
   - **Work through todo items one by one**, marking them complete as you go
   - **Provide high-level explanations** of changes at each step
   - Keep the user informed of progress

#### 4. **Keep Changes Simple**
   - **Every change should be as simple as humanly possible**
   - **Impact only the code necessary** for the task
   - **Avoid massive or complex refactors**
   - Minimize the blast radius of each change

#### 5. **Review & Document**
   - Add a **review section** to `tasks/todo.md` summarizing:
     - What was changed
     - Why it was changed
     - Impact of the changes
   - Ensure all todos are marked complete

#### 6. **Senior Developer Mindset**
   - **NEVER be lazy** - find root causes, not symptoms
   - **NO temporary fixes** - solve problems properly
   - **NO introducing new bugs** - test thoroughly
   - If there's a bug, trace it to the source and fix it properly

### Example Workflow

```markdown
## tasks/todo.md

### Task: Fix metadata extraction bug

**Root Cause Analysis:**
- The NER chunking fails on documents with special characters
- Error occurs in `2_extract_metadata.py:87` when tokenizing

**Plan:**
- [ ] Add input validation for special characters
- [ ] Update tokenization to handle edge cases
- [ ] Add error logging for debugging
- [ ] Test on sample documents with known issues
- [ ] Verify fix doesn't break existing functionality

**Changes Made:**
- [x] Added input validation for special characters
- [x] Updated tokenization to handle edge cases
- [x] Added error logging for debugging
- [x] Test on sample documents with known issues
- [x] Verify fix doesn't break existing functionality

**Review:**
- Modified `2_extract_metadata.py:82-90` to add try-except with specific handling
- Added character sanitization before tokenization
- No breaking changes to API or data structures
- All existing tests pass
```

---

## 🔑 Key Conventions

### Code Style
- **Python Version:** 3.8+
- **Formatting:** Follow PEP 8 conventions
- **Documentation:** Docstrings for all functions (Google style)
- **Comments:** Explain complex logic, not obvious code

### Data Conventions
- **Document IDs:** Use `doc_{index}` format (0-indexed)
- **Entity IDs:** `loc_*`, `person_*`, `org_*`, `type_*`, `domain_*`, etc.
- **File encoding:** UTF-8 everywhere
- **JSON formatting:** `indent=2`, `ensure_ascii=False`

### Git Conventions
- **Branch naming:** `claude/claude-md-{session-id}`
- **Commit messages:** Clear, descriptive (what and why)
- **Never commit:** Large data files, models, embeddings (see `.gitignore`)

### Error Handling
- **Use try-except** for file I/O and external APIs
- **Log errors** with context (file name, line, operation)
- **Fail gracefully** - don't crash the entire pipeline
- **Validate inputs** at boundaries (user input, file loading)

### Performance Considerations
- **Batch processing:** Use batches for embeddings/ML operations
- **Limit text processing:** First N chars/sentences for speed
- **Cache when possible:** Reuse computed results
- **GPU support:** Check for CUDA availability (torch, transformers)

---

## 🏗️ Technical Architecture

### Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. DATA COLLECTION (1_collect_all_sources.py)                      │
│     ├── Wikipedia (1a)                                              │
│     ├── UNESCO (1b)                                                 │
│     ├── Indian Heritage (1c)                                        │
│     └── Archive.org (1d)                                            │
│     Output: data/raw/*.json                                         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  2. METADATA EXTRACTION (2_extract_metadata.py)                     │
│     ├── Named Entity Recognition (NLTK)                            │
│     ├── Heritage Type Classification (rule-based)                  │
│     ├── Domain Classification (cultural, religious, military, etc) │
│     ├── Time Period Detection (ancient, medieval, modern)          │
│     └── TF-IDF Keyword Extraction                                  │
│     Output: data/metadata/enriched_metadata.json                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  3. EMBEDDING GENERATION (3_generate_embeddings.py)                 │
│     ├── Model: sentence-transformers (all-MiniLM-L6-v2)           │
│     ├── Input: Title + Keywords + Content (first 2000 chars)       │
│     └── Output: 384-dimensional dense vectors                      │
│     Output: data/embeddings/document_embeddings.npy                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  4. AUTOENCODER CLASSIFICATION (4_train_autoencoder.py)             │
│     ├── Architecture: 384 → 128 → 64 → 128 → 384                   │
│     ├── Clustering: K-Means (12 clusters)                          │
│     ├── Visualization: t-SNE projection                            │
│     └── Semantic Labeling: Based on dominant heritage types        │
│     Output: data/classified/classified_documents.json               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  5. KNOWLEDGE GRAPH CONSTRUCTION (5_build_knowledge_graph.py)       │
│     ├── Nodes:                                                      │
│     │   ├── Documents (369)                                         │
│     │   ├── Entities (locations, persons, orgs)                    │
│     │   └── Concepts (types, domains, periods, regions)            │
│     ├── Edges:                                                      │
│     │   ├── Document-Entity relationships                          │
│     │   ├── Embedding similarity (cosine > 0.6)                    │
│     │   ├── Cluster membership (same_cluster)                      │
│     │   └── Concept similarity (Lesk algorithm)                    │
│     └── Statistics: Density, centrality, connectivity              │
│     Output: knowledge_graph/heritage_kg.gpickle                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Knowledge Graph Schema

**Node Types:**
- `document` - Heritage documents (369 nodes)
- `location` - Geographic entities
- `person` - Named individuals
- `organization` - Institutions/groups
- `heritage_type` - monument, site, artifact, architecture, tradition, art
- `domain` - religious, military, royal, cultural, archaeological, architectural
- `time_period` - ancient, medieval, modern
- `region` - north, south, east, west, central, india

**Edge Types:**
- `mentions_location` - Document → Location
- `mentions_person` - Document → Person
- `mentions_org` - Document → Organization
- `has_type` - Document → Heritage Type
- `belongs_to_domain` - Document → Domain
- `from_period` - Document → Time Period
- `located_in_region` - Document → Region
- `similar_to` - Document ↔ Document (cosine similarity)
- `same_cluster` - Document ↔ Document (clustering)
- `semantically_related` - Concept ↔ Concept (Lesk similarity)

**Graph Statistics (as of last build):**
- Total Nodes: 390
- Total Edges: 9,495
- Density: 0.125
- Average Degree: 48.69
- Connected: Yes (single component)

---

## 📊 Data Pipeline

### Input Data Sources

1. **Wikipedia** (`1a_collect_wikipedia.py`)
   - Heritage-related articles
   - Categories: monuments, sites, traditions

2. **UNESCO** (`1b_collect_unesco.py`)
   - World Heritage Sites database
   - Official descriptions and metadata

3. **Indian Heritage** (`1c_collect_indian_heritage.py`)
   - Archaeological Survey of India
   - State-specific monument databases

4. **Archive.org** (`1d_collect_archives.py`)
   - Historical documents
   - Digitized heritage texts

### Data Processing Stages

#### Stage 1: Collection
- **Output Format:** JSON files in `data/raw/`
- **Required Fields:** `title`, `content`, `url`, `source`, `collection_date`

#### Stage 2: Metadata Extraction
- **NER Entities:** Persons, locations, organizations, dates
- **Classifications:** Heritage types, domains, time periods, regions
- **Keywords:** TF-IDF top 15 keywords per document
- **Output:** `data/metadata/enriched_metadata.json`

#### Stage 3: Embedding Generation
- **Model:** `all-MiniLM-L6-v2` (384 dimensions)
- **Input Text:** `{title}. {keywords}. {content[:2000]}`
- **Normalization:** L2-normalized for cosine similarity
- **Output:** `data/embeddings/document_embeddings.npy`

#### Stage 4: Classification
- **Autoencoder:** Compress 384D → 64D → 384D
- **Clustering:** K-Means with k=12
- **Semantic Labels:** Derived from dominant heritage types/domains
- **Output:** `data/classified/classified_documents.json`

#### Stage 5: Knowledge Graph
- **Library:** NetworkX
- **Format:** Pickle (`.gpickle`) and GML (`.gml`)
- **Similarity Metrics:** Cosine (embeddings), Lesk (concepts)
- **Output:** `knowledge_graph/heritage_kg.gpickle`

---

## 🔧 Dependencies & Environment

### Python Dependencies (requirements.txt)

```txt
requests==2.31.0              # HTTP requests for scraping
beautifulsoup4==4.12.2        # HTML parsing
pandas==2.1.0                 # Data manipulation
numpy==1.24.3                 # Numerical operations
nltk==3.8.1                   # NLP (NER, tokenization, WordNet)
scikit-learn==1.3.0           # ML (TF-IDF, K-Means, PCA, t-SNE)
networkx==3.1                 # Graph construction/analysis
matplotlib==3.7.2             # Visualization
sentence-transformers==2.2.2  # Document embeddings
transformers==4.33.0          # Hugging Face models
torch==2.0.1                  # PyTorch (autoencoder)
wikipedia==1.4.0              # Wikipedia API
```

### NLTK Data Requirements

```python
# Run once to download required data
import nltk
nltk.download('punkt')           # Sentence tokenization
nltk.download('averaged_perceptron_tagger')  # POS tagging
nltk.download('maxent_ne_chunker')  # NER
nltk.download('words')           # Word corpus
nltk.download('stopwords')       # Stopword lists
nltk.download('wordnet')         # WordNet for Lesk similarity
nltk.download('omw-1.4')         # Open Multilingual WordNet
```

### System Requirements

- **Python:** 3.8 or higher
- **RAM:** 8GB minimum (16GB recommended for large datasets)
- **GPU:** Optional (CUDA-compatible for faster training)
- **Storage:** 2-5GB for data, models, and graphs

---

## 🛠️ Common Tasks

### Running the Full Pipeline

```bash
# 1. Collect data from all sources (~20-30 minutes)
python src/1_collect_all_sources.py

# 2. Extract metadata and classify documents (~5-10 minutes)
python src/2_extract_metadata.py

# 3. Generate embeddings (~5-10 minutes)
python src/3_generate_embeddings.py

# 4. Train autoencoder and cluster (~10-15 minutes)
python src/4_train_autoencoder.py

# 5. Build knowledge graph (~5 minutes)
python src/5_build_knowledge_graph.py
```

### Running Individual Scrapers

```bash
# Collect only Wikipedia data
python src/1a_collect_wikipedia.py

# Collect only UNESCO data
python src/1b_collect_unesco.py

# Collect only Indian heritage data
python src/1c_collect_indian_heritage.py

# Collect only Archive.org data
python src/1d_collect_archives.py
```

### Inspecting Data

```python
# Load metadata
import json
with open('data/metadata/enriched_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load embeddings
import numpy as np
embeddings = np.load('data/embeddings/document_embeddings.npy')

# Load knowledge graph
import networkx as nx
import pickle
with open('knowledge_graph/heritage_kg.gpickle', 'rb') as f:
    G = pickle.load(f)

# Load classified documents
with open('data/classified/classified_documents.json', 'r') as f:
    classified = json.load(f)
```

### Testing Changes

```python
# Test metadata extraction on a single document
from src.2_extract_metadata import extract_named_entities, classify_heritage_type

sample_text = "The Taj Mahal is a white marble mausoleum..."
entities = extract_named_entities(sample_text)
heritage_types = classify_heritage_type(sample_text)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **NLTK Data Not Found**
```python
# Error: Resource not found: punkt
# Solution: Download required NLTK data
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

#### 2. **File Not Found Errors**
```bash
# Error: data/metadata/enriched_metadata.json not found
# Solution: Run pipeline in correct order
# Each step depends on the previous step's output

# Check which step failed:
ls -R data/
```

#### 3. **Memory Issues**
```python
# Error: MemoryError during embedding generation
# Solution 1: Process in smaller batches
batch_size = 16  # Reduce from 32

# Solution 2: Reduce document length
content_snippet = text[:1000]  # Reduce from 2000
```

#### 4. **CUDA/GPU Issues**
```python
# Error: CUDA out of memory
# Solution: Force CPU usage
import torch
device = 'cpu'  # Instead of auto-detection

# Or in sentence-transformers:
model = SentenceTransformer(MODEL_NAME, device='cpu')
```

#### 5. **Encoding/Decoding Errors**
```python
# Error: UnicodeDecodeError
# Solution: Always use UTF-8 encoding
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
```

### Debugging Tips

1. **Check Pipeline Order**
   - Each script depends on outputs from previous steps
   - Verify all intermediate files exist before running next step

2. **Inspect Intermediate Outputs**
   - Check JSON files are valid (use `json.load()`)
   - Verify embeddings shape matches expected dimensions
   - Ensure graph has expected nodes/edges

3. **Log Everything**
   - All scripts include progress prints
   - Check console output for warnings and errors
   - Look for file paths and counts

4. **Test with Subsets**
   - When debugging, limit to first 10-20 documents
   - Speeds up iteration time
   - Example: `documents = documents[:10]`

5. **Validate Data Quality**
   - Check for empty documents or missing fields
   - Verify entity extraction found results
   - Ensure classifications are reasonable

---

## 📝 Notes for AI Assistants

### When Making Changes

1. **Always read relevant files first** - Understand context before modifying
2. **Test incrementally** - Don't make massive changes at once
3. **Preserve backward compatibility** - Don't break existing data files
4. **Update documentation** - Reflect changes in this file
5. **Follow the naming conventions** - Maintain consistency
6. **Use the TodoWrite tool** - Track complex multi-step tasks
7. **Ask for clarification** - If requirements are ambiguous

### What to Avoid

❌ **Don't:**
- Make changes without reading the code first
- Introduce breaking changes to data schemas
- Skip error handling
- Ignore the sequential pipeline order
- Commit large binary files (data, models)
- Use temporary fixes instead of finding root causes
- Introduce unnecessary complexity
- Make massive refactors that affect multiple files

✅ **Do:**
- Find and fix root causes
- Keep changes minimal and focused
- Add validation and error handling
- Test edge cases
- Document non-obvious logic
- Follow existing code patterns
- Use the mandatory workflow described above

### Code Modification Guidelines

**Before modifying any file:**
1. Read the entire file
2. Understand its role in the pipeline
3. Check what files depend on it
4. Identify the minimal change needed

**When fixing bugs:**
1. Reproduce the bug
2. Trace to root cause (not symptoms)
3. Write a fix that addresses the root cause
4. Test the fix thoroughly
5. Verify no new bugs introduced

**When adding features:**
1. Plan the implementation in `tasks/todo.md`
2. Get user approval
3. Implement incrementally
4. Test each increment
5. Update documentation

---

## 🎓 Academic References

This project implements techniques from:
- **Knowledge Graphs:** Entity-relation modeling, graph-based ranking
- **Deep Learning:** Autoencoders for dimensionality reduction
- **NLP:** Named entity recognition, semantic similarity (Lesk)
- **Information Retrieval:** TF-IDF, cosine similarity, clustering

**Future Enhancements (as per README):**
- SimRank algorithm for graph-based similarity
- Horn's Index for entity importance weighting
- Firework Algorithm for metaheuristic optimization
- Query processing and recommendation interface

---

## 📞 Contact & Support

**Project Author:** Akchhya Singh
**Email:** akchhya1108@gmail.com
**Repository:** https://github.com/aryananand-04/heritage_doc_recomm
**Status:** In Development (v1.0.0)

**For AI Assistants:**
- This is an academic project (final year submission)
- Code quality and correctness are critical
- Always follow the mandatory workflow
- When in doubt, ask the user for clarification
- Never compromise on finding root causes

---

**END OF CLAUDE.MD**
