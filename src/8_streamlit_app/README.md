# Heritage Document Recommendation System - Streamlit Dashboard

Interactive web application for searching and exploring heritage documents using hybrid knowledge graph-based recommendations.

## Features

### 🔍 Page 1: Search Interface
- Natural language query input
- Heritage filters (type, domain, period, region, architectural style)
- Real-time recommendations with hybrid scoring
- Score breakdown visualization
- Customizable component weights (SimRank, Horn's Index, Embeddings)
- Example queries for quick testing

### 🕸️ Page 2: Knowledge Graph Visualization
- Interactive PyVis network visualization
- Multiple visualization modes:
  - Full graph
  - Subgraph from search results
  - Subgraph around specific node
  - Neighborhood sampling
- Node and edge type filtering
- Customizable layout algorithms (spring, hierarchical, circular)
- Physics simulation toggle
- Export options (GraphML, statistics JSON)

### 📊 Page 3: Results & Explanations
- Detailed recommendation results
- Interactive score breakdown charts (Plotly)
- Knowledge graph path explanations
- Document metadata display
- Score distribution analysis
- Comparison charts across recommendations
- Sorting and filtering options
- Export to CSV, JSON, or Markdown report

### 📈 Page 4: Evaluation Dashboard
- Performance metrics comparison (Precision, Recall, NDCG, MRR)
- Interactive visualizations:
  - Bar charts for metric comparison
  - Radar charts for multi-dimensional analysis
  - Heatmaps for all metrics
  - Precision-Recall curves
- Best performing method identification
- Detailed metrics breakdown
- Export evaluation results

## Prerequisites

Before running the Streamlit app, ensure you have:

1. **Built the Knowledge Graph:**
   ```bash
   python src/4_knowledge_graph/5_build_knowledge_graph.py
   ```

2. **Computed Horn's Index weights:**
   ```bash
   python src/6_query_system/horn_index.py
   ```

3. **Generated SimRank matrix** (if not already available):
   ```bash
   python src/4_knowledge_graph/compute_simrank.py
   ```

4. **Installed all dependencies:**
   ```bash
   pip install streamlit plotly pyvis spacy sentence-transformers faiss-cpu networkx
   python -m spacy download en_core_web_sm
   ```

## Running the App

### Option 1: From the streamlit_app directory

```bash
cd src/7_streamlit_app
streamlit run streamlit_app.py
```

### Option 2: From the project root

```bash
streamlit run src/7_streamlit_app/streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Configuration

### Score Weights

Default hybrid scoring weights (can be customized in the Search page):
- **SimRank**: 0.4 (40%) - Structural graph similarity
- **Horn's Index**: 0.3 (30%) - Entity importance weights
- **Embedding Similarity**: 0.3 (30%) - Semantic similarity

### File Paths

The app expects the following files to be available:

- **Knowledge Graph**: `knowledge_graph/heritage_kg.gpickle`
- **SimRank Matrix**: `knowledge_graph/simrank/simrank_matrix.npy`
- **Embeddings**: `data/embeddings/document_embeddings.npy`
- **Embedding Metadata**: `data/embeddings/embedding_mapping.json`
- **FAISS Index**: `models/ranker/faiss/hnsw_index.faiss`
- **Horn's Weights**: `knowledge_graph/horn_weights.json`
- **Evaluation Results** (optional): `results/evaluation_results.json`

## Usage Examples

### Example Queries

1. **"Mughal temples in North India"**
   - Extracts: heritage_type=temple, time_period=medieval, region=north, style=mughal
   - Returns documents about Mughal architecture in northern India

2. **"Ancient forts in Rajasthan"**
   - Extracts: heritage_type=fort, time_period=ancient, region=west, location=Rajasthan
   - Returns documents about historical fortifications in Rajasthan

3. **"Buddhist stupas and monasteries"**
   - Extracts: heritage_type=stupa, monastery, style=buddhist
   - Returns documents about Buddhist religious architecture

### Workflow

1. **Search**: Enter a natural language query on the Search page
2. **View Results**: See top-K recommendations with score breakdowns
3. **Explore**: Navigate to Results page for detailed analysis
4. **Visualize**: View the Knowledge Graph to understand connections
5. **Evaluate**: Check the Evaluation Dashboard for performance metrics

## Troubleshooting

### Common Issues

**Issue**: "Failed to load recommendation system"
- **Solution**: Make sure you've run `python src/6_query_system/horn_index.py` to generate Horn weights

**Issue**: "No module named 'pyvis'"
- **Solution**: Install missing dependencies: `pip install pyvis`

**Issue**: "File not found: knowledge_graph/heritage_kg.gpickle"
- **Solution**: Run the KG builder: `python src/4_knowledge_graph/5_build_knowledge_graph.py`

**Issue**: Slow visualization on large graphs
- **Solution**: Use the "Max Nodes to Display" slider to limit the number of nodes rendered

### Performance Tips

1. **Limit visualizations**: Use the max_nodes slider (default 100) for faster rendering
2. **Disable physics**: Turn off physics simulation for static layouts
3. **Filter nodes/edges**: Use type filters to reduce graph complexity
4. **Use subgraphs**: Visualize only relevant portions of the graph

## Architecture

```
src/7_streamlit_app/
├── streamlit_app.py           # Main app entry point
├── pages/
│   ├── __init__.py
│   ├── search_page.py         # Search interface
│   ├── kg_viz_page.py         # KG visualization
│   ├── results_page.py        # Results & explanations
│   └── evaluation_page.py     # Evaluation dashboard
└── README.md                   # This file
```

## Technologies

- **Streamlit**: Web framework
- **Plotly**: Interactive charts
- **PyVis**: Network visualization
- **NetworkX**: Graph analysis
- **Pandas**: Data manipulation
- **spaCy**: NLP for query processing
- **Sentence Transformers**: Query embeddings
- **FAISS**: Fast similarity search

## Session State

The app uses Streamlit session state to share data between pages:

- `query_text`: Original search query
- `parsed_query`: Extracted query attributes
- `recommendations`: Top-K recommendation results
- `top_k`: Number of results requested

## Contributing

To add new features or pages:

1. Create a new page in `src/7_streamlit_app/pages/`
2. Implement a `render()` function
3. Import and call in `streamlit_app.py`

## License

Part of the Heritage Document Recommendation System research project.

## Contact

For issues or questions, please create an issue in the project repository.
