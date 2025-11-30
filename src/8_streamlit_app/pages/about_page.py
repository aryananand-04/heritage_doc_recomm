"""
About Page
Information about the system, methodology, and team
"""

import streamlit as st


def render():
    """Render about page."""
    st.markdown('<h1 class="main-header">ℹ️ About This System</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    The **Heritage Document Recommendation System** is an AI-powered search and discovery platform
    for heritage-related documents. It combines knowledge graphs, graph algorithms, and deep learning
    to provide accurate and explainable recommendations.
    """)
    
    # Tabs for organized content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Overview", 
        "🔬 Methodology", 
        "📊 Performance", 
        "👥 Team", 
        "📄 Citation"
    ])
    
    with tab1:
        st.markdown("## System Overview")
        
        st.markdown("""
        ### 🎯 Purpose
        
        Heritage documents are scattered across multiple sources and lack unified search capabilities.
        This system addresses the challenge by:
        
        - **Aggregating** heritage documents from diverse sources
        - **Organizing** documents using semantic clustering and knowledge graphs
        - **Recommending** relevant documents based on user queries
        - **Explaining** recommendations through graph-based reasoning
        
        ### 📚 Dataset
        
        - **Total Documents**: 369 heritage documents
        - **Sources**: Wikipedia, UNESCO World Heritage, Indian Heritage, Archive.org
        - **Coverage**: Global heritage sites with focus on Indian monuments
        - **Clusters**: 12 thematic clusters (temples, forts, palaces, etc.)
        
        ### 🕸️ Knowledge Graph
        
        - **Nodes**: ~500+ entities
          - Documents (369)
          - Locations (cities, states, regions)
          - Persons (historical figures, architects)
          - Organizations (ASI, UNESCO)
          - Concepts (heritage types, domains, time periods)
        
        - **Edges**: ~6,500+ relationships
          - Document similarities
          - Entity mentions
          - Semantic relationships
          - Temporal/spatial connections
        
        ### 🎨 Features
        
        ✅ Natural language query processing  
        ✅ Hybrid ranking (SimRank + Horn's Index + Embeddings)  
        ✅ Interactive knowledge graph visualization  
        ✅ Explainable recommendations with KG paths  
        ✅ Real-time search (<1ms latency)  
        ✅ Comprehensive evaluation metrics  
        """)
    
    with tab2:
        st.markdown("## Methodology")
        
        st.markdown("""
        ### 🔄 System Pipeline
        
        ```
        Data Collection → Preprocessing → Embeddings → Knowledge Graph
                ↓
        Query Input → Query Processing → Hybrid Ranking → Results
        ```
        
        ### 1️⃣ Data Collection & Processing
        
        **Collection:**
        - Scraped 369 documents from 4 authoritative sources
        - Validated content quality and removed duplicates
        - Extracted metadata (title, URL, categories, date)
        
        **Preprocessing:**
        - Text cleaning (HTML removal, normalization)
        - Entity extraction using spaCy + custom rules
        - Classification into heritage types, domains, time periods
        
        **Representation:**
        - Sentence embeddings using `all-MiniLM-L6-v2` (384-dim)
        - Autoencoder compression (384 → 64 dimensions)
        - K-Means clustering (12 clusters)
        
        ### 2️⃣ Knowledge Graph Construction
        
        **Nodes:**
        - **Documents**: Heritage articles/texts
        - **Entities**: Locations, persons, organizations
        - **Concepts**: Heritage types, domains, periods, regions
        
        **Edges:**
        - **Similarity**: Cosine similarity (threshold: 0.6)
        - **Mentions**: Document ↔ Entity connections
        - **Semantic**: Concept relationships (Lesk similarity)
        - **Structural**: Cluster membership
        
        **Algorithms:**
        - **SimRank**: Computes structural similarity between documents
          - Formula: `S(a,b) = C × Σ S(neighbors(a), neighbors(b)) / |neighbors|`
          - Decay factor C = 0.8, Max iterations = 10
        
        - **Horn's Index**: Weights entity importance
          - Formula: `Horn(e) = 0.3×degree + 0.2×betweenness + 0.3×pagerank + 0.2×doc_freq`
          - Identifies central/important entities
        
        ### 3️⃣ Query Processing
        
        **Input**: Natural language query (e.g., "Mughal temples in North India")
        
        **Extraction:**
        - Heritage types (temple, fort, etc.)
        - Domains (religious, military, etc.)
        - Time period (ancient, medieval, modern)
        - Region (north, south, east, west, central)
        - Architectural styles (Mughal, Dravidian, etc.)
        - Named entities (locations, persons, organizations)
        
        **Embedding**: Query → 384-dim vector using sentence transformer
        
        ### 4️⃣ Hybrid Ranking
        
        **Three Components:**
        
        1. **SimRank Score (40% weight)**
           - Graph-based structural similarity
           - Measures "relatedness" through KG connections
           - Best performer in evaluations
        
        2. **Horn's Index Score (30% weight)**
           - Entity importance weighting
           - Matches query entities to document entities
           - Rewards documents with important/central entities
        
        3. **Embedding Similarity (30% weight)**
           - Semantic similarity via FAISS index
           - Cosine similarity between query and document embeddings
           - Captures textual/semantic relevance
        
        **Final Score:**
        ```
        Hybrid = 0.4 × SimRank + 0.3 × Horn + 0.3 × Embedding
        ```
        
        **Output**: Top-K documents ranked by hybrid score
        
        ### 5️⃣ Explainability
        
        - **KG Paths**: Find shortest paths between query-relevant docs and recommendations
        - **Score Breakdown**: Show contribution of each component
        - **Entity Overlap**: Highlight shared entities between query and results
        """)
    
    with tab3:
        st.markdown("## Performance & Evaluation")
        
        st.markdown("""
        ### 📊 Evaluation Setup
        
        **Ground Truth Generation:**
        - 140 test queries from 4 strategies:
          - Cluster-based (same cluster = relevant)
          - Metadata-based (shared attributes = relevant)
          - Embedding-based (cosine sim > 0.7 = relevant)
          - SimRank-based (SimRank > 0.02 = relevant)
        
        **Metrics:**
        - Precision@K, Recall@K, F1@K
        - NDCG@K (Normalized Discounted Cumulative Gain)
        - MAP (Mean Average Precision)
        - Coverage, Diversity
        - Heritage-specific: Temporal accuracy, Spatial relevance, Domain alignment
        
        ### 🏆 Results Summary
        
        | Method | Precision@5 | NDCG@10 | Latency | Verdict |
        |--------|-------------|---------|---------|---------|
        | **Hybrid (50-50)** | **82.8%** | **68.7%** | **0.21ms** | ✅ **Best** |
        | SimRank-Only | 82.4% | 68.4% | 0.24ms | ✅ Excellent |
        | Hybrid (70% SR) | 82.4% | 68.6% | 0.33ms | ⚠️ Slower |
        | Hybrid (70% Emb) | 81.2% | 68.3% | 0.21ms | ⚠️ Lower |
        | Embedding-Only | 27.6% | 28.8% | 0.28ms | ❌ Poor |
        
        ### 🎯 Key Findings
        
        1. **SimRank Dominates**: Graph-based similarity outperforms embeddings by 199%
        2. **Hybrid is Best**: 50-50 mix achieves highest precision (82.8%)
        3. **Fast & Accurate**: <0.3ms latency with >80% precision
        4. **Good Coverage**: 87.8% of catalog appears in recommendations
        
        ### 📈 Strengths
        
        ✅ **High Precision**: 82.8% @ P@5 (excellent for recommendation)  
        ✅ **Real-time**: <1ms query latency (production-ready)  
        ✅ **Explainable**: KG paths provide interpretable reasoning  
        ✅ **Diverse**: 87.8% catalog coverage (explores full collection)  
        ✅ **Scalable**: FAISS indexing enables fast search  
        
        ### 🔧 Areas for Improvement
        
        ⚠️ **Recall**: 15% @ R@10 (misses some relevant docs)  
        ⚠️ **Spatial Relevance**: 43% (geographic info poorly captured)  
        ⚠️ **Domain Alignment**: 53% (multi-domain docs confuse system)  
        
        ### 💡 Future Work
        
        - Fine-tune embeddings on heritage domain corpus
        - Improve location/region extraction algorithms
        - Add hierarchical domain taxonomy
        - Expand KG with more entity relationships
        - User studies for real-world validation
        """)
    
    with tab4:
        st.markdown("## Team & Development")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### 👨‍💻 Developer
            
            **Akchhya Singh**  
            Final Year B.Tech Student  
            Computer Science & Engineering
            
            📧 akchhya1108@gmail.com  
            🔗 GitHub: [Link]
            
            ---
            
            ### 🎓 Academic Context
            
            **Project Type**: Final Year Project  
            **Year**: 2024-2025  
            **Status**: ✅ Complete  
            **Target**: Conference Publication
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ Technology Stack
            
            **Data Processing:**
            - Python 3.8+
            - Pandas, NumPy
            - spaCy (NLP)
            - BeautifulSoup (scraping)
            
            **Machine Learning:**
            - PyTorch (autoencoder)
            - Sentence Transformers (embeddings)
            - scikit-learn (clustering)
            - FAISS (similarity search)
            
            **Knowledge Graph:**
            - NetworkX (graph construction)
            - SimRank algorithm
            - Custom Horn's Index implementation
            
            **Visualization:**
            - Streamlit (web UI)
            - Plotly (interactive charts)
            - PyVis (KG visualization)
            
            **Storage:**
            - NumPy arrays (embeddings)
            - Pickle (KG serialization)
            - JSON (metadata, results)
            
            ### 📦 Project Statistics
            
            - **Lines of Code**: ~8,000+
            - **Python Files**: 40+
            - **Dependencies**: 25+ packages
            - **Development Time**: 6 months
            - **Dataset Size**: 369 documents, 500+ KG nodes
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🙏 Acknowledgments
        
        This project was made possible with guidance and resources from:
        
        - **Faculty Advisors**: [Names]
        - **Department**: Computer Science & Engineering
        - **Institution**: [University Name]
        - **Data Sources**: Wikipedia, UNESCO, Archaeological Survey of India, Archive.org
        
        Special thanks to the open-source community for providing excellent tools and libraries.
        """)
    
    with tab5:
        st.markdown("## Citation & License")
        
        st.markdown("""
        ### 📄 How to Cite
        
        If you use this system or methodology in your research, please cite:
        
        ```bibtex
        @misc{singh2025heritage,
          title={Heritage Document Recommendation System: A Hybrid Knowledge Graph Approach},
          author={Singh, Akchhya},
          year={2025},
          institution={[Your University]},
          type={Final Year Project},
          note={Available at: [GitHub URL]}
        }
        ```
        
        ### 📜 License
        
        This project is released under the **MIT License**.
        
        ```
        MIT License
        
        Copyright (c) 2025 Akchhya Singh
        
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        ```
        
        ### 📚 References
        
        **Key Papers:**
        
        1. Jeh, G., & Widom, J. (2002). SimRank: A measure of structural-context similarity. 
           *KDD '02: Proceedings of the eighth ACM SIGKDD*.
        
        2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using 
           Siamese BERT-Networks. *EMNLP 2019*.
        
        3. Page, L., et al. (1999). The PageRank Citation Ranking: Bringing Order to the Web. 
           *Stanford InfoLab Technical Report*.
        
        **Datasets:**
        
        - Wikipedia (CC BY-SA 3.0)
        - UNESCO World Heritage Centre
        - Archaeological Survey of India
        - Archive.org
        
        ### 🔗 Useful Links
        
        - **Source Code**: [GitHub Repository]
        - **Documentation**: [Project Wiki]
        - **Demo Video**: [YouTube Link]
        - **Paper Preprint**: [ArXiv Link]
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #1e293b; border-radius: 8px; border: 1px solid #334155;">
        <p style="font-size: 1.2rem; font-weight: bold; color: #f1f5f9;">Questions or Feedback?</p>
        <p style="color: #f1f5f9;">Contact: <a href="mailto:akchhya1108@gmail.com" style="color: #60a5fa;">akchhya1108@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    render()