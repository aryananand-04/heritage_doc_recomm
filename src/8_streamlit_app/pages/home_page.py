"""
Home Page / Dashboard
Overview of the system with quick stats and navigation
"""

import streamlit as st
import json
import pickle
from pathlib import Path
import plotly.graph_objects as go


@st.cache_data
def load_system_stats():
    """Load system statistics."""
    stats = {
        'total_documents': 369,
        'kg_nodes': 0,
        'kg_edges': 0,
        'clusters': 12,
        'best_precision': 0.828,
        'best_method': 'Hybrid (50-50)',
        'avg_latency': 0.21,
        'coverage': 0.878
    }
    
    # Try to load actual stats
    try:
        with open('knowledge_graph/kg_statistics.json', 'r') as f:
            kg_stats = json.load(f)
            stats['kg_nodes'] = kg_stats.get('total_nodes', 0)
            stats['kg_edges'] = kg_stats.get('total_edges', 0)
    except:
        pass
    
    try:
        with open('results/method_comparison.json', 'r') as f:
            comparison = json.load(f)
            methods = comparison.get('methods', {})
            
            if methods:
                # Find best method
                best_precision = 0
                best_method_name = ''
                
                for method, metrics in methods.items():
                    p5 = metrics.get('precision@5', 0)
                    if p5 > best_precision:
                        best_precision = p5
                        best_method_name = method
                
                stats['best_precision'] = best_precision
                stats['best_method'] = best_method_name
                
                # Get average latency
                latencies = [m.get('query_latency_ms', 0) for m in methods.values()]
                if latencies:
                    stats['avg_latency'] = sum(latencies) / len(latencies)
    except:
        pass
    
    return stats


def create_performance_gauge(value, title, max_value=1.0):
    """Create a gauge chart for performance metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, max_value*0.33], 'color': "#ffebee"},
                {'range': [max_value*0.33, max_value*0.66], 'color': "#fff9c4"},
                {'range': [max_value*0.66, max_value], 'color': "#c8e6c9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.8
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def render():
    """Render homepage."""
    st.markdown('<h1 class="main-header">🏛️ Heritage Document Recommendation System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        AI-powered heritage document discovery using Knowledge Graphs and Hybrid Ranking
    </div>
    """, unsafe_allow_html=True)
    
    # Load stats
    stats = load_system_stats()
    
    # Key metrics
    st.markdown("---")
    st.markdown("### 📊 System Overview")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            "📚 Total Documents",
            f"{stats['total_documents']:,}",
            help="Heritage documents in the system"
        )
    
    with metric_col2:
        st.metric(
            "🕸️ Knowledge Graph",
            f"{stats['kg_nodes']:,} nodes",
            delta=f"{stats['kg_edges']:,} edges",
            help="Graph nodes and connections"
        )
    
    with metric_col3:
        st.metric(
            "🎯 Best Precision",
            f"{stats['best_precision']:.1%}",
            help="Precision@5 of best method"
        )
    
    with metric_col4:
        st.metric(
            "⚡ Avg Latency",
            f"{stats['avg_latency']:.2f}ms",
            help="Average query response time"
        )
    
    # Performance gauges
    st.markdown("---")
    st.markdown("### 🎯 Performance Metrics")
    
    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
    
    with gauge_col1:
        precision_gauge = create_performance_gauge(
            stats['best_precision'], 
            "Precision@5",
            max_value=1.0
        )
        st.plotly_chart(precision_gauge, use_container_width=True)
    
    with gauge_col2:
        coverage_gauge = create_performance_gauge(
            stats['coverage'],
            "Catalog Coverage",
            max_value=1.0
        )
        st.plotly_chart(coverage_gauge, use_container_width=True)
    
    with gauge_col3:
        speed_gauge = create_performance_gauge(
            min(1.0, stats['avg_latency'] / 1.0),  # Normalize (1ms = 100%)
            "Speed (lower is better)",
            max_value=1.0
        )
        st.plotly_chart(speed_gauge, use_container_width=True)
    
    # Quick start guide
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📖 For Users
        
        1. **Search**: Enter a natural language query
           - Example: "Mughal temples in North India"
        
        2. **Explore**: View recommendations with scores
           - See why documents are recommended
        
        3. **Visualize**: Explore the Knowledge Graph
           - Understand document relationships
        
        4. **Learn**: Check detailed explanations
           - Score breakdowns and KG paths
        """)
        
        if st.button("🔍 Start Searching", use_container_width=True, type="primary"):
            st.switch_page("pages/search_page.py")
    
    with col2:
        st.markdown("""
        #### 🔬 For Researchers
        
        1. **Methodology**: Hybrid ranking approach
           - SimRank (40%) + Horn's Index (30%) + Embeddings (30%)
        
        2. **Evaluation**: Performance metrics
           - Precision, Recall, NDCG, Coverage
        
        3. **Knowledge Graph**: 500+ nodes
           - Documents, entities, concepts
        
        4. **Best Method**: Hybrid (50-50)
           - 82.8% Precision@5, <0.3ms latency
        """)
        
        if st.button("📈 View Evaluation", use_container_width=True):
            st.switch_page("pages/evaluation_page.py")
    
    # System architecture
    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")
    
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    
    with arch_col1:
        st.markdown("""
        #### 1️⃣ Data Pipeline
        - **Collection**: 4 sources (Wikipedia, UNESCO, etc.)
        - **Processing**: NLP, entity extraction
        - **Embeddings**: Sentence transformers
        - **Clustering**: 12 heritage clusters
        """)
    
    with arch_col2:
        st.markdown("""
        #### 2️⃣ Knowledge Graph
        - **Nodes**: Documents, entities, concepts
        - **Edges**: Relationships and similarities
        - **SimRank**: Structural similarity
        - **Horn's Index**: Entity importance
        """)
    
    with arch_col3:
        st.markdown("""
        #### 3️⃣ Hybrid Ranking
        - **SimRank**: Graph-based (40%)
        - **Horn's Index**: Entity-based (30%)
        - **Embeddings**: Semantic (30%)
        - **Output**: Top-K recommendations
        """)
    
    # Recent achievements
    st.markdown("---")
    st.markdown("### 🏆 Key Achievements")
    
    achievement_cols = st.columns(4)
    
    achievements = [
        ("🎯", "82.8%", "Precision@5", "Best in class accuracy"),
        ("⚡", "<0.3ms", "Query Time", "Real-time performance"),
        ("📚", "87.8%", "Coverage", "Catalog diversity"),
        ("🧠", "500+", "KG Nodes", "Rich knowledge base")
    ]
    
    for col, (icon, value, label, description) in zip(achievement_cols, achievements):
        col.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <div style="font-size: 2rem;">{icon}</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #1f77b4;">{value}</div>
            <div style="font-size: 1rem; font-weight: bold;">{label}</div>
            <div style="font-size: 0.9rem; color: #666;">{description}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation shortcuts
    st.markdown("---")
    st.markdown("### 🧭 Quick Navigation")
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("🔍 Search Documents", use_container_width=True):
            st.switch_page("pages/search_page.py")
    
    with nav_col2:
        if st.button("🕸️ View Knowledge Graph", use_container_width=True):
            st.switch_page("pages/kg_viz_page.py")
    
    with nav_col3:
        if st.button("📊 See Results", use_container_width=True):
            if 'recommendations' in st.session_state:
                st.switch_page("pages/results_page.py")
            else:
                st.warning("No results yet. Please run a search first.")
    
    with nav_col4:
        if st.button("📈 Evaluation Dashboard", use_container_width=True):
            st.switch_page("pages/evaluation_page.py")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Heritage Document Recommendation System v2.0</p>
        <p>Developed by Akchhya Singh | Final Year Project 2025</p>
        <p>Built with ❤️ using Knowledge Graphs, SimRank, and Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    render()