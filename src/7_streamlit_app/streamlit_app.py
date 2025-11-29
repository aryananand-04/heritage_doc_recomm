"""
Heritage Document Recommendation System - Streamlit Dashboard

4-page application:
1. Search Interface - Query with filters
2. Knowledge Graph Visualization - Interactive PyVis graph
3. Recommendation Results - Detailed explanations with score breakdowns
4. Evaluation Dashboard - Performance metrics comparison
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent / '6_query_system'))

# Page configuration
st.set_page_config(
    page_title="Heritage Doc Recommender",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .score-breakdown {
        font-family: monospace;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .kg-path {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🏛️ Heritage Recommender")
page = st.sidebar.radio(
    "Navigation",
    ["🔍 Search", "🕸️ Knowledge Graph", "📊 Results & Explanations", "📈 Evaluation"],
    label_visibility="collapsed"
)

# Import pages
if page == "🔍 Search":
    from pages import search_page
    search_page.render()
elif page == "🕸️ Knowledge Graph":
    from pages import kg_viz_page
    kg_viz_page.render()
elif page == "📊 Results & Explanations":
    from pages import results_page
    results_page.render()
elif page == "📈 Evaluation":
    from pages import evaluation_page
    evaluation_page.render()
