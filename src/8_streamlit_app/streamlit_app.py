"""
Heritage Document Recommendation System - Modern Dashboard

Enhanced UI with:
- Material Design color scheme
- Responsive layouts
- Interactive widgets
- Real-time feedback
- Better visual hierarchy
"""

import streamlit as st
import sys
from pathlib import Path

# Configure page FIRST
st.set_page_config(
    page_title="Heritage Recommender",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Heritage Document Recommendation System v2.0"
    }
)

# Add paths
sys.path.append(str(Path(__file__).parent.parent / '6_query_system'))

# ========== MODERN CSS STYLING ==========
# Note: Main styling is in style.css, this is just for compatibility
st.markdown("""
<style>
    /* ===== Global Styles ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #f1f5f9;
        background-color: #0f172a;
    }
    
    /* ===== Main Header ===== */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #f1f5f9;
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background: #1e293b;
        border-right: 1px solid #334155;
        color: #f1f5f9;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* ===== Cards & Containers ===== */
    .metric-card {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #f1f5f9;
        margin: 1rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid #334155;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        background: #334155;
    }
    
    .result-card {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        margin: 1rem 0;
        transition: all 0.2s ease;
        border: 1px solid #334155;
        color: #f1f5f9;
    }
    
    .result-card:hover {
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        border-left-color: #0d9488;
        background: #334155;
    }
    
    /* ===== Buttons ===== */
    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* ===== Score Badges ===== */
    .score-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
    
    .score-high {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .score-medium {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .score-low {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    /* ===== KG Path Styling ===== */
    .kg-path {
        background: #1e293b;
        padding: 1rem;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        border: 1px solid #334155;
        color: #f1f5f9;
    }
    
    /* ===== Info Boxes ===== */
    .info-box {
        background: #1e3a5f;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #3b82f6;
        color: #e0e7ff;
    }
    
    .success-box {
        background: #064e3b;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #10b981;
        color: #a7f3d0;
    }
    
    .warning-box {
        background: #78350f;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #f59e0b;
        color: #fde68a;
    }
    
    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1e293b;
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        border: 1px solid #334155;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background: #2563eb;
        color: white;
    }
    
    /* ===== Expanders ===== */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 8px;
        font-weight: 600;
        padding: 1rem;
        transition: all 0.2s ease;
        border: 1px solid #334155;
        color: #f1f5f9;
    }
    
    .streamlit-expanderHeader:hover {
        background: #334155;
        border-color: #2563eb;
    }
    
    /* ===== Progress Bars ===== */
    .stProgress > div > div {
        background: #2563eb;
    }
    
    /* ===== Metrics ===== */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    
    /* ===== Dividers ===== */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #334155;
    }
    
    /* ===== Animations ===== */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .animated-card {
        animation: slideIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ========== SIDEBAR NAVIGATION ==========

# Sidebar navigation
st.sidebar.title("🏛️ Heritage Recommender")
st.sidebar.markdown("---")

# Add logo or image (optional)
# st.sidebar.image("assets/logo.png", width=200)

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Search", "🕸️ Knowledge Graph", "📊 Results & Explanations", "📈 Evaluation", "ℹ️ About"],
    label_visibility="collapsed"
)

# System stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Documents", "369")
st.sidebar.metric("Best Precision", "82.8%")
st.sidebar.metric("Avg Latency", "<0.3ms")

# Import and render pages
if page == "🏠 Home":
    from pages import home_page
    home_page.render()
elif page == "🔍 Search":
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
elif page == "ℹ️ About":
    from pages import about_page
    about_page.render()