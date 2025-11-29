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
st.markdown("""
<style>
    /* ===== Global Styles ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* ===== Main Header ===== */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 2px solid #dee2e6;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* ===== Cards & Containers ===== */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left-color: #764ba2;
    }
    
    /* ===== Buttons ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .score-medium {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    
    .score-low {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* ===== KG Path Styling ===== */
    .kg-path {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-left: 4px solid #2196f3;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* ===== Info Boxes ===== */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* ===== Expanders ===== */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #e9ecef;
    }
    
    /* ===== Progress Bars ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* ===== Metrics ===== */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* ===== Dividers ===== */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e9ecef;
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

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 0.8rem; color: #666;">
    <p>Heritage Document Recommender v2.0</p>
    <p>© 2025 Akchhya Singh</p>
</div>
""", unsafe_allow_html=True)

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
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 2.5rem; margin: 0;'>🏛️</h1>
        <h2 style='color: #667eea; font-weight: 700; margin: 0.5rem 0;'>Heritage</h2>
        <p style='color: #6c757d; font-size: 0.9rem;'>Document Recommender v2.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation with icons
    page = st.radio(
        "Navigate to:",
        [
            "🔍 Search & Discover",
            "🕸️ Knowledge Graph",
            "📊 Results & Analysis",
            "📈 Performance Metrics"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📈 System Stats")
    
    try:
        import json
        with open('knowledge_graph/kg_statistics.json', 'r') as f:
            kg_stats = json.load(f)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", f"{kg_stats['total_nodes']:,}")
        with col2:
            st.metric("Edges", f"{kg_stats['total_edges']:,}")
    except:
        st.info("KG stats unavailable")
    
    st.markdown("---")
    
    # Help section
    with st.expander("ℹ️ Need Help?"):
        st.markdown("""
        **Quick Guide:**
        1. Start with **Search** page
        2. Enter a natural language query
        3. View recommendations
        4. Explore **Knowledge Graph** connections
        5. Analyze **Results** in detail
        
        **Example Queries:**
        - "Mughal temples in North India"
        - "Ancient Buddhist stupas"
        - "Rajput forts in Rajasthan"
        """)

# ========== MAIN CONTENT ==========

# Route to pages
if page == "🔍 Search & Discover":
    from pages import search_page
    search_page.render()
    
elif page == "🕸️ Knowledge Graph":
    from pages import kg_viz_page
    kg_viz_page.render()
    
elif page == "📊 Results & Analysis":
    from pages import results_page
    results_page.render()
    
elif page == "📈 Performance Metrics":
    from pages import evaluation_page
    evaluation_page.render()

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 2rem 0; font-size: 0.9rem;'>
    <p>Built with ❤️ using Streamlit • Knowledge Graph-based Recommendations</p>
    <p>📧 Contact: akchhya1108@gmail.com | 🎓 Research Project 2025</p>
</div>
""", unsafe_allow_html=True)