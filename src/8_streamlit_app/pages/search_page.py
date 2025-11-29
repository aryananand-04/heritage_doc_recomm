import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent.parent / '6_query_system'))

from query_processor import QueryProcessor
from recommender import HeritageRecommender


@st.cache_resource
def load_system():
    """Load system (cached)."""
    try:
        processor = QueryProcessor()
        recommender = HeritageRecommender()
        return processor, recommender, None
    except Exception as e:
        return None, None, str(e)


def create_score_gauge(score, title, color):
    """Create a gauge chart for scores."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.33], 'color': 'rgba(255,0,0,0.1)'},
                {'range': [0.33, 0.67], 'color': 'rgba(255,255,0,0.1)'},
                {'range': [0.67, 1], 'color': 'rgba(0,255,0,0.1)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def render():
    """Render enhanced search page."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Heritage Document Search</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <strong>🎯 How It Works:</strong> Enter a natural language query about heritage documents. 
        Our AI system will understand your intent, extract relevant attributes, and recommend 
        documents using a hybrid approach combining <strong>graph structure</strong>, 
        <strong>entity importance</strong>, and <strong>semantic similarity</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    # Load system
    processor, recommender, error = load_system()
    
    if error:
        st.error(f"""
        **🚨 System Error:** {error}
        
        **Required Setup:**
        1. Run Horn's Index: `python src/4_knowledge_graph/horn_index.py`
        2. Verify KG exists: `knowledge_graph/heritage_kg.gpickle`
        """)
        return
    
    # ========== SEARCH INPUT SECTION ==========
    st.markdown("## 📝 Enter Your Search Query")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query_text = st.text_input(
            "Search Query",
            placeholder="e.g., 'Buddhist monasteries in ancient India', 'Mughal palaces', 'Temples in South India'...",
            label_visibility="collapsed",
            key="main_query"
        )
    
    with col2:
        top_k = st.number_input(
            "Results", 
            min_value=5, 
            max_value=50, 
            value=10, 
            step=5,
            help="Number of recommendations to return"
        )
    
    # ========== EXAMPLE QUERIES ==========
    st.markdown("### 💡 Try These Examples")
    
    example_cols = st.columns(4)
    
    examples = [
        "🕌 Mughal architecture",
        "🏯 Ancient Buddhist monasteries",
        "⛰️ Forts in Rajasthan",
        "🛕 Dravidian temples"
    ]
    
    for i, example in enumerate(examples):
        if example_cols[i].button(example, key=f"ex_{i}", use_container_width=True):
            query_text = example.split(" ", 1)[1]  # Remove emoji
            st.session_state.main_query = query_text
            st.rerun()
    
    st.markdown("---")
    
    # ========== ADVANCED OPTIONS ==========
    with st.expander("⚙️ Advanced Options", expanded=False):
        tab1, tab2 = st.tabs(["🎚️ Score Weights", "🔍 Filters"])
        
        with tab1:
            st.markdown("**Customize Hybrid Scoring:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                simrank_weight = st.slider(
                    "🕸️ SimRank (Graph)",
                    0.0, 1.0, 0.4, 0.05,
                    help="Weight for graph structure similarity"
                )
            
            with col2:
                horn_weight = st.slider(
                    "⭐ Horn's Index (Importance)",
                    0.0, 1.0, 0.3, 0.05,
                    help="Weight for entity importance"
                )
            
            with col3:
                embedding_weight = st.slider(
                    "🧠 Embeddings (Semantic)",
                    0.0, 1.0, 0.3, 0.05,
                    help="Weight for semantic similarity"
                )
            
            total = simrank_weight + horn_weight + embedding_weight
            if abs(total - 1.0) > 0.01:
                st.warning(f"⚠️ Weights sum to {total:.2f}, will normalize to 1.0")
        
        with tab2:
            st.markdown("**Filter Results:**")
            
            fcol1, fcol2 = st.columns(2)
            
            with fcol1:
                filter_period = st.multiselect(
                    "Time Period",
                    ["ancient", "medieval", "modern"]
                )
                
                filter_region = st.multiselect(
                    "Region",
                    ["north", "south", "east", "west", "central"]
                )
            
            with fcol2:
                filter_heritage = st.multiselect(
                    "Heritage Type",
                    ["temple", "fort", "palace", "monastery", "mosque", "stupa"]
                )
                
                filter_domain = st.multiselect(
                    "Domain",
                    ["religious", "military", "royal", "cultural"]
                )
    
    # ========== SEARCH BUTTON ==========
    search_clicked = st.button("🔍 Search Now", type="primary", use_container_width=True)
    
    if not search_clicked and not query_text:
        st.info("👆 Enter a query above to start searching")
        return
    
    if query_text:
        # ========== QUERY PARSING ==========
        with st.spinner("🔍 Parsing your query..."):
            parsed_query = processor.parse_query(query_text)
        
        # Display parsed attributes
        st.markdown("---")
        st.markdown("## 🔎 Query Understanding")
        
        attr_col1, attr_col2, attr_col3, attr_col4 = st.columns(4)
        
        with attr_col1:
            if parsed_query['heritage_types']:
                st.markdown(f"""
                <div class='result-card animated-card'>
                    <strong>🏛️ Heritage Types</strong><br>
                    {', '.join(parsed_query['heritage_types'])}
                </div>
                """, unsafe_allow_html=True)
        
        with attr_col2:
            if parsed_query['domains']:
                st.markdown(f"""
                <div class='result-card animated-card'>
                    <strong>🎯 Domains</strong><br>
                    {', '.join(parsed_query['domains'])}
                </div>
                """, unsafe_allow_html=True)
        
        with attr_col3:
            if parsed_query['time_period']:
                st.markdown(f"""
                <div class='result-card animated-card'>
                    <strong>⏳ Period</strong><br>
                    {parsed_query['time_period'].title()}
                </div>
                """, unsafe_allow_html=True)
        
        with attr_col4:
            if parsed_query['region']:
                st.markdown(f"""
                <div class='result-card animated-card'>
                    <strong>📍 Region</strong><br>
                    {parsed_query['region'].title()}
                </div>
                """, unsafe_allow_html=True)
        
        # ========== RECOMMENDATIONS ==========
        st.markdown("---")
        
        with st.spinner(f"🤖 Finding top-{top_k} recommendations..."):
            # Update weights if customized
            if 'simrank_weight' in locals():
                total = simrank_weight + horn_weight + embedding_weight
                recommender.simrank_weight = simrank_weight / total
                recommender.horn_weight = horn_weight / total
                recommender.embedding_weight = embedding_weight / total
            
            recommendations = recommender.recommend(parsed_query, top_k=top_k, explain=True)
        
        # Store in session
        st.session_state.update({
            'query_text': query_text,
            'parsed_query': parsed_query,
            'recommendations': recommendations,
            'top_k': top_k
        })
        
        # Success message
        st.success(f"✅ Found {len(recommendations)} relevant documents!")
        
        # ========== RESULTS PREVIEW ==========
        st.markdown("## 📚 Top Recommendations")
        
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            with st.expander(
                f"**#{i}** {rec['title'][:80]}{'...' if len(rec['title']) > 80 else ''} — Score: {rec['hybrid_score']:.4f}",
                expanded=(i <= 2)
            ):
                # Metadata row
                meta_cols = st.columns(4)
                
                with meta_cols[0]:
                    if rec['metadata'].get('heritage_type'):
                        st.markdown(f"🏛️ **Type:** {rec['metadata']['heritage_type']}")
                
                with meta_cols[1]:
                    if rec['metadata'].get('domain'):
                        st.markdown(f"🎯 **Domain:** {rec['metadata']['domain']}")
                
                with meta_cols[2]:
                    if rec['metadata'].get('time_period'):
                        st.markdown(f"⏳ **Period:** {rec['metadata']['time_period']}")
                
                with meta_cols[3]:
                    if rec['metadata'].get('region'):
                        st.markdown(f"📍 **Region:** {rec['metadata']['region']}")
                
                # Score breakdown with gauges
                st.markdown("**Score Components:**")
                
                gauge_cols = st.columns(3)
                
                with gauge_cols[0]:
                    fig1 = create_score_gauge(
                        rec['component_scores']['simrank'],
                        "SimRank",
                        "#667eea"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with gauge_cols[1]:
                    fig2 = create_score_gauge(
                        rec['component_scores']['horn'],
                        "Horn's Index",
                        "#764ba2"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                with gauge_cols[2]:
                    fig3 = create_score_gauge(
                        rec['component_scores']['embedding'],
                        "Embedding",
                        "#f093fb"
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                # KG explanations
                if rec.get('kg_explanations'):
                    st.markdown("**🕸️ Why Recommended (Knowledge Graph):**")
                    for path in rec['kg_explanations'][:3]:
                        st.markdown(f'<div class="kg-path">{path}</div>', unsafe_allow_html=True)
        
        # ========== NAVIGATION ==========
        st.markdown("---")
        
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        
        with nav_col1:
            if st.button("📊 View All Results", use_container_width=True):
                st.switch_page("pages/results_page.py")
        
        with nav_col2:
            if st.button("🕸️ Explore Graph", use_container_width=True):
                st.switch_page("pages/kg_viz_page.py")
        
        with nav_col3:
            if st.button("📈 Check Performance", use_container_width=True):
                st.switch_page("pages/evaluation_page.py")


if __name__ == "__main__":
    render()