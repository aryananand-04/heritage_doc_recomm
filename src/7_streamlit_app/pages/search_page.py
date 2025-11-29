"""
Search Interface Page

Features:
- Natural language query input
- Heritage filters (type, domain, period, region, style)
- Top-K selector
- Real-time recommendations
"""

import streamlit as st
import sys
from pathlib import Path

# Add query system to path
sys.path.append(str(Path(__file__).parent.parent.parent / '6_query_system'))

from query_processor import QueryProcessor
from recommender import HeritageRecommender


@st.cache_resource
def load_system():
    """Load query processor and recommender (cached)."""
    processor = QueryProcessor()
    recommender = HeritageRecommender()
    return processor, recommender


def render():
    """Render search interface page."""
    st.markdown('<h1 class="main-header">🔍 Heritage Document Search</h1>', unsafe_allow_html=True)

    st.markdown("""
    Search for heritage documents using natural language queries.
    The system will extract heritage attributes and return relevant recommendations
    using hybrid scoring (SimRank + Horn's Index + Embeddings).
    """)

    # Load system
    try:
        processor, recommender = load_system()
    except Exception as e:
        st.error(f"Failed to load recommendation system: {str(e)}")
        st.info("Make sure you have run:\n- `python src/6_query_system/horn_index.py` to generate Horn weights")
        return

    # Query input
    st.markdown("### 📝 Enter Your Query")

    col1, col2 = st.columns([3, 1])

    with col1:
        query_text = st.text_input(
            "Search Query",
            placeholder="e.g., Mughal temples in North India, Ancient forts in Rajasthan...",
            label_visibility="collapsed"
        )

    with col2:
        top_k = st.number_input("Top-K Results", min_value=1, max_value=50, value=10, step=1)

    # Example queries
    st.markdown("**Example Queries:**")
    example_cols = st.columns(3)

    examples = [
        "Mughal temples in North India",
        "Ancient forts in Rajasthan",
        "Buddhist stupas and monasteries"
    ]

    for i, example in enumerate(examples):
        if example_cols[i].button(example, key=f"example_{i}"):
            query_text = example
            st.rerun()

    # Advanced filters (optional)
    with st.expander("🔧 Advanced Filters"):
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            heritage_types = st.multiselect(
                "Heritage Types",
                options=["monument", "site", "artifact", "architecture", "tradition", "art",
                        "temple", "fort", "palace", "mosque", "church"],
                help="Filter by heritage type"
            )

            domains = st.multiselect(
                "Domains",
                options=["religious", "military", "royal", "cultural", "archaeological", "architectural"],
                help="Filter by domain"
            )

        with filter_col2:
            time_period = st.selectbox(
                "Time Period",
                options=["Any", "ancient", "medieval", "modern"],
                help="Filter by time period"
            )

            region = st.selectbox(
                "Region",
                options=["Any", "north", "south", "east", "west", "central"],
                help="Filter by region"
            )

        with filter_col3:
            architectural_styles = st.multiselect(
                "Architectural Styles",
                options=["indo-islamic", "mughal", "dravidian", "nagara", "vesara",
                        "buddhist", "colonial", "rajput"],
                help="Filter by architectural style"
            )

            score_weights = st.checkbox("Customize Score Weights", value=False)

    # Score weight customization
    if score_weights:
        st.markdown("**Score Component Weights:**")
        weight_col1, weight_col2, weight_col3 = st.columns(3)

        with weight_col1:
            simrank_weight = st.slider("SimRank Weight", 0.0, 1.0, 0.4, 0.05)
        with weight_col2:
            horn_weight = st.slider("Horn's Index Weight", 0.0, 1.0, 0.3, 0.05)
        with weight_col3:
            embedding_weight = st.slider("Embedding Weight", 0.0, 1.0, 0.3, 0.05)

        total_weight = simrank_weight + horn_weight + embedding_weight
        if total_weight != 1.0:
            st.warning(f"Weights sum to {total_weight:.2f}, will be normalized to 1.0")
    else:
        simrank_weight, horn_weight, embedding_weight = 0.4, 0.3, 0.3

    # Search button
    st.markdown("---")

    if st.button("🔍 Search", type="primary", use_container_width=True) or query_text:
        if not query_text:
            st.warning("Please enter a search query")
            return

        # Parse query
        with st.spinner("Parsing query..."):
            parsed_query = processor.parse_query(query_text)

        # Display parsed query
        st.markdown("### 🔎 Query Analysis")

        analysis_cols = st.columns(4)

        with analysis_cols[0]:
            if parsed_query['heritage_types']:
                st.info(f"**Heritage Types:**\n{', '.join(parsed_query['heritage_types'])}")
            if parsed_query['domains']:
                st.info(f"**Domains:**\n{', '.join(parsed_query['domains'])}")

        with analysis_cols[1]:
            if parsed_query['time_period']:
                st.info(f"**Time Period:**\n{parsed_query['time_period']}")
            if parsed_query['region']:
                st.info(f"**Region:**\n{parsed_query['region']}")

        with analysis_cols[2]:
            if parsed_query['architectural_styles']:
                st.info(f"**Styles:**\n{', '.join(parsed_query['architectural_styles'])}")
            if parsed_query['locations']:
                st.info(f"**Locations:**\n{', '.join(parsed_query['locations'])}")

        with analysis_cols[3]:
            if parsed_query['persons']:
                st.info(f"**Persons:**\n{', '.join(parsed_query['persons'])}")
            if parsed_query['organizations']:
                st.info(f"**Organizations:**\n{', '.join(parsed_query['organizations'])}")

        # Apply manual filters (override parsed query)
        if heritage_types:
            parsed_query['heritage_types'] = set(heritage_types)
        if domains:
            parsed_query['domains'] = set(domains)
        if time_period != "Any":
            parsed_query['time_period'] = time_period
        if region != "Any":
            parsed_query['region'] = region
        if architectural_styles:
            parsed_query['architectural_styles'] = set(architectural_styles)

        # Get recommendations
        with st.spinner(f"Finding top-{top_k} recommendations..."):
            # Update recommender weights if customized
            if score_weights:
                recommender.simrank_weight = simrank_weight / total_weight
                recommender.horn_weight = horn_weight / total_weight
                recommender.embedding_weight = embedding_weight / total_weight

            recommendations = recommender.recommend(parsed_query, top_k=top_k, explain=True)

        # Store results in session state for other pages
        st.session_state['query_text'] = query_text
        st.session_state['parsed_query'] = parsed_query
        st.session_state['recommendations'] = recommendations
        st.session_state['top_k'] = top_k

        # Display results summary
        st.markdown("---")
        st.markdown(f"### 📊 Found {len(recommendations)} Results")

        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"#{i} {rec['title']} — Score: {rec['hybrid_score']:.4f}", expanded=(i <= 3)):
                # Metadata
                meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)

                with meta_col1:
                    if rec['metadata'].get('heritage_type'):
                        st.markdown(f"**Type:** {rec['metadata']['heritage_type']}")

                with meta_col2:
                    if rec['metadata'].get('domain'):
                        st.markdown(f"**Domain:** {rec['metadata']['domain']}")

                with meta_col3:
                    if rec['metadata'].get('time_period'):
                        st.markdown(f"**Period:** {rec['metadata']['time_period']}")

                with meta_col4:
                    if rec['metadata'].get('region'):
                        st.markdown(f"**Region:** {rec['metadata']['region']}")

                # Score breakdown
                st.markdown("**Score Breakdown:**")
                score_data = {
                    "Component": ["SimRank", "Horn's Index", "Embedding", "**Total**"],
                    "Score": [
                        f"{rec['component_scores']['simrank']:.4f}",
                        f"{rec['component_scores']['horn']:.4f}",
                        f"{rec['component_scores']['embedding']:.4f}",
                        f"**{rec['hybrid_score']:.4f}**"
                    ],
                    "Weight": [
                        f"{recommender.simrank_weight:.2f}",
                        f"{recommender.horn_weight:.2f}",
                        f"{recommender.embedding_weight:.2f}",
                        "1.00"
                    ],
                    "Weighted": [
                        f"{recommender.simrank_weight * rec['component_scores']['simrank']:.4f}",
                        f"{recommender.horn_weight * rec['component_scores']['horn']:.4f}",
                        f"{recommender.embedding_weight * rec['component_scores']['embedding']:.4f}",
                        f"**{rec['hybrid_score']:.4f}**"
                    ]
                }

                st.table(score_data)

                # KG explanations
                if rec.get('kg_explanations'):
                    st.markdown("**Why Recommended (Knowledge Graph Paths):**")
                    for path in rec['kg_explanations'][:3]:
                        st.markdown(f'<div class="kg-path">• {path}</div>', unsafe_allow_html=True)

        # Quick navigation
        st.markdown("---")
        nav_col1, nav_col2 = st.columns(2)

        with nav_col1:
            if st.button("📊 View Detailed Results", use_container_width=True):
                st.switch_page("pages/results_page.py")

        with nav_col2:
            if st.button("🕸️ Visualize Knowledge Graph", use_container_width=True):
                st.switch_page("pages/kg_viz_page.py")


if __name__ == "__main__":
    render()
