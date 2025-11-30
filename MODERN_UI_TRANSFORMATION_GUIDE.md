# Heritage Document Recommender - Modern UI Transformation Guide

## 🎨 Color Palette (2025 Modern Theme)

- **Primary (Ocean Blue)**: `#9CC6DB` - Used for primary actions, KG nodes, trust indicators
- **Secondary (Cream)**: `#FCF6D9` - Background gradients, secondary badges
- **Accent (Burnt Orange)**: `#CF4B00` - Call-to-action buttons, important metrics, headings
- **Gold**: `#DDBA7D` - Horn's Index visualization, premium features, highlights

## ✅ Completed Transformations

### 1. **Custom CSS Framework (`src/8_streamlit_app/style.css`)**

**Features Implemented:**
- ✅ Modern font system (Inter font family)
- ✅ Neumorphism & Glassmorphism effects
- ✅ Smooth animations (fade-in, slide-in, pulse, shimmer)
- ✅ Custom button styles with gradients
- ✅ Enhanced input fields with focus effects
- ✅ Modern card designs with hover effects
- ✅ Score visualization bars with gradients
- ✅ KG path styling with custom formatting
- ✅ Responsive design (mobile-friendly)
- ✅ Custom scrollbar styling
- ✅ Badge/tag system
- ✅ Loading spinners
- ✅ Tooltips
- ✅ Info/Success/Warning/Error boxes

**CSS Classes Available:**
```css
/* Cards */
.glass-card                 /* Glassmorphism card */
.result-card                /* Search result card */
.metric-card                /* Metric display card */
.feature-card               /* Feature highlight card */

/* Badges */
.badge-primary             /* Ocean blue badge */
.badge-secondary           /* Cream badge with orange border */
.badge-accent              /* Burnt orange badge */
.badge-gold                /* Gold badge */

/* Animations */
.fade-in                   /* Fade in animation */
.slide-in-left             /* Slide from left */
.slide-in-right            /* Slide from right */
.pulse                     /* Pulsing animation */
.shimmer                   /* Shimmer loading effect */

/* Score Visualization */
.score-container           /* Score display container */
.score-bar                 /* Score progress bar */
.score-fill-simrank        /* SimRank score gradient */
.score-fill-horn           /* Horn's Index score gradient */
.score-fill-embedding      /* Embedding score gradient */
.score-fill-hybrid         /* Hybrid score gradient */

/* Knowledge Graph */
.kg-path                   /* KG path display box */
.kg-path-node              /* Individual KG node */

/* Info Boxes */
.info-box                  /* Blue informational box */
.success-box               /* Green success box */
.warning-box               /* Yellow/gold warning box */
.error-box                 /* Orange error box */

/* Hero Section */
.hero-section              /* Homepage hero section */
.hero-title                /* Large hero title */
.hero-subtitle             /* Hero subtitle */
.hero-description          /* Hero description text */

/* Sections */
.section-header            /* Section heading with bottom border */
```

### 2. **Modernized Home Page (`src/8_streamlit_app/pages/home_page.py`)**

**Implemented Features:**
- ✅ Hero section with gradient background
- ✅ Animated metric cards (4 stats with icons)
- ✅ Feature showcase cards (3 columns)
- ✅ Performance gauge charts (Precision & Coverage)
- ✅ Quick action buttons
- ✅ System architecture overview
- ✅ Performance highlights boxes
- ✅ Example queries showcase
- ✅ All using new color palette
- ✅ Smooth animations on load

**Key Metrics Displayed:**
- 📚 Total Documents: 369
- 🕸️ KG Nodes: 500+
- 🎯 Precision@5: 82.8%
- ⚡ Latency: <0.3ms

## 🚀 Pages to Update Next

### 3. **Search Page** (`src/8_streamlit_app/pages/search_page.py`)

**Required Modifications:**

```python
def render():
    from pathlib import Path

    # Load CSS
    css_path = Path(__file__).parent.parent / 'style.css'
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Modern search header
    st.markdown("""
        <div class='hero-section fade-in' style='padding: 40px 20px;'>
            <h1 class='hero-title' style='font-size: 2.5rem;'>🔍 Search Heritage Documents</h1>
            <p class='hero-subtitle' style='font-size: 1.2rem;'>
                Enter natural language queries about heritage sites, monuments, and cultural artifacts
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Search input with modern styling
    col1, col2 = st.columns([4, 1])

    with col1:
        query_text = st.text_input(
            "Search Query",
            placeholder="e.g., 'Buddhist monasteries in ancient India', 'Mughal palaces'...",
            label_visibility="collapsed",
            key="main_query"
        )

    with col2:
        top_k = st.number_input("Results", min_value=5, max_value=50, value=10,
                                label_visibility="collapsed")

    # Example queries as chips
    st.markdown("**💡 Try these examples:**")
    examples = [
        "Mughal architecture",
        "Buddhist monasteries",
        "Forts in Rajasthan",
        "Dravidian temples"
    ]

    cols = st.columns(4)
    for i, (col, example) in enumerate(zip(cols, examples)):
        with col:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state.main_query = example
                st.rerun()

    # Advanced options in expander
    with st.expander("⚙️ Advanced Options", expanded=False):
        st.markdown("#### 🎚️ Score Weights")
        col1, col2, col3 = st.columns(3)
        with col1:
            simrank_w = st.slider("🕸️ SimRank", 0.0, 1.0, 0.4, 0.05)
        with col2:
            horn_w = st.slider("⭐ Horn's Index", 0.0, 1.0, 0.3, 0.05)
        with col3:
            embedding_w = st.slider("🧠 Embeddings", 0.0, 1.0, 0.3, 0.05)

    # Search button
    if st.button("🔍 Search Now", type="primary", use_container_width=True):
        # ... your existing search logic ...

        # Display results with modern cards
        for i, rec in enumerate(recommendations[:10], 1):
            st.markdown(f"""
                <div class='result-card fade-in' style='animation-delay: {i*0.1}s;'>
                    <div style='display: flex; justify-content: space-between; align-items: start;'>
                        <div style='flex: 1;'>
                            <h3 style='color: #CF4B00; margin: 0;'>
                                #{i} {rec['title']}
                            </h3>
                            <div style='margin: 12px 0;'>
                                <span class='badge badge-primary'>{rec['metadata'].get('heritage_type', 'N/A')}</span>
                                <span class='badge badge-secondary'>{rec['metadata'].get('domain', 'N/A')}</span>
                            </div>
                        </div>
                        <div style='font-size: 1.8rem; font-weight: 700; color: #CF4B00;'>
                            {rec['hybrid_score']:.3f}
                        </div>
                    </div>

                    <div style='margin-top: 20px;'>
                        <div style='font-weight: 600; color: #666; margin-bottom: 8px;'>Score Breakdown:</div>

                        <div class='score-container'>
                            <div class='score-label' style='color: #9CC6DB;'>SimRank</div>
                            <div class='score-bar'>
                                <div class='score-fill score-fill-simrank'
                                     style='width: {rec['component_scores']['simrank']*100}%;'></div>
                            </div>
                            <div class='score-value'>{rec['component_scores']['simrank']:.3f}</div>
                        </div>

                        <div class='score-container'>
                            <div class='score-label' style='color: #DDBA7D;'>Horn's Index</div>
                            <div class='score-bar'>
                                <div class='score-fill score-fill-horn'
                                     style='width: {rec['component_scores']['horn']*100}%;'></div>
                            </div>
                            <div class='score-value'>{rec['component_scores']['horn']:.3f}</div>
                        </div>

                        <div class='score-container'>
                            <div class='score-label' style='color: #CF4B00;'>Embedding</div>
                            <div class='score-bar'>
                                <div class='score-fill score-fill-embedding'
                                     style='width: {rec['component_scores']['embedding']*100}%;'></div>
                            </div>
                            <div class='score-value'>{rec['component_scores']['embedding']:.3f}</div>
                        </div>
                    </div>

                    {f"<div class='kg-path' style='margin-top: 16px;'>" +
                     f"<strong>🕸️ Knowledge Graph Path:</strong><br>" +
                     f"{rec['kg_explanations'][0]}</div>"
                     if rec.get('kg_explanations') else ""}
                </div>
            """, unsafe_allow_html=True)
```

### 4. **Results Page** (`src/8_streamlit_app/pages/results_page.py`)

**Plotly Chart Theme:**

```python
# Standard chart configuration
CHART_THEME = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': {'family': 'Inter', 'size': 12, 'color': '#666'},
    'title_font': {'size': 18, 'color': '#CF4B00', 'family': 'Inter', 'weight': 700},
    'colorway': ['#9CC6DB', '#CF4B00', '#DDBA7D', '#FCF6D9']
}

def create_score_distribution_chart(recommendations):
    """Create modern score distribution histogram."""
    fig = go.Figure()

    scores = [rec['hybrid_score'] for rec in recommendations]

    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color='#CF4B00',
        marker_line_color='white',
        marker_line_width=2,
        opacity=0.85,
        name='Hybrid Score'
    ))

    fig.update_layout(
        **CHART_THEME,
        title="Score Distribution",
        xaxis_title="Hybrid Score",
        yaxis_title="Frequency",
        height=400,
        showlegend=False
    )

    return fig

def create_component_comparison_chart(recommendations):
    """Create stacked bar chart for component scores."""
    fig = go.Figure()

    titles = [rec['title'][:30] + '...' for rec in recommendations[:10]]

    fig.add_trace(go.Bar(
        name='SimRank',
        x=titles,
        y=[rec['component_scores']['simrank'] for rec in recommendations[:10]],
        marker_color='#9CC6DB',
        marker_line_color='white',
        marker_line_width=1.5
    ))

    fig.add_trace(go.Bar(
        name="Horn's Index",
        x=titles,
        y=[rec['component_scores']['horn'] for rec in recommendations[:10]],
        marker_color='#DDBA7D',
        marker_line_color='white',
        marker_line_width=1.5
    ))

    fig.add_trace(go.Bar(
        name='Embedding',
        x=titles,
        y=[rec['component_scores']['embedding'] for rec in recommendations[:10]],
        marker_color='#CF4B00',
        marker_line_color='white',
        marker_line_width=1.5
    ))

    fig.update_layout(
        **CHART_THEME,
        title="Top 10 Component Scores",
        barmode='stack',
        xaxis_tickangle=-45,
        xaxis_title="Documents",
        yaxis_title="Score",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def render():
    # Load CSS
    from pathlib import Path
    css_path = Path(__file__).parent.parent / 'style.css'
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Query summary card
    if 'query_text' in st.session_state:
        st.markdown(f"""
            <div class='glass-card fade-in' style='
                background: linear-gradient(135deg, #FCF6D9 0%, #9CC6DB 100%);
                border: none;
            '>
                <h2 style='color: #CF4B00; margin: 0;'>
                    🔍 Query: "{st.session_state['query_text']}"
                </h2>
                <p style='margin-top: 12px; color: #666; font-size: 1.1rem;'>
                    Found <strong style='color: #CF4B00;'>{len(st.session_state.get('recommendations', []))}</strong> results
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    st.markdown("### 📊 Visual Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = create_score_distribution_chart(st.session_state.get('recommendations', []))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_component_comparison_chart(st.session_state.get('recommendations', []))
        st.plotly_chart(fig, use_container_width=True)
```

### 5. **Main App** (`src/8_streamlit_app/streamlit_app.py`)

**Enhanced Sidebar:**

```python
# After st.set_page_config()

# Load CSS globally
from pathlib import Path
css_path = Path(__file__).parent / 'style.css'
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Modern sidebar
with st.sidebar:
    # Logo/Header
    st.markdown("""
        <div style='text-align: center; padding: 30px 20px 20px 20px;'>
            <div style='font-size: 4rem; margin-bottom: 12px;'>🏛️</div>
            <h1 style='color: #CF4B00; margin: 8px 0; font-size: 1.8rem; font-weight: 800;'>Heritage</h1>
            <p style='color: #666; font-size: 0.95rem; margin: 4px 0;'>Document Recommender</p>
            <p style='color: #9CC6DB; font-size: 0.8rem; margin: 8px 0;'>v2.0 • 2025 Edition</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    st.markdown("### 🗺️ Navigation")
    page = st.radio(
        "Go to:",
        [
            "🏠 Home Dashboard",
            "🔍 Search Documents",
            "🕸️ Knowledge Graph",
            "📊 Results & Analysis",
            "📈 Evaluation Metrics",
            "ℹ️ About System"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", "369", delta=None)
    with col2:
        st.metric("Precision", "82.8%", delta="+5%")

    st.markdown("---")

    # Help section
    with st.expander("❓ Need Help?", expanded=False):
        st.markdown("""
            **Quick Start:**
            1. Go to **Search** page
            2. Enter a heritage query
            3. View recommendations
            4. Explore **Knowledge Graph**
            5. Check **Evaluation** metrics

            **Support:** akchhya1108@gmail.com
        """)
```

## 📋 Implementation Checklist

### Immediate Tasks:
- [x] Create `style.css` with modern framework
- [x] Update `home_page.py` with new design
- [ ] Update `search_page.py` with modern search interface
- [ ] Update `results_page.py` with Plotly charts
- [ ] Update `kg_viz_page.py` with modern styling
- [ ] Update `evaluation_page.py` with modern metrics
- [ ] Update `about_page.py` with modern layout
- [ ] Update `streamlit_app.py` with enhanced sidebar

### Testing:
- [ ] Test all pages load correctly
- [ ] Test CSS animations work smoothly
- [ ] Test responsive design on mobile
- [ ] Test color accessibility (contrast ratios)
- [ ] Test all interactive elements
- [ ] Test Plotly charts render properly

## 🎨 Color Usage Guide

### Primary (#9CC6DB - Ocean Blue)
**Use for:**
- SimRank score visualizations
- Primary badges
- Trust indicators
- KG node colors
- Secondary buttons
- Border accents

**Example:**
```css
background: #9CC6DB;
border-color: #9CC6DB;
color: #9CC6DB;
```

### Secondary (#FCF6D9 - Cream)
**Use for:**
- Background gradients
- Secondary badges (with accent border)
- Soft backgrounds
- Card backgrounds (with opacity)

**Example:**
```css
background: linear-gradient(135deg, #FCF6D9 0%, #9CC6DB 100%);
background: rgba(252, 246, 217, 0.3);
```

### Accent (#CF4B00 - Burnt Orange)
**Use for:**
- Primary CTA buttons
- Headings and titles
- Important metrics
- Embedding score bars
- Hover states
- Active states

**Example:**
```css
background: linear-gradient(135deg, #CF4B00 0%, #DDBA7D 100%);
color: #CF4B00;
border-left: 4px solid #CF4B00;
```

### Gold (#DDBA7D)
**Use for:**
- Horn's Index visualizations
- Premium features
- Gold badges
- Highlight accents
- Secondary gradients

**Example:**
```css
background: #DDBA7D;
color: #DDBA7D;
```

## 🚀 Running the Modernized App

```bash
cd src/8_streamlit_app
streamlit run streamlit_app.py
```

## 📱 Responsive Breakpoints

- **Desktop**: > 768px (full features)
- **Tablet**: 481px - 768px (adjusted layouts)
- **Mobile**: < 480px (stacked layouts)

## ✨ Animation Timing

- **Fade In**: 0.6s ease-out
- **Slide In**: 0.5s ease-out
- **Hover Effects**: 0.3s cubic-bezier
- **Button Transforms**: 0.3s ease
- **Score Bars**: 0.8s cubic-bezier (delayed animation)

## 🔧 Troubleshooting

### CSS Not Loading
```python
# Ensure this at the top of each page
from pathlib import Path
css_path = Path(__file__).parent.parent / 'style.css'
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
```

### Animations Not Working
- Clear Streamlit cache: `streamlit cache clear`
- Hard refresh browser: Ctrl+Shift+R (Windows) / Cmd+Shift+R (Mac)

### Colors Not Showing
- Check HTML is using `unsafe_allow_html=True`
- Verify hex codes are correct
- Check CSS class names match

## 📞 Support

**Developer:** Akchhya Singh
**Email:** akchhya1108@gmail.com
**Version:** 2.0 (2025 Modern Edition)

---

**Built with ❤️ using Streamlit, Plotly, and modern web design principles**
