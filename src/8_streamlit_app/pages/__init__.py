"""Pages package for Streamlit app."""
"""
Pages package for Heritage Document Recommendation System Streamlit App

Available pages:
- home_page: Dashboard with system overview
- search_page: Query interface with recommendations
- kg_viz_page: Interactive knowledge graph visualization
- results_page: Detailed results and explanations
- evaluation_page: Performance metrics and comparisons
- about_page: System information and methodology
"""

# Import all page modules for easy access
from . import home_page
from . import search_page
from . import kg_viz_page
from . import results_page
from . import evaluation_page
from . import about_page

__all__ = [
    'home_page',
    'search_page',
    'kg_viz_page',
    'results_page',
    'evaluation_page',
    'about_page'
]

__version__ = '2.0.0'
__author__ = 'Akchhya Singh'