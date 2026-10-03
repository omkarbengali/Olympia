"""
OLYMPIA: Olympic Performance Analytics & Intelligence System
Main Application Entry Point.
Implements 3 Role-Based Experiences (Admin, Sport Enthusiast, Player/Academy),
Star Schema Data Warehousing, OLAP Cube operations, and Machine Learning models.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from warehouse.warehouse import get_denormalized_medals
from dashboard.auth import (
    init_auth,
    is_authenticated,
    get_current_role,
    get_current_user,
    render_landing_page,
    render_sidebar_user_badge,
    ROLE_ADMIN,
    ROLE_ENTHUSIAST,
    ROLE_ACADEMY
)

# Import Page Renderers
from dashboard.pages import (
    admin_dashboard,
    enthusiast_dashboard,
    academy_dashboard,
    dashboard,
    country_analysis,
    medal_analysis,
    sport_analysis,
    olap_analysis,
    preprocessing,
    regression,
    classification,
    clustering,
    association_rules,
    warehouse,
)

# ------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="OLYMPIA - Olympic Intelligence System",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 2. Polished Dark Analytics Theme Styling
# ------------------------------------------------------------
st.markdown("""
<style>
    /* Main Background & Clean Font */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: rgba(11, 17, 32, 0.85);
        backdrop-filter: blur(8px);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Headings */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    button[aria-selected="true"] {
        color: #38bdf8 !important;
        background-color: #1e293b !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
    }
    
    /* Tables & DataFrames */
    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 3. Cached Data Loading
# ------------------------------------------------------------
@st.cache_data(show_spinner="Connecting to SQLite Star Schema...")
def load_olympic_data() -> pd.DataFrame:
    """Loads denormalized medal facts from SQLite Star Schema."""
    return get_denormalized_medals()


# ------------------------------------------------------------
# 4. Authentication & Landing Portal Router
# ------------------------------------------------------------
init_auth()

if not is_authenticated():
    # Show the 3-Role Landing / Login Selection Portal
    login_page = st.Page(render_landing_page, title="Sign In", icon="🏅")
    nav = st.navigation([login_page], position="hidden")
    nav.run()
    st.stop()


# ------------------------------------------------------------
# 5. Role-Based Navigation Routing
# ------------------------------------------------------------
user_role = get_current_role()


def make_page(render_func):
    """Wraps a page renderer so it receives loaded warehouse data."""
    def page_entry():
        df = load_olympic_data()
        render_func(df)
    page_entry.__name__ = render_func.__name__
    return page_entry


# Sidebar Branding
st.sidebar.markdown("""
<div style="padding: 0.25rem 0 0.75rem 0; border-bottom: 1px solid #334155; margin-bottom: 0.75rem;">
    <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 0.5rem;">
        <span>🏅</span> OLYMPIA
    </div>
    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.15rem; letter-spacing: 0.04em; text-transform: uppercase;">
        Olympic Intelligence Platform
    </div>
</div>
""", unsafe_allow_html=True)

# User Badge and Logout Button
render_sidebar_user_badge()

# Build Role-Specific Page Dictionaries
if user_role == ROLE_ADMIN:
    # Full Administrator Experience
    nav_pages = {
        "🏛️ EXECUTIVE": [
            st.Page(make_page(admin_dashboard.render_page), title="Admin Dashboard", icon="🛡️", url_path="admin_overview", default=True),
        ],
        "MAIN": [
            st.Page(make_page(dashboard.render_page), title="Olympic Overview", icon="📊", url_path="olympic_overview"),
            st.Page(make_page(country_analysis.render_page), title="Country Intelligence", icon="🌍", url_path="country_intel"),
            st.Page(make_page(medal_analysis.render_page), title="Medal Analytics", icon="🥇", url_path="medal_analytics"),
            st.Page(make_page(sport_analysis.render_page), title="Sport Intelligence", icon="🏃", url_path="sport_intel"),
        ],
        "ANALYTICS & ML": [
            st.Page(make_page(olap_analysis.render_page), title="OLAP Multi-Dimensional", icon="🧊", url_path="olap_cube"),
            st.Page(make_page(regression.render_page), title="Linear Regression", icon="📈", url_path="linear_reg"),
            st.Page(make_page(classification.render_page), title="Classification Models", icon="🌲", url_path="classification_lab"),
            st.Page(make_page(clustering.render_page), title="Clustering Lab", icon="🔍", url_path="clustering_lab"),
            st.Page(make_page(association_rules.render_page), title="Sport Associations", icon="🔗", url_path="apriori_rules"),
        ],
        "TECHNICAL": [
            st.Page(make_page(warehouse.render_page), title="Star Schema Warehouse", icon="🗄️", url_path="star_schema_wh"),
            st.Page(make_page(preprocessing.render_page), title="Data Quality & ETL", icon="🧹", url_path="data_quality_lab"),
        ]
    }

elif user_role == ROLE_ENTHUSIAST:
    # Visual, Intuitive Fan Zone (Zero database jargon)
    nav_pages = {
        "OLYMPIC FAN ZONE": [
            st.Page(make_page(enthusiast_dashboard.render_page), title="Fan Dashboard", icon="🎉", url_path="fan_home", default=True),
            st.Page(make_page(country_analysis.render_page), title="Country Explorer", icon="🌍", url_path="fan_countries"),
            st.Page(make_page(sport_analysis.render_page), title="Sport Explorer", icon="🏃", url_path="fan_sports"),
            st.Page(make_page(medal_analysis.render_page), title="Medal Highlights", icon="🏅", url_path="fan_medals"),
            st.Page(make_page(country_analysis.render_comparison_page), title="Head-to-Head Showdown", icon="⚔️", url_path="fan_head_to_head"),
        ]
    }

elif user_role == ROLE_ACADEMY:
    # High-Performance Professional Player & Academy Experience
    nav_pages = {
        "PERFORMANCE INTELLIGENCE": [
            st.Page(make_page(academy_dashboard.render_page), title="Performance Overview", icon="🎯", url_path="academy_home", default=True),
            st.Page(make_page(sport_analysis.render_page), title="Discipline & Event Depth", icon="🏃", url_path="academy_sports"),
            st.Page(make_page(country_analysis.render_page), title="National Trajectory", icon="🌍", url_path="academy_countries"),
            st.Page(make_page(country_analysis.render_comparison_page), title="Competitor Comparison", icon="⚔️", url_path="academy_comparison"),
            st.Page(make_page(clustering.render_page), title="Performance Clusters & Tiers", icon="🧩", url_path="academy_clusters"),
            st.Page(make_page(regression.render_page), title="Historical Estimate Baseline", icon="🔮", url_path="academy_regression"),
        ]
    }

else:
    # Fallback default
    nav_pages = {
        "MAIN": [
            st.Page(make_page(dashboard.render_page), title="Dashboard", icon="📊", url_path="default_dash", default=True),
        ]
    }

# Render Navigation
pg = st.navigation(nav_pages)
pg.run()

# Sidebar Metadata
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.72rem; color: #64748b; line-height: 1.4;">
    <strong>Architecture:</strong> Python • SQLite3 Star Schema<br>
    <strong>Verified Facts:</strong> 20,240 records (1896–2024)<br>
    <strong>System:</strong> OLYMPIA Analytics & Intelligence
</div>
""", unsafe_allow_html=True)
