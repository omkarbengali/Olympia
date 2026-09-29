"""
OLYMPIA: Olympic Performance Analytics & Intelligence System
Main Application Entry Point.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from warehouse.warehouse import get_denormalized_medals, get_table_counts
from dashboard.components.filters import render_sidebar_filters

# Import Page Renderers
from dashboard.pages import (
    dashboard,
    medal_analysis,
    country_analysis,
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
# 2. Dark Analytics Custom CSS Styling
# ------------------------------------------------------------
st.markdown("""
<style>
    /* Main Background & Typography */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: rgba(11, 17, 32, 0.8);
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
    
    /* DataTables & DataFrames */
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


try:
    df_raw = load_olympic_data()
except Exception as e:
    st.error(f"Failed to load data from SQLite warehouse: {e}")
    st.info("Please initialize the database by running: `python -m etl.load`")
    st.stop()


# ------------------------------------------------------------
# 4. Sidebar Branding & Navigation
# ------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #334155; margin-bottom: 1rem;">
    <div style="font-size: 1.6rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 0.5rem;">
        <span>🏅</span> OLYMPIA
    </div>
    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem; letter-spacing: 0.04em; text-transform: uppercase;">
        Olympic Performance Intelligence System
    </div>
</div>
""", unsafe_allow_html=True)

PAGES = {
    "📊 Main Dashboard": dashboard.render_page,
    "🥇 Medal Analytics & Charts": medal_analysis.render_page,
    "🌍 Country Intelligence": country_analysis.render_page,
    "🏃 Sport & Gender Analysis": sport_analysis.render_page,
    "🧊 OLAP Cube Multi-Dimensional": olap_analysis.render_page,
    "🧹 Data Preprocessing & Quality": preprocessing.render_page,
    "📈 Linear Regression": regression.render_page,
    "🌲 Classification (Tree vs Bayes)": classification.render_page,
    "🔍 Clustering (K-Means & Hierarchical)": clustering.render_page,
    "🔗 Association Rules (Apriori)": association_rules.render_page,
    "🏛️ Star Schema Data Warehouse": warehouse.render_page,
}

selected_page_name = st.sidebar.radio(
    "Navigation Menu",
    options=list(PAGES.keys()),
    index=0
)

# ------------------------------------------------------------
# 5. Global Filters (Applied to Analytical & Exploratory Pages)
# ------------------------------------------------------------
# Pages that should operate on filtered data
FILTERABLE_PAGES = {
    "📊 Main Dashboard",
    "🥇 Medal Analytics & Charts",
    "🌍 Country Intelligence",
    "🏃 Sport & Gender Analysis",
    "🧊 OLAP Cube Multi-Dimensional"
}

if selected_page_name in FILTERABLE_PAGES:
    filtered_df, filter_state = render_sidebar_filters(df_raw)
else:
    # ML and Quality pages require the full dataset to maintain sample sizes and distributions
    filtered_df = df_raw.copy()
    st.sidebar.caption("⚡ *Full warehouse dataset active for this model experiment.*")

# Sidebar System Metadata
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
    <strong>Architecture:</strong> Python 3.14 • SQLite3<br>
    <strong>Design:</strong> Star Schema (5 Dims, 1 Fact)<br>
    <strong>Integrity:</strong> 20,240 Verified Facts<br>
    <strong>Coverage:</strong> 1896 Athens – 2024 Paris
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 6. Render Active Page
# ------------------------------------------------------------
page_func = PAGES[selected_page_name]
page_func(filtered_df)
