"""
OLYMPIA ⚡ | Olympic Data Warehouse & Mining Intelligence
High-Contrast Modern Dark-Themed Olympic Analytics Platform.
Features:
- Mandatory 3-Role Authentication Flow (Admin, Sport Enthusiast, Player/Academy)
- High-Contrast Dark Theme Palette:
  * #0F0F0F (Pitch Black Canvas Background)
  * #202020 (Charcoal Surface Container)
  * #5DD62C (Electric Lime Primary Accent & CTA)
  * #337418 (Forest Green Secondary Accent)
  * #F8F8F8 (Off-White Primary Typography)
- Zero HTML code-block indentation bugs (dedented raw HTML rendering)
- Protected Layout margins & responsive z-indexes
"""

import os
import sys
import textwrap
import re

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

from warehouse.warehouse import get_denormalized_medals
from analytics.visualization import (
    THEME_BG,
    PAPER_BG,
    TEXT_COLOR,
    GRID_COLOR,
    PRIMARY_ACCENT,
    SECONDARY_ACCENT,
    MEDAL_COLORS,
    OLYMPIC_COLORS,
    apply_dark_theme,
    plot_top_countries_horizontal_stacked,
    plot_medal_density_heatmap,
    plot_gender_age_violins,
)
from analytics.kmeans import run_kmeans, compute_elbow_curve
from analytics.apriori import mine_association_rules
from analytics.olap import olap_slice, olap_dice, olap_rollup
from dashboard.auth import (
    init_auth,
    is_authenticated,
    get_current_role,
    get_current_user,
    render_landing_page,
    render_sidebar_user_badge,
    render_login_scrolling_background,
    ROLE_ADMIN,
    ROLE_ENTHUSIAST,
    ROLE_ACADEMY
)

# ------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="OLYMPIA ⚡ | Olympic Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Helper function to prevent any Markdown code-block interpretation
def render_html(html_str: str):
    """Renders HTML safely without Markdown 4-space code block formatting."""
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)


# ------------------------------------------------------------
# 2. Design System: High-Contrast Modern Dark-Themed Styling
# ------------------------------------------------------------
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* #0F0F0F: Pitch Black / Canvas Background */
    .stApp {
        background-color: #0F0F0F !important;
        color: #F8F8F8 !important;
    }

    /* Prevent content from sliding under sidebar (Layout Fix) */
    .stMain {
        position: relative !important;
        z-index: 1 !important;
    }

    /* Sidebar Styling (#202020 Charcoal Surface) */
    section[data-testid="stSidebar"] {
        background-color: #202020 !important;
        border-right: 1px solid #337418 !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.7);
        z-index: 99 !important;
    }

    /* Main Content Container Padding & Boundaries */
    .main .block-container, div[data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 100% !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: rgba(15, 15, 15, 0.92) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid #202020 !important;
    }

    /* #F8F8F8: Off-White / Primary Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #F8F8F8 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    p, span, label, div {
        color: #F8F8F8;
    }

    /* Hero Banner Section (#202020 Surface Container with #5DD62C Accent) */
    .hero-banner {
        background: linear-gradient(135deg, #202020 0%, #151515 100%);
        border: 1px solid #337418;
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(93, 214, 44, 0.25);
        position: relative;
        overflow: hidden;
        width: 100%;
        box-sizing: border-box;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #5DD62C 0%, #337418 50%, #5DD62C 100%);
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #F8F8F8 0%, #5DD62C 65%, #A3E635 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #A0A0A0;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* Quick Stat Ticker Cards (#202020 Charcoal Surface with #337418 Borders) */
    .ticker-card {
        background: #202020;
        border: 1px solid #337418;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: left;
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        box-sizing: border-box;
    }
    .ticker-card:hover {
        transform: translateY(-4px);
        border-color: #5DD62C;
        box-shadow: 0 8px 25px rgba(93, 214, 44, 0.25);
    }
    .ticker-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }
    .ticker-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8F8F8;
        line-height: 1.1;
    }
    .ticker-label {
        font-size: 0.8rem;
        color: #A0A0A0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* 3D Podium Cards Styling - Fully responsive with no overflow */
    .podium-container {
        display: flex;
        flex-direction: row;
        align-items: flex-end;
        justify-content: center;
        gap: 1.2rem;
        padding: 1.5rem 0.5rem 1rem 0.5rem;
        margin: 1.2rem 0 2rem 0;
        width: 100% !important;
        box-sizing: border-box !important;
        flex-wrap: wrap;
    }
    .podium-step {
        flex: 1;
        min-width: 220px;
        max-width: 290px;
        border-radius: 16px 16px 10px 10px;
        text-align: center;
        padding: 1.5rem 1.1rem;
        position: relative;
        backdrop-filter: blur(14px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
        box-sizing: border-box;
    }
    .podium-step:hover {
        transform: translateY(-6px);
    }
    /* 1st Place Champion: Electric Lime Glow (#5DD62C) */
    .podium-step.gold {
        background: linear-gradient(180deg, rgba(93, 214, 44, 0.18) 0%, #202020 100%);
        border: 2px solid #5DD62C;
        box-shadow: 0 0 28px rgba(93, 214, 44, 0.35), 0 15px 35px rgba(0, 0, 0, 0.6);
        min-height: 275px;
    }
    /* 2nd Place: Off-White / Silver Accent on Charcoal Surface */
    .podium-step.silver {
        background: linear-gradient(180deg, rgba(248, 248, 248, 0.08) 0%, #202020 100%);
        border: 2px solid #337418;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
        min-height: 235px;
    }
    /* 3rd Place: Forest Green Accent (#337418) */
    .podium-step.bronze {
        background: linear-gradient(180deg, rgba(51, 116, 24, 0.18) 0%, #202020 100%);
        border: 2px solid #337418;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        min-height: 205px;
    }
    .podium-badge {
        font-size: 2.4rem;
        line-height: 1;
        margin-bottom: 0.4rem;
        filter: drop-shadow(0 4px 10px rgba(0,0,0,0.6));
    }
    .podium-country {
        font-family: 'Outfit', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        color: #F8F8F8;
        margin-bottom: 0.2rem;
    }
    .podium-medals {
        font-size: 1.85rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
    }
    .podium-step.gold .podium-medals { color: #5DD62C; }
    .podium-step.silver .podium-medals { color: #F8F8F8; }
    .podium-step.bronze .podium-medals { color: #A3E635; }

    .podium-breakdown {
        display: flex;
        justify-content: center;
        gap: 0.4rem;
        margin-top: 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .podium-chip {
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        background: #0F0F0F;
        border: 1px solid #337418;
    }

    /* Container Cards (#202020 Charcoal Surface) */
    .glass-card {
        background: #202020;
        border: 1px solid #337418;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        box-sizing: border-box;
    }

    /* High-contrast Tabs styling */
    button[data-baseweb="tab"] {
        background: transparent !important;
        color: #A0A0A0 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.75rem 1.4rem !important;
        transition: all 0.2s ease !important;
        border-bottom: 2px solid transparent !important;
    }
    button[aria-selected="true"] {
        color: #5DD62C !important;
        background: rgba(93, 214, 44, 0.08) !important;
        border-bottom: 3px solid #5DD62C !important;
        text-shadow: 0 0 12px rgba(93, 214, 44, 0.4);
    }

    /* #5DD62C: Primary Action Buttons (Electric Lime CTA) */
    .stButton>button {
        background: #5DD62C !important;
        color: #0F0F0F !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.3rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 15px rgba(93, 214, 44, 0.3) !important;
    }
    .stButton>button:hover {
        background: #337418 !important;
        color: #F8F8F8 !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(93, 214, 44, 0.5) !important;
    }

    /* Secondary Download Buttons */
    .stDownloadButton>button {
        background: #202020 !important;
        color: #5DD62C !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        border: 1.5px solid #5DD62C !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(93, 214, 44, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton>button:hover {
        background: #337418 !important;
        color: #F8F8F8 !important;
    }

    /* Fallback Warning Box */
    .fallback-notice {
        background: rgba(51, 116, 24, 0.2);
        border: 1px solid #5DD62C;
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        color: #F8F8F8;
        font-size: 1.05rem;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    /* Athlete Infographic Card */
    .athlete-card {
        background: #202020;
        border: 2px solid #337418;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        box-sizing: border-box;
    }

    /* Dataframe Containers */
    div[data-testid="stDataFrame"] {
        border: 1px solid #337418;
        border-radius: 10px;
        background-color: #202020;
    }
</style>
""")


# ------------------------------------------------------------
# 3. Fast Cached Data Loading
# ------------------------------------------------------------
@st.cache_data(show_spinner="⚡ Connecting to Olympic Star Schema Warehouse...")
def load_all_data() -> pd.DataFrame:
    """Loads all historical Olympic medal facts from the Star Schema."""
    return get_denormalized_medals()


df_master = load_all_data()


# ------------------------------------------------------------
# 4. Mandatory Authentication Gateway (Bug 3 Fix)
# ------------------------------------------------------------
init_auth()

if not is_authenticated():
    # 4. LOGIN-STATE EXCLUSIVITY (CRITICAL): Only rendered when not logged in
    if not st.session_state.get("logged_in", False):
        render_login_scrolling_background()

    # User must log in through 3-Role entry portal before accessing dashboard
    render_landing_page()
    st.stop()


# ------------------------------------------------------------
# 5. Sidebar Controller (The Control Center)
# ------------------------------------------------------------
with st.sidebar:
    # Authenticated User Badge & Logout Button at top of sidebar
    render_sidebar_user_badge()

    render_html("""
    <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #337418; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-size: 1.5rem; font-weight: 800; font-family: 'Outfit', sans-serif; color: #5DD62C; display: flex; align-items: center; gap: 0.4rem;">
                <span>⚡</span> OLYMPIA
            </div>
            <span style="font-size: 0.72rem; background: rgba(93, 214, 44, 0.15); color: #5DD62C; border: 1px solid #5DD62C; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 700;">
                LIVE ARENA
            </span>
        </div>
        <div style="font-size: 0.78rem; color: #A0A0A0; margin-top: 0.3rem; letter-spacing: 0.05em; text-transform: uppercase;">
            ATHLETE DASHBOARD CONTROLLER
        </div>
    </div>
    """)

    # Quick Mode Toggle
    st.markdown("### 🎯 Quick Mode Controller")
    selected_mode = st.sidebar.radio(
        "Quick Mode Controller",
        options=[
            "🏆 The Podium & Medal Tally",
            "🔍 OLAP Cube (Slice & Dice)",
            "🤖 Data Mining Arena",
            "⚡ Athlete Hall of Fame"
        ],
        index=0,
        label_visibility="collapsed",
        key="quick_mode_controller"
    )

    st.markdown("---")
    st.markdown("### 🎛️ Filter Controls")

    # Season Select
    season_options = ["All", "Summer", "Winter"]
    selected_season = st.selectbox("Olympic Season:", options=season_options, index=0)

    # Era / Year Range Slider
    min_year_master = int(df_master["year"].min())
    max_year_master = int(df_master["year"].max())
    year_range = st.slider(
        "Era / Year Range:",
        min_value=min_year_master,
        max_value=max_year_master,
        value=(min_year_master, max_year_master),
        step=4
    )

    # Country / NOC Multi-select
    all_countries = sorted(df_master["country"].unique().tolist())
    selected_countries = st.multiselect(
        "NOC / Countries (Default: All):",
        options=all_countries,
        placeholder="Filter by nation (e.g., United States)..."
    )

    # Sport / Discipline Filter
    all_sports = sorted(df_master["sport"].unique().tolist())
    selected_sports = st.multiselect(
        "Sport / Discipline (Default: All):",
        options=all_sports,
        placeholder="Filter by sport (e.g., Athletics)..."
    )

    # Filter Presets
    st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reset to All-Time Glory", use_container_width=True):
        st.rerun()

    # Controller Status Metrics
    st.markdown("---")
    render_html(f"""
    <div style="font-size: 0.78rem; color: #A0A0A0; line-height: 1.5;">
        <strong style="color: #F8F8F8;">Warehouse Engine:</strong> SQLite3 Star Schema<br>
        <strong style="color: #F8F8F8;">Historical Span:</strong> 1896 – 2024 ({max_year_master - min_year_master} Years)<br>
        <strong style="color: #F8F8F8;">Fact Medals:</strong> {len(df_master):,} verified records
    </div>
    """)


# ------------------------------------------------------------
# 6. Dynamic Data Filtering
# ------------------------------------------------------------
filtered_df = df_master.copy()

if selected_season != "All":
    filtered_df = filtered_df[filtered_df["season"] == selected_season]

filtered_df = filtered_df[
    (filtered_df["year"] >= year_range[0]) & (filtered_df["year"] <= year_range[1])
]

if selected_countries:
    filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]

if selected_sports:
    filtered_df = filtered_df[filtered_df["sport"].isin(selected_sports)]


# ------------------------------------------------------------
# 7. HERO HEADER SECTION
# ------------------------------------------------------------
render_html("""
<div class="hero-banner">
    <div class="hero-title">OLYMPIA ⚡ | Olympic Data Warehouse & Mining Intelligence</div>
    <div class="hero-subtitle">Unleashing Decades of Athletic Glory through Data Analytics & Machine Learning</div>
</div>
""")

# Quick Stat Tickers
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

total_medals_count = len(filtered_df)
participating_nations = filtered_df["country"].nunique()
sports_covered = filtered_df["sport"].nunique()
editions_analyzed = filtered_df["games"].nunique()

with col_stat1:
    render_html(f"""
    <div class="ticker-card">
        <div class="ticker-icon">🏅</div>
        <div class="ticker-value">{total_medals_count:,}</div>
        <div class="ticker-label">Total Medals Analyzed</div>
    </div>
    """)

with col_stat2:
    render_html(f"""
    <div class="ticker-card">
        <div class="ticker-icon">🌐</div>
        <div class="ticker-value">{participating_nations:,}</div>
        <div class="ticker-label">Participating Nations</div>
    </div>
    """)

with col_stat3:
    render_html(f"""
    <div class="ticker-card">
        <div class="ticker-icon">⚡</div>
        <div class="ticker-value">{sports_covered:,}</div>
        <div class="ticker-label">Sports & Disciplines</div>
    </div>
    """)

with col_stat4:
    years_span = f"{year_range[0]} – {year_range[1]}" if year_range[0] != year_range[1] else str(year_range[0])
    render_html(f"""
    <div class="ticker-card">
        <div class="ticker-icon">📅</div>
        <div class="ticker-value">{editions_analyzed} Editions</div>
        <div class="ticker-label">{years_span} Analyzed</div>
    </div>
    """)

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 8. Check for Empty Data Fallback
# ------------------------------------------------------------
if filtered_df.empty:
    render_html("""
    <div class="fallback-notice">
        <span style="font-size: 2rem;">⚠️</span>
        <div>
            <strong style="font-size: 1.15rem; color: #5DD62C;">No athletes qualified under these exact criteria!</strong><br>
            Try widening your search filter or resetting the season and era sliders in the Athlete Dashboard Controller to get back in the game.
        </div>
    </div>
    """)
    st.stop()


# ------------------------------------------------------------
# 9. VIEW RENDERERS & CONDITIONAL ROUTING
# ------------------------------------------------------------

def render_podium_view(filtered_df: pd.DataFrame):
    """Renders the top countries podium, horizontal stacked bar chart, and discipline heatmap."""
    st.markdown("### 🥇 The Olympic Podium Finishers")
    st.markdown("Explore national medal supremacy, 3D podium champions, and cross-sport density.")

    # Calculate Top Countries
    country_totals = filtered_df.groupby("country").agg(
        total_medals=("medal_fact_id", "count"),
        gold=("medal", lambda s: (s == "Gold").sum()),
        silver=("medal", lambda s: (s == "Silver").sum()),
        bronze=("medal", lambda s: (s == "Bronze").sum()),
        points=("medal_points", "sum")
    ).reset_index()

    # Olympic Ranking Order: Gold count first, then Silver, then Bronze
    country_totals = country_totals.sort_values(
        by=["gold", "silver", "bronze", "total_medals"],
        ascending=False
    ).reset_index(drop=True)

    # 3D Literal Podium Finishers Cards (Silver 2nd on Left, Gold 1st in Center, Bronze 3rd on Right)
    if len(country_totals) >= 1:
        first = country_totals.iloc[0]
        second = country_totals.iloc[1] if len(country_totals) > 1 else None
        third = country_totals.iloc[2] if len(country_totals) > 2 else None

        podium_parts = ['<div class="podium-container">']

        # 2nd Place (Silver)
        if second is not None:
            podium_parts.append(
                f'<div class="podium-step silver">'
                f'<div class="podium-badge">🥈</div>'
                f'<div style="font-size: 0.8rem; font-weight: 700; color: #F8F8F8; letter-spacing: 0.08em; text-transform: uppercase;">2ND PLACE</div>'
                f'<div class="podium-country">{second["country"]}</div>'
                f'<div class="podium-medals">{second["total_medals"]:,} <span style="font-size: 0.9rem; font-weight: 400; color: #A0A0A0;">Medals</span></div>'
                f'<div class="podium-breakdown">'
                f'<span class="podium-chip" style="color: #5DD62C;">🥇 {second["gold"]}</span>'
                f'<span class="podium-chip" style="color: #F8F8F8;">🥈 {second["silver"]}</span>'
                f'<span class="podium-chip" style="color: #A3E635;">🥉 {second["bronze"]}</span>'
                f'</div>'
                f'</div>'
            )

        # 1st Place (Gold / Champion - Elevated in Center with #5DD62C Electric Lime Glow)
        podium_parts.append(
            f'<div class="podium-step gold">'
            f'<div class="podium-badge">🥇</div>'
            f'<div style="font-size: 0.85rem; font-weight: 800; color: #5DD62C; letter-spacing: 0.08em; text-transform: uppercase;">OLYMPIC CHAMPION</div>'
            f'<div class="podium-country" style="font-size: 1.55rem; color: #F8F8F8;">{first["country"]}</div>'
            f'<div class="podium-medals" style="font-size: 2.2rem; color: #5DD62C;">{first["total_medals"]:,} <span style="font-size: 1rem; font-weight: 400; color: #A0A0A0;">Medals</span></div>'
            f'<div class="podium-breakdown">'
            f'<span class="podium-chip" style="color: #5DD62C; font-weight: 800;">🥇 {first["gold"]}</span>'
            f'<span class="podium-chip" style="color: #F8F8F8;">🥈 {first["silver"]}</span>'
            f'<span class="podium-chip" style="color: #A3E635;">🥉 {first["bronze"]}</span>'
            f'</div>'
            f'</div>'
        )

        # 3rd Place (Bronze)
        if third is not None:
            podium_parts.append(
                f'<div class="podium-step bronze">'
                f'<div class="podium-badge">🥉</div>'
                f'<div style="font-size: 0.8rem; font-weight: 700; color: #A3E635; letter-spacing: 0.08em; text-transform: uppercase;">3RD PLACE</div>'
                f'<div class="podium-country">{third["country"]}</div>'
                f'<div class="podium-medals" style="color: #A3E635;">{third["total_medals"]:,} <span style="font-size: 0.9rem; font-weight: 400; color: #A0A0A0;">Medals</span></div>'
                f'<div class="podium-breakdown">'
                f'<span class="podium-chip" style="color: #5DD62C;">🥇 {third["gold"]}</span>'
                f'<span class="podium-chip" style="color: #F8F8F8;">🥈 {third["silver"]}</span>'
                f'<span class="podium-chip" style="color: #A3E635;">🥉 {third["bronze"]}</span>'
                f'</div>'
                f'</div>'
            )

        podium_parts.append('</div>')
        render_html("".join(podium_parts))

    # Row 1: Horizontal Stacked Bar Chart for Top Nations
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    top_n_slider = st.slider("Select Number of Top Nations to Display:", min_value=5, max_value=25, value=10, key="podium_top_n")
    fig_stacked = plot_top_countries_horizontal_stacked(filtered_df, top_n=top_n_slider)
    st.plotly_chart(fig_stacked, use_container_width=True)

    # Row 2: Medal Density Heatmap
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 🔥 Discipline Dominance Matrix (Heatmap)")
    st.markdown("Discover where each powerhouse derives its Olympic glory across specific sports.")
    
    col_heat1, col_heat2 = st.columns([1, 1])
    with col_heat1:
        heat_countries = st.slider("Top Countries for Matrix:", min_value=5, max_value=15, value=10, key="heat_c")
    with col_heat2:
        heat_sports = st.slider("Top Sports for Matrix:", min_value=6, max_value=20, value=12, key="heat_s")

    fig_heat = plot_medal_density_heatmap(filtered_df, top_n_countries=heat_countries, top_n_sports=heat_sports)
    st.plotly_chart(fig_heat, use_container_width=True)


def render_olap_view(filtered_df: pd.DataFrame, all_countries: list):
    """Renders the OLAP Cube operations: Drill-Down, Roll-Up, Pivot Table, and Gender/Age Distributions."""
    st.markdown("### 🧊 Multi-Dimensional OLAP Cube Operations")
    st.markdown("Interactive multidimensional exploration: Roll-Up, Drill-Down, Slice & Dice with CSV extraction.")

    olap_view_mode = st.radio(
        "Choose OLAP Operation:",
        options=["📐 Drill-Down (NOC ➔ Sport ➔ Athlete)", "🏛️ Roll-Up (Decade Summary)", "📊 Multi-Dimensional Pivot Matrix", "🎻 Gender & Age Violin Analysis"],
        horizontal=True
    )

    if olap_view_mode == "📐 Drill-Down (NOC ➔ Sport ➔ Athlete)":
        st.markdown("#### 🔍 Drill-Down Hierarchy: Country ➔ Sport ➔ Athletes")
        col_dd1, col_dd2 = st.columns(2)

        available_countries = sorted(filtered_df["country"].unique().tolist())
        if not available_countries:
            available_countries = all_countries

        with col_dd1:
            drill_country = st.selectbox(
                "1. Select Nation (Top Level Hierarchy):",
                options=available_countries,
                index=0
            )

        country_sub = filtered_df[filtered_df["country"] == drill_country]
        sports_in_country = country_sub["sport"].value_counts().index.tolist()

        with col_dd2:
            drill_sport = st.selectbox(
                f"2. Select Sport in {drill_country} (Mid Level Hierarchy):",
                options=sports_in_country if sports_in_country else ["No Sports"],
                index=0
            )

        if sports_in_country:
            drill_detail = country_sub[country_sub["sport"] == drill_sport]
            
            # Key Discipline Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Total Medals", f"{len(drill_detail):,}")
            col_m2.metric("🥇 Gold", f"{(drill_detail['medal'] == 'Gold').sum():,}")
            col_m3.metric("🥈 Silver", f"{(drill_detail['medal'] == 'Silver').sum():,}")
            col_m4.metric("🥉 Bronze", f"{(drill_detail['medal'] == 'Bronze').sum():,}")

            st.markdown(f"##### 🏃 Athlete Roster & Event Details ({drill_country} in {drill_sport})")
            athlete_roster = drill_detail[["year", "games", "event_name", "event_gender", "athletes", "medal"]].sort_values(by="year", ascending=False)
            st.dataframe(athlete_roster, use_container_width=True, hide_index=True)

    elif olap_view_mode == "🏛️ Roll-Up (Decade Summary)":
        st.markdown("#### 📈 Roll-Up Aggregation: Year ➔ Decade")
        df_rollup = filtered_df.copy()
        df_rollup["decade"] = (df_rollup["year"] // 10) * 10
        df_rollup["decade_label"] = df_rollup["decade"].astype(str) + "s"

        decade_summary = df_rollup.groupby("decade_label").agg(
            total_medals=("medal_fact_id", "count"),
            gold=("medal", lambda s: (s == "Gold").sum()),
            silver=("medal", lambda s: (s == "Silver").sum()),
            bronze=("medal", lambda s: (s == "Bronze").sum()),
            active_nations=("country", "nunique"),
            sports_contested=("sport", "nunique")
        ).reset_index().sort_values(by="decade_label")

        # Plotly Decade Line Chart with Electric Lime and Forest Green
        fig_decade = px.bar(
            decade_summary,
            x="decade_label",
            y=["gold", "silver", "bronze"],
            barmode="stack",
            color_discrete_map={"gold": "#5DD62C", "silver": "#F8F8F8", "bronze": "#337418"},
            title="Roll-Up: Total Medals Awarded Per Olympic Decade"
        )
        fig_decade = apply_dark_theme(fig_decade, title="Decade Medal Volume", x_title="Olympic Decade", y_title="Medals")
        st.plotly_chart(fig_decade, use_container_width=True)

        st.dataframe(decade_summary, use_container_width=True, hide_index=True)

    elif olap_view_mode == "📊 Multi-Dimensional Pivot Matrix":
        st.markdown("#### 🔀 Interactive Pivot Table & CSV Export")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            pivot_row = st.selectbox("Row Dimension:", options=["country", "sport", "year", "season"], index=0)
        with col_p2:
            pivot_col = st.selectbox("Column Dimension:", options=["medal", "season", "event_gender", "sport"], index=0)
        with col_p3:
            pivot_val = st.selectbox("Aggregation Metric:", options=["Count of Medals", "Sum of Points"], index=0)

        # Generate Pivot
        val_col = "medal_fact_id" if pivot_val == "Count of Medals" else "medal_points"
        agg_fn = "count" if pivot_val == "Count of Medals" else "sum"

        pivot_table = pd.crosstab(
            index=filtered_df[pivot_row],
            columns=filtered_df[pivot_col],
            values=filtered_df[val_col],
            aggfunc=agg_fn
        ).fillna(0).astype(int)

        # Add Row Total
        pivot_table["TOTAL"] = pivot_table.sum(axis=1)
        pivot_table = pivot_table.sort_values(by="TOTAL", ascending=False)

        st.dataframe(pivot_table, use_container_width=True)

        # Instant CSV Export with Secondary Button Styling
        csv_data = pivot_table.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Export Pivot Matrix to CSV",
            data=csv_data,
            file_name=f"olympia_pivot_{pivot_row}_vs_{pivot_col}.csv",
            mime="text/csv",
            use_container_width=False
        )

    elif olap_view_mode == "🎻 Gender & Age Violin Analysis":
        st.markdown("#### 🏃 Demographic Depth: Gender Participation & Age Distributions")
        st.markdown("Statistical kernel densities comparing competitor ages across Men's, Women's, and Mixed medal events.")
        fig_violin = plot_gender_age_violins(filtered_df)
        st.plotly_chart(fig_violin, use_container_width=True)


def render_data_mining_view(filtered_df: pd.DataFrame, df_master: pd.DataFrame):
    """Renders Data Mining Arena: K-Means performance clusters, Linear Regression trend forecaster, and Apriori rules."""
    st.markdown("### 🤖 Advanced Machine Learning & Pattern Intelligence")
    st.markdown("Cluster nations by historical performance DNA, forecast future medal tallies, and mine sport co-occurrences.")

    ml_view = st.radio(
        "Select Machine Learning Module:",
        options=["🔍 K-Means Performance Clusters", "📈 Trend Forecaster (Linear Regression)", "🔗 Apriori Sport Associations"],
        horizontal=True
    )

    if ml_view == "🔍 K-Means Performance Clusters":
        st.markdown("#### 🧩 Country Performance Clustering")
        st.markdown("Groups nations into athletic archetypes (*'Global Superpowers'*, *'Specialist Powerhouses'*, *'Emerging Contenders'*).")

        col_k1, col_k2 = st.columns([1, 2])
        with col_k1:
            k_val = st.slider("Select Number of Clusters (K):", min_value=2, max_value=6, value=3)
            show_elbow = st.checkbox("Show Elbow Method Curve", value=False)

        # Run K-Means
        kmeans_res = run_kmeans(df=filtered_df if len(filtered_df['country'].unique()) >= k_val else df_master, n_clusters=k_val)
        res_df = kmeans_res["result_df"]

        # Color mapping for archetypes using Electric Lime and Forest Green palette
        cluster_palette = {
            "👑 Global Superpowers": "#5DD62C",         # Electric Lime
            "⚡ Specialist Powerhouses": "#337418",       # Forest Green
            "🎯 Emerging Contenders": "#A3E635",          # Bright Lime
            "🌱 Occasional Podium Finishers": "#707070",  # Charcoal Muted
            "🔥 Rising Challenger Nations": "#42991E",    # Mid Green
            "🏅 Developing Olympic Programs": "#5DD62C"
        }

        fig_cluster = px.scatter(
            res_df,
            x="total_medals",
            y="medal_points",
            color="cluster",
            hover_name="country",
            hover_data=["gold_medals", "silver_medals", "bronze_medals", "total_medals"],
            size="gold_medals",
            size_max=35,
            color_discrete_map=cluster_palette,
            template="plotly_dark",
            title=f"K-Means Performance Clusters (K={k_val})"
        )
        fig_cluster = apply_dark_theme(fig_cluster, title=f"K-Means Performance Clusters (K={k_val})", x_title="Total Historical Medals", y_title="Medal Points (3-2-1 Weighted)")
        st.plotly_chart(fig_cluster, use_container_width=True)

        # Cluster Breakdown Cards
        st.markdown("##### 📊 Cluster Archetype Composition")
        cluster_summary_cols = st.columns(len(kmeans_res["cluster_sizes"]))
        for idx, (c_name, c_count) in enumerate(kmeans_res["cluster_sizes"].items()):
            with cluster_summary_cols[idx]:
                render_html(f"""
                <div class="ticker-card" style="border-color: {cluster_palette.get(c_name, '#337418')};">
                    <div style="font-size: 1.05rem; font-weight: 800; color: #F8F8F8;">{c_name}</div>
                    <div class="ticker-value" style="color: {cluster_palette.get(c_name, '#5DD62C')}; font-size: 1.6rem;">{c_count} <span style="font-size: 0.8rem; font-weight: 400; color: #A0A0A0;">Nations</span></div>
                </div>
                """)

        if show_elbow:
            st.markdown("##### 📐 Optimal Cluster Evaluation (Elbow Curve)")
            elbow_data = compute_elbow_curve(df=df_master, max_k=8)
            elbow_df = pd.DataFrame(elbow_data)
            fig_elbow = px.line(elbow_df, x="k", y="inertia", markers=True, title="Elbow Method Curve (Inertia vs K)")
            fig_elbow.update_traces(line=dict(color="#5DD62C", width=3), marker=dict(size=8, color="#F8F8F8"))
            fig_elbow = apply_dark_theme(fig_elbow, title="Optimal Cluster Selection (Elbow Curve)", x_title="Number of Clusters K", y_title="Inertia (Within-Cluster Sum of Squares)")
            st.plotly_chart(fig_elbow, use_container_width=True)

    elif ml_view == "📈 Trend Forecaster (Linear Regression)":
        st.markdown("#### 🔮 Olympic Medal Trajectory Forecaster")
        st.markdown("Predict future medal tallies using chronological linear regression modeling across historical editions.")

        top_candidate_countries = filtered_df["country"].value_counts().head(20).index.tolist()
        selected_forecast_country = st.selectbox(
            "Select Nation to Forecast (e.g., United States, China, Great Britain):",
            options=top_candidate_countries if top_candidate_countries else ["United States"],
            index=0
        )

        country_history = df_master[df_master["country"] == selected_forecast_country]
        yearly_tallies = country_history.groupby("year").size().reset_index(name="medals")

        if len(yearly_tallies) >= 3:
            X = yearly_tallies[["year"]].values
            y = yearly_tallies["medals"].values

            model = LinearRegression()
            model.fit(X, y)

            # Predict on future Olympic Games (2028 Los Angeles, 2032 Brisbane)
            future_years = np.array([[2028], [2032]])
            future_preds = model.predict(future_years)
            future_preds = np.clip(future_preds, 0, None)  # No negative medals

            # Build forecast chart with Electric Lime & Forest Green
            fig_fc = go.Figure()

            # Historical actuals (Electric Lime #5DD62C)
            fig_fc.add_trace(go.Scatter(
                x=yearly_tallies["year"],
                y=yearly_tallies["medals"],
                mode="lines+markers",
                name="Historical Medals",
                line=dict(color="#5DD62C", width=3),
                marker=dict(size=8, color="#5DD62C")
            ))

            # Fitted trendline (Forest Green #337418)
            fitted_y = model.predict(X)
            fig_fc.add_trace(go.Scatter(
                x=yearly_tallies["year"],
                y=fitted_y,
                mode="lines",
                name="Historical Trendline",
                line=dict(color="#337418", width=2, dash="dot")
            ))

            # Future Projections (Bright Lime with star markers)
            all_future_x = [yearly_tallies["year"].iloc[-1], 2028, 2032]
            all_future_y = [yearly_tallies["medals"].iloc[-1], round(future_preds[0]), round(future_preds[1])]

            fig_fc.add_trace(go.Scatter(
                x=all_future_x,
                y=all_future_y,
                mode="lines+markers",
                name="Future Forecast (2028 / 2032)",
                line=dict(color="#5DD62C", width=3, dash="dash"),
                marker=dict(size=12, color="#5DD62C", symbol="star")
            ))

            fig_fc = apply_dark_theme(
                fig_fc,
                title=f"📈 Medal Projection: {selected_forecast_country} (Through LA 2028 & Brisbane 2032)",
                x_title="Olympic Year",
                y_title="Projected Medals"
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Forecast metrics
            col_fc1, col_fc2, col_fc3 = st.columns(3)
            col_fc1.metric("Historical Slope", f"{model.coef_[0]:+.2f} medals/yr")
            col_fc2.metric("🔮 2028 Los Angeles Estimate", f"~{int(round(future_preds[0]))} medals")
            col_fc3.metric("🔮 2032 Brisbane Estimate", f"~{int(round(future_preds[1]))} medals")
        else:
            st.warning("Insufficient historical appearances for reliable regression forecasting.")

    elif ml_view == "🔗 Apriori Sport Associations":
        st.markdown("#### 🛒 Pattern Detector: Apriori Association Rules")
        st.markdown("Mines frequent sport combinations won by the same nation within an Olympic Games edition.")

        col_ap1, col_ap2 = st.columns(2)
        with col_ap1:
            support_pct = st.slider("Minimum Support Threshold (%):", min_value=2, max_value=20, value=5, step=1)
        with col_ap2:
            conf_pct = st.slider("Minimum Confidence Threshold (%):", min_value=20, max_value=80, value=35, step=5)

        with st.spinner("Mining frequent sport co-occurrence patterns..."):
            rules_res = mine_association_rules(
                df=filtered_df,
                min_support=support_pct / 100.0,
                min_confidence=conf_pct / 100.0
            )

        if rules_res["status"] == "SUCCESS" and not rules_res["rules"].empty:
            rules_df = rules_res["rules"]

            st.markdown(f"Found **{len(rules_df)} strong association rules** across Olympic editions:")
            
            # Render as sporty rule badges with Electric Lime & Forest Green
            for _, r in rules_df.head(10).iterrows():
                ant = ", ".join(list(r["antecedents"]))
                con = ", ".join(list(r["consequents"]))
                supp = r["support"] * 100
                conf = r["confidence"] * 100
                lift = r["lift"]

                render_html(f"""
                <div style="background: #202020; border: 1px solid #337418; border-radius: 10px; padding: 0.9rem 1.2rem; margin-bottom: 0.6rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.8rem;">
                    <div style="display: flex; align-items: center; gap: 0.6rem; font-weight: 700; font-size: 1.05rem;">
                        <span style="background: rgba(93, 214, 44, 0.15); color: #5DD62C; padding: 0.25rem 0.65rem; border-radius: 6px; border: 1px solid #5DD62C;">{ant}</span>
                        <span style="color: #5DD62C; font-weight: 800;">➔</span>
                        <span style="background: rgba(51, 116, 24, 0.25); color: #A3E635; padding: 0.25rem 0.65rem; border-radius: 6px; border: 1px solid #337418;">{con}</span>
                    </div>
                    <div style="display: flex; gap: 1rem; font-size: 0.85rem; color: #A0A0A0;">
                        <span>Support: <strong style="color: #F8F8F8;">{supp:.1f}%</strong></span>
                        <span>Confidence: <strong style="color: #5DD62C;">{conf:.1f}%</strong></span>
                        <span>Lift: <strong style="color: #5DD62C;">{lift:.2f}x</strong></span>
                    </div>
                </div>
                """)
        else:
            st.info("No frequent sport combinations met the current thresholds. Try reducing the Minimum Support slider.")


@st.cache_data
def get_unique_athlete_list(df: pd.DataFrame) -> list:
    """Extracts cleaned individual athlete names across historical facts and legend lists."""
    names = set()
    for s in df["athletes"].dropna():
        for sub in s.split(","):
            cleaned = sub.strip()
            if len(cleaned) > 2:
                names.add(cleaned)
    for leg in [
        "Michael PHELPS",
        "Usain BOLT",
        "Larisa LATYNINA",
        "Carl LEWIS",
        "Paavo NURMI",
        "Simone BILES",
        "Katie LEDECKY",
        "Birgit FISCHER",
        "Nikolai ANDRIANOV",
        "Mark SPITZ"
    ]:
        names.add(leg)
    return sorted(list(names))


def find_athlete_records(df: pd.DataFrame, athlete_name: str) -> pd.DataFrame:
    """Finds all medal records for an athlete with support for reversed names and team relays."""
    if not athlete_name or not isinstance(athlete_name, str):
        return pd.DataFrame()
    name = athlete_name.strip()
    mask = df["athletes"] == name
    
    tokens = [t for t in re.split(r'\s+', name) if len(t) > 1]
    if len(tokens) >= 2:
        rev_name = ' '.join(tokens[::-1])
        mask = mask | (df["athletes"] == rev_name)
        # Match surname & first name token
        p0 = r'(?i)\b' + re.escape(tokens[0][:4])
        p1 = r'(?i)\b' + re.escape(tokens[-1]) + r'\b'
        sub_mask = df["athletes"].str.contains(p0, na=False, regex=True) & df["athletes"].str.contains(p1, na=False, regex=True)
        mask = mask | sub_mask
    elif len(tokens) == 1:
        p = r'(?i)\b' + re.escape(tokens[0]) + r'\b'
        mask = mask | df["athletes"].str.contains(p, na=False, regex=True)
        
    return df[mask]


def render_hall_of_fame_view(df_master: pd.DataFrame):
    """Renders Athlete Hall of Fame: Quick legend selectors, live search, infographic card, timeline, and catalog."""
    st.markdown("### ⚡ Olympic Legends & Athlete Hall of Fame")
    st.markdown("Spotlight legendary olympians, their career progression, age milestones, and medal catalogs.")

    # 1. SYNCHRONIZE SESSION STATE
    if "selected_athlete" not in st.session_state:
        st.session_state["selected_athlete"] = "Michael PHELPS"

    LEGENDARY_NAMES = [
        "Michael PHELPS",
        "Usain BOLT",
        "Larisa LATYNINA",
        "Carl LEWIS",
        "Paavo NURMI",
        "Simone BILES",
        "Katie LEDECKY",
        "Birgit FISCHER",
        "Nikolai ANDRIANOV",
        "Mark SPITZ"
    ]

    all_athlete_names = get_unique_athlete_list(df_master)
    if st.session_state["selected_athlete"] not in all_athlete_names:
        all_athlete_names = [st.session_state["selected_athlete"]] + all_athlete_names

    # Callback when quick-select buttons are clicked
    def set_athlete_selection(name: str):
        st.session_state["selected_athlete"] = name
        st.session_state["athlete_selectbox_key"] = name

    st.markdown("##### 🌟 Quick Select Legendary Champions:")
    quick_cols = st.columns(len(LEGENDARY_NAMES[:5]))

    for i, name in enumerate(LEGENDARY_NAMES[:5]):
        if quick_cols[i].button(
            name,
            key=f"legend_btn_{i}",
            use_container_width=True,
            on_click=set_athlete_selection,
            args=(name,)
        ):
            st.session_state["selected_athlete"] = name

    # Sync selectbox widget key with session state
    if "athlete_selectbox_key" not in st.session_state:
        st.session_state["athlete_selectbox_key"] = st.session_state["selected_athlete"]
    elif st.session_state["athlete_selectbox_key"] != st.session_state["selected_athlete"]:
        st.session_state["athlete_selectbox_key"] = st.session_state["selected_athlete"]

    current_idx = all_athlete_names.index(st.session_state["athlete_selectbox_key"]) if st.session_state["athlete_selectbox_key"] in all_athlete_names else 0

    def on_selectbox_change():
        st.session_state["selected_athlete"] = st.session_state["athlete_selectbox_key"]

    searched_athlete = st.selectbox(
        "🔎 Search & Inspect Any Olympic Medalist:",
        options=all_athlete_names,
        index=current_idx,
        key="athlete_selectbox_key",
        on_change=on_selectbox_change
    )
    st.session_state["selected_athlete"] = searched_athlete
    active_athlete = st.session_state["selected_athlete"]

    # 2. FILTER DATA FOR SELECTED ATHLETE
    athlete_records = find_athlete_records(df_master, active_athlete).sort_values(by="year")

    # 3. FALLBACK HANDLING
    if athlete_records.empty:
        st.warning("No detailed historical records found for this athlete.")
        return

    # 4. COMPUTE PROFILE METRICS
    ath_country = athlete_records["country"].iloc[0]
    ath_country_code = athlete_records["country_code"].iloc[0] if "country_code" in athlete_records.columns else ""
    ath_sport = athlete_records["sport"].mode().iloc[0]
    ath_all_sports = ", ".join(athlete_records["sport"].unique())
    ath_total = len(athlete_records)
    ath_gold = int((athlete_records["medal"] == "Gold").sum())
    ath_silver = int((athlete_records["medal"] == "Silver").sum())
    ath_bronze = int((athlete_records["medal"] == "Bronze").sum())
    ath_points = int(athlete_records["medal_points"].sum())

    first_year = int(athlete_records["year"].min())
    last_year = int(athlete_records["year"].max())
    editions_count = athlete_records["games"].nunique()
    games_span = f"{first_year} – {last_year}" if first_year != last_year else str(first_year)

    # 5. ATHLETE INFOGRAPHIC HERO CARD
    render_html(f"""
    <div class="athlete-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1.2rem;">
            <div>
                <div style="font-size: 0.85rem; color: #5DD62C; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 800;">
                    ⚡ HALL OF FAME INDUCTEE PROFILE
                </div>
                <div style="font-size: 2.3rem; font-weight: 900; font-family: 'Outfit', sans-serif; color: #F8F8F8; margin: 0.3rem 0;">
                    {active_athlete}
                </div>
                <div style="font-size: 1.05rem; color: #A0A0A0; display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;">
                    <span>🌐 <strong style="color: #F8F8F8;">{ath_country}</strong> ({ath_country_code})</span>
                    <span>•</span>
                    <span>⚡ <strong style="color: #5DD62C;">{ath_all_sports}</strong></span>
                    <span>•</span>
                    <span>📅 <strong style="color: #F8F8F8;">{games_span}</strong> ({editions_count} Editions)</span>
                </div>
            </div>
            <div style="display: flex; gap: 0.8rem; flex-wrap: wrap;">
                <div style="background: rgba(93, 214, 44, 0.15); border: 1.5px solid #5DD62C; border-radius: 10px; padding: 0.75rem 1.1rem; text-align: center; min-width: 80px;">
                    <div style="font-size: 1.7rem; font-weight: 800; color: #5DD62C;">{ath_gold}</div>
                    <div style="font-size: 0.75rem; color: #F8F8F8; text-transform: uppercase; font-weight: 700;">🥇 Gold</div>
                </div>
                <div style="background: #202020; border: 1.5px solid #337418; border-radius: 10px; padding: 0.75rem 1.1rem; text-align: center; min-width: 80px;">
                    <div style="font-size: 1.7rem; font-weight: 800; color: #F8F8F8;">{ath_silver}</div>
                    <div style="font-size: 0.75rem; color: #A0A0A0; text-transform: uppercase; font-weight: 700;">🥈 Silver</div>
                </div>
                <div style="background: rgba(51, 116, 24, 0.25); border: 1.5px solid #337418; border-radius: 10px; padding: 0.75rem 1.1rem; text-align: center; min-width: 80px;">
                    <div style="font-size: 1.7rem; font-weight: 800; color: #A3E635;">{ath_bronze}</div>
                    <div style="font-size: 0.75rem; color: #A0A0A0; text-transform: uppercase; font-weight: 700;">🥉 Bronze</div>
                </div>
                <div style="background: #5DD62C; border: 1.5px solid #5DD62C; border-radius: 10px; padding: 0.75rem 1.1rem; text-align: center; min-width: 85px;">
                    <div style="font-size: 1.7rem; font-weight: 900; color: #0F0F0F;">{ath_total}</div>
                    <div style="font-size: 0.75rem; color: #0F0F0F; font-weight: 800; text-transform: uppercase;">🏅 Total</div>
                </div>
            </div>
        </div>
    </div>
    """)

    # 6. KEY METRICS CARDS
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        render_html(f"""
        <div class="ticker-card">
            <div class="ticker-icon">🏅</div>
            <div class="ticker-value">{ath_total} Medals</div>
            <div class="ticker-label">🥇 {ath_gold} | 🥈 {ath_silver} | 🥉 {ath_bronze}</div>
        </div>
        """)
    with col_k2:
        render_html(f"""
        <div class="ticker-card">
            <div class="ticker-icon">🌐</div>
            <div class="ticker-value">{ath_country}</div>
            <div class="ticker-label">NOC Code: {ath_country_code}</div>
        </div>
        """)
    with col_k3:
        render_html(f"""
        <div class="ticker-card">
            <div class="ticker-icon">⚡</div>
            <div class="ticker-value">{ath_sport}</div>
            <div class="ticker-label">Discipline / Program</div>
        </div>
        """)
    with col_k4:
        render_html(f"""
        <div class="ticker-card">
            <div class="ticker-icon">📅</div>
            <div class="ticker-value">{games_span}</div>
            <div class="ticker-label">{editions_count} Olympic Editions</div>
        </div>
        """)

    # 7. VISUAL BREAKDOWN CHARTS
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"#### 📊 Olympic Medal Breakdown by Edition: {active_athlete}")

    edition_medals = athlete_records.groupby(["games", "year", "medal"]).size().reset_index(name="count")
    edition_medals = edition_medals.sort_values(by="year")

    fig_bar = px.bar(
        edition_medals,
        x="games",
        y="count",
        color="medal",
        color_discrete_map={"Gold": "#5DD62C", "Silver": "#F8F8F8", "Bronze": "#337418"},
        barmode="stack",
        category_orders={"medal": ["Gold", "Silver", "Bronze"]},
        title=f"Medal Haul Per Olympic Games: {active_athlete}"
    )
    fig_bar = apply_dark_theme(
        fig_bar,
        title=f"Olympic Medals by Edition ({ath_total} Total Medals Won)",
        x_title="Olympic Games Edition",
        y_title="Medals Won"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Career Event Timeline Scatter Plot
    st.markdown(f"#### 📅 Career Event Timeline & Disciplines: {active_athlete}")
    timeline_df = athlete_records.copy().sort_values(by=["year", "event_name"])

    fig_ath = px.scatter(
        timeline_df,
        x="year",
        y="event_name",
        color="medal",
        color_discrete_map={"Gold": "#5DD62C", "Silver": "#F8F8F8", "Bronze": "#337418"},
        category_orders={"medal": ["Gold", "Silver", "Bronze"]},
        size=[14] * len(timeline_df),
        hover_data=["games", "sport", "event_gender", "medal"],
        title=f"Olympic Podium Timeline: {active_athlete}"
    )
    fig_ath.update_layout(xaxis=dict(tickmode="linear", dtick=4))
    fig_ath = apply_dark_theme(
        fig_ath,
        title=f"Career Event Timeline: {active_athlete}",
        x_title="Olympic Year",
        y_title="Event Name"
    )
    st.plotly_chart(fig_ath, use_container_width=True)

    # 8. MEDAL CATALOG / TIMELINE TABLE
    st.markdown(f"#### 📜 Complete Olympic Medal Catalog: {active_athlete}")
    display_ath_df = athlete_records[["year", "games", "sport", "event_name", "event_gender", "medal", "medal_points"]].copy()
    display_ath_df = display_ath_df.sort_values(by=["year", "medal_points"], ascending=[False, False])
    display_ath_df.columns = ["Year", "Olympic Games / City", "Sport", "Event Name", "Event Gender", "Medal Won", "Points"]
    st.dataframe(display_ath_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# 10. CONDITIONAL PAGE ROUTING
# ------------------------------------------------------------
if "Podium" in selected_mode or selected_mode == "🏆 The Podium & Medal Tally":
    render_podium_view(filtered_df)
elif "OLAP" in selected_mode or selected_mode == "🔍 OLAP Cube (Slice & Dice)":
    render_olap_view(filtered_df, all_countries)
elif "Mining" in selected_mode or selected_mode == "🤖 Data Mining Arena":
    render_data_mining_view(filtered_df, df_master)
elif "Athlete" in selected_mode or "Hall" in selected_mode or selected_mode == "⚡ Athlete Hall of Fame":
    render_hall_of_fame_view(df_master)
else:
    render_podium_view(filtered_df)

