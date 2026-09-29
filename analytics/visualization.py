"""
Visualization Module for OLYMPIA.
Builds interactive Plotly charts adhering to the Dark Analytics theme:
1. BAR CHART: Top countries, sport distributions, categorical comparisons.
2. LINE CHART: Medal trajectories across Olympic years.
3. SCATTER PLOT: Correlation between Gold medals and Total medal points.
4. HISTOGRAM: Distribution of medal points / counts across participating nations.
5. BOX PLOT: Spread and outliers of medals won per Olympic edition across sports.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any

# Olympic Palette & Dark Theme styling constants
THEME_BG = "#0f172a"        # Deep slate navy
PAPER_BG = "#1e293b"        # Card dark slate
TEXT_COLOR = "#f8fafc"      # Off-white
GRID_COLOR = "#334155"      # Muted slate border

MEDAL_COLORS = {
    "Gold": "#fbbf24",      # Warm Olympic gold
    "Silver": "#94a3b8",    # Crisp silver
    "Bronze": "#d97706",    # Rich bronze
}

OLYMPIC_COLORS = [
    "#38bdf8",  # Sky blue
    "#fbbf24",  # Yellow
    "#22c55e",  # Green
    "#ef4444",  # Red
    "#a855f7",  # Purple
    "#06b6d4",  # Cyan
    "#f97316",  # Orange
]


def apply_dark_theme(fig: go.Figure, title: str = "", x_title: str = "", y_title: str = "") -> go.Figure:
    """Applies a consistent, polished dark theme to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 16, "color": TEXT_COLOR, "family": "Inter, Roboto, sans-serif"},
            "x": 0.02,
            "xanchor": "left"
        },
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=THEME_BG,
        font={"color": TEXT_COLOR, "family": "Inter, Roboto, sans-serif"},
        xaxis={
            "title": x_title,
            "gridcolor": GRID_COLOR,
            "zerolinecolor": GRID_COLOR,
            "tickfont": {"color": "#94a3b8"},
        },
        yaxis={
            "title": y_title,
            "gridcolor": GRID_COLOR,
            "zerolinecolor": GRID_COLOR,
            "tickfont": {"color": "#94a3b8"},
        },
        legend={
            "bgcolor": "rgba(30, 41, 59, 0.8)",
            "bordercolor": GRID_COLOR,
            "font": {"color": TEXT_COLOR},
        },
        hoverlabel={
            "bgcolor": "#1e293b",
            "font": {"color": "#ffffff", "family": "Inter, sans-serif"},
            "bordercolor": "#38bdf8"
        },
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        autosize=True
    )
    return fig


# ============================================================
# 1. BAR CHART
# ============================================================
def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "Bar Chart",
    orientation: str = "v",
    barmode: str = "group"
) -> go.Figure:
    """Generates an interactive bar chart with dynamic coloring and tooltips."""
    color_map = MEDAL_COLORS if color == "medal" else None
    color_seq = None if color_map else OLYMPIC_COLORS

    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        orientation=orientation,
        barmode=barmode,
        color_discrete_map=color_map,
        color_discrete_sequence=color_seq,
        template="plotly_dark"
    )
    return apply_dark_theme(fig, title=title, x_title=x.replace("_", " ").title(), y_title=y.replace("_", " ").title())


# ============================================================
# 2. LINE CHART
# ============================================================
def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "Line Chart",
    markers: bool = True
) -> go.Figure:
    """Generates an interactive line chart showing trends over time."""
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        markers=markers,
        color_discrete_sequence=OLYMPIC_COLORS,
        template="plotly_dark"
    )
    fig.update_traces(line={"width": 3}, marker={"size": 7})
    return apply_dark_theme(fig, title=title, x_title=x.replace("_", " ").title(), y_title=y.replace("_", " ").title())


# ============================================================
# 3. SCATTER PLOT
# ============================================================
def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: Optional[str] = None,
    color: Optional[str] = None,
    hover_name: Optional[str] = None,
    title: str = "Scatter Plot"
) -> go.Figure:
    """Generates an interactive scatter plot with bubble sizing and hover names."""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        hover_name=hover_name,
        color_discrete_sequence=OLYMPIC_COLORS,
        template="plotly_dark"
    )
    return apply_dark_theme(fig, title=title, x_title=x.replace("_", " ").title(), y_title=y.replace("_", " ").title())


# ============================================================
# 4. HISTOGRAM
# ============================================================
def create_histogram(
    df: pd.DataFrame,
    x: str,
    nbins: int = 30,
    color: Optional[str] = None,
    title: str = "Histogram"
) -> go.Figure:
    """Generates an interactive histogram showing continuous variable distribution."""
    fig = px.histogram(
        df,
        x=x,
        nbins=nbins,
        color=color,
        color_discrete_sequence=["#38bdf8"],
        template="plotly_dark"
    )
    fig.update_layout(bargap=0.08)
    return apply_dark_theme(fig, title=title, x_title=x.replace("_", " ").title(), y_title="Frequency / Count")


# ============================================================
# 5. BOX PLOT
# ============================================================
def create_box_plot(
    df: pd.DataFrame,
    y: str,
    x: Optional[str] = None,
    color: Optional[str] = None,
    title: str = "Box Plot"
) -> go.Figure:
    """Generates an interactive box plot visualizing statistical spread, quartiles, and outliers."""
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color,
        points="outliers",
        color_discrete_sequence=OLYMPIC_COLORS,
        template="plotly_dark"
    )
    return apply_dark_theme(fig, title=title, x_title=(x.replace("_", " ").title() if x else ""), y_title=y.replace("_", " ").title())


# ============================================================
# SPECIALIZED OLYMPIC DASHBOARD VISUALIZERS
# ============================================================
def plot_top_countries(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Bar chart of top N countries by total medals, broken down by Gold, Silver, Bronze."""
    country_medals = (
        df.groupby(["country", "medal"])
        .size()
        .reset_index(name="count")
    )
    # Get top N countries by total count
    top_country_names = (
        country_medals.groupby("country")["count"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    filtered = country_medals[country_medals["country"].isin(top_country_names)].copy()
    
    # Sort countries by total
    order_map = {c: i for i, c in enumerate(top_country_names)}
    filtered["rank"] = filtered["country"].map(order_map)
    filtered = filtered.sort_values(by=["rank", "medal"])

    fig = px.bar(
        filtered,
        x="country",
        y="count",
        color="medal",
        barmode="stack",
        category_orders={"country": top_country_names, "medal": ["Gold", "Silver", "Bronze"]},
        color_discrete_map=MEDAL_COLORS,
        template="plotly_dark"
    )
    return apply_dark_theme(fig, title=f"Top {top_n} Countries by Medal Distribution", x_title="Country", y_title="Medals Won")


def plot_medals_over_time(df: pd.DataFrame, countries: Optional[List[str]] = None) -> go.Figure:
    """Line chart tracking medal performance over Olympic editions."""
    filtered = df.copy()
    if countries:
        filtered = filtered[filtered["country"].isin(countries)]

    yearly = (
        filtered.groupby(["year", "country"])
        .size()
        .reset_index(name="medals")
    )
    fig = px.line(
        yearly,
        x="year",
        y="medals",
        color="country",
        markers=True,
        color_discrete_sequence=OLYMPIC_COLORS,
        template="plotly_dark"
    )
    fig.update_traces(line={"width": 2.5}, marker={"size": 6})
    return apply_dark_theme(fig, title="Olympic Medal Trajectory Across Years", x_title="Olympic Year", y_title="Medals Awarded")


def plot_gender_breakdown(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing Men's, Women's, and Mixed medal events."""
    gender_counts = df["event_gender"].value_counts().reset_index()
    gender_counts.columns = ["event_gender", "count"]

    fig = px.pie(
        gender_counts,
        names="event_gender",
        values="count",
        hole=0.45,
        color="event_gender",
        color_discrete_map={
            "Men's": "#38bdf8",
            "Women's": "#f43f5e",
            "Mixed": "#a855f7"
        },
        template="plotly_dark"
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return apply_dark_theme(fig, title="Medal Event Gender Distribution", x_title="", y_title="")
