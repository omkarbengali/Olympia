"""
Charts Bridge Component for OLYMPIA.
Integrates Plotly charts with Streamlit styling.
"""

import streamlit as st
import plotly.graph_objects as go


def display_chart(fig: go.Figure, key: str = None):
    """Renders a Plotly chart with full responsive width."""
    st.plotly_chart(fig, use_container_width=True, key=key)
