"""
UI Metrics and KPI Cards component for OLYMPIA.
Renders dark analytics KPI cards with Olympic accents.
"""

import streamlit as st
from typing import List, Dict, Any, Optional


def render_kpi_card(title: str, value: Any, subtitle: Optional[str] = None, icon: Optional[str] = None):
    """Renders a single styled KPI card."""
    icon_html = f"<span style='font-size: 1.5rem; margin-right: 0.5rem;'>{icon}</span>" if icon else ""
    sub_html = f"<div style='font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem;'>{subtitle}</div>" if subtitle else ""

    card_html = f"""
    <div style="
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    ">
        <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
            {title}
        </div>
        <div style="font-size: 2rem; font-weight: 700; color: #f8fafc; margin-top: 0.3rem; display: flex; align-items: center;">
            {icon_html}{value}
        </div>
        {sub_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_kpi_row(kpis: List[Dict[str, Any]]):
    """Renders a responsive horizontal row of KPI cards."""
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            render_kpi_card(
                title=kpi.get("title", ""),
                value=kpi.get("value", 0),
                subtitle=kpi.get("subtitle"),
                icon=kpi.get("icon")
            )


def render_viva_note(title: str, description: str, theory: Optional[str] = None):
    """Renders a viva-friendly conceptual explanation callout."""
    theory_html = f"<div style='margin-top: 0.5rem; font-size: 0.85rem; color: #cbd5e1;'><strong>Viva Tip:</strong> {theory}</div>" if theory else ""
    note_html = f"""
    <div style="
        background: rgba(30, 41, 59, 0.8);
        border-left: 4px solid #38bdf8;
        border-radius: 4px 8px 8px 4px;
        padding: 0.9rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #e2e8f0;
    ">
        <div style="font-weight: 600; color: #38bdf8; margin-bottom: 0.25rem;">
            💡 How it works: {title}
        </div>
        <div>{description}</div>
        {theory_html}
    </div>
    """
    st.markdown(note_html, unsafe_allow_html=True)
