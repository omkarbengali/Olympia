"""
Authentication and Role-Based Access Control for OLYMPIA.
Supports 3 distinct user roles:
1. ADMIN: Full system access (Warehouse, ML, OLAP, Data Quality, Overview).
2. SPORT ENTHUSIAST: Simplified, visual, zero-jargon exploration.
3. PROFESSIONAL PLAYER / ACADEMY: High-performance intelligence and historical benchmarks.
"""

import streamlit as st
from typing import Optional, Tuple

# Role Constants
ROLE_ADMIN = "ADMIN"
ROLE_ENTHUSIAST = "SPORT_ENTHUSIAST"
ROLE_ACADEMY = "PLAYER_ACADEMY"

ROLE_DISPLAY = {
    ROLE_ADMIN: {
        "title": "Administrator",
        "icon": "🛡️",
        "badge_color": "#ef4444",
        "description": "Full access to Data Warehouse, ETL status, ML models, and administrative tools."
    },
    ROLE_ENTHUSIAST: {
        "title": "Sport Enthusiast",
        "icon": "🎉",
        "badge_color": "#06b6d4",
        "description": "Streamlined, visual, interactive Olympic exploration without database theory."
    },
    ROLE_ACADEMY: {
        "title": "Professional Player / Academy",
        "icon": "🎯",
        "badge_color": "#10b981",
        "description": "High-performance discipline intelligence, competitive landscape, and historical benchmarks."
    }
}

DEMO_ACCOUNTS = {
    "admin": {"password": "olympia2024", "role": ROLE_ADMIN, "name": "System Administrator"},
    "fan": {"password": "olympia", "role": ROLE_ENTHUSIAST, "name": "Olympic Fan"},
    "coach": {"password": "olympia", "role": ROLE_ACADEMY, "name": "Elite Sports Academy"},
}


def init_auth():
    """Initializes session state keys for authentication."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "user_fullname" not in st.session_state:
        st.session_state["user_fullname"] = None
    if "selected_role_intent" not in st.session_state:
        st.session_state["selected_role_intent"] = None


def is_authenticated() -> bool:
    """Checks if the current session has an authenticated user."""
    init_auth()
    return st.session_state.get("authenticated", False)


def get_current_role() -> Optional[str]:
    """Returns the active user role."""
    init_auth()
    return st.session_state.get("user_role")


def get_current_user() -> Optional[dict]:
    """Returns user profile metadata."""
    init_auth()
    if not is_authenticated():
        return None
    return {
        "username": st.session_state.get("username"),
        "role": st.session_state.get("user_role"),
        "fullname": st.session_state.get("user_fullname", "User"),
    }


def login(username: str, role: str, fullname: Optional[str] = None):
    """Sets session state on successful authentication."""
    init_auth()
    st.session_state["authenticated"] = True
    st.session_state["username"] = username
    st.session_state["user_role"] = role
    st.session_state["user_fullname"] = fullname or DEMO_ACCOUNTS.get(username, {}).get("name", username)
    st.session_state["selected_role_intent"] = None
    st.rerun()


def logout():
    """Clears authentication session state."""
    init_auth()
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["user_role"] = None
    st.session_state["user_fullname"] = None
    st.session_state["selected_role_intent"] = None
    st.rerun()


def render_landing_page():
    """
    Renders the modern, 3-role entry portal.
    Users can either 1-click quick launch a role or sign in with demo credentials.
    """
    init_auth()

    # Hero Banner
    st.markdown("""
    <div style="text-align: center; padding: 2.5rem 1rem 1.5rem 1rem;">
        <div style="font-size: 3.2rem; line-height: 1;">🏅</div>
        <h1 style="font-size: 2.6rem; font-weight: 800; margin-top: 0.5rem; margin-bottom: 0.2rem; color: #f8fafc; letter-spacing: -0.03em;">
            OLYMPIA
        </h1>
        <div style="font-size: 1.15rem; color: #38bdf8; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
            Olympic Performance Analytics & Intelligence System
        </div>
        <p style="max-width: 650px; margin: 0.8rem auto 0 auto; color: #94a3b8; font-size: 0.95rem; line-height: 1.5;">
            Explore 128 years of modern Olympic history (1896 – 2024) powered by a verified Star Schema Data Warehouse,
            unsupervised clustering, predictive modeling, and multi-dimensional OLAP analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: #e2e8f0; margin-bottom: 1.5rem;'>Choose Your Experience</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    # 1. ADMIN ROLE CARD
    with col1:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; height: 320px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div>
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🛡️</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">Administrator</div>
                <div style="font-size: 0.8rem; color: #ef4444; font-weight: 600; text-transform: uppercase; margin-bottom: 0.75rem;">Full System Access</div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.4;">
                    Data Warehouse inspection, ETL health, data quality audits, OLAP cube laboratory, and machine learning models.
                </div>
            </div>
            <div style="font-size: 0.75rem; color: #64748b; border-top: 1px solid #334155; padding-top: 0.5rem;">
                Default user: <code>admin</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Enter as Admin", key="btn_quick_admin", use_container_width=True):
            login("admin", ROLE_ADMIN, "System Administrator")

    # 2. SPORT ENTHUSIAST ROLE CARD
    with col2:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; height: 320px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div>
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎉</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">Sport Enthusiast</div>
                <div style="font-size: 0.8rem; color: #06b6d4; font-weight: 600; text-transform: uppercase; margin-bottom: 0.75rem;">Visual & Intuitive</div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.4;">
                    Engaging Olympic storyboards, country highlights, sport milestones, head-to-head nation showdowns, and medal trends.
                </div>
            </div>
            <div style="font-size: 0.75rem; color: #64748b; border-top: 1px solid #334155; padding-top: 0.5rem;">
                Default user: <code>fan</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🌟 Enter as Enthusiast", key="btn_quick_fan", use_container_width=True):
            login("fan", ROLE_ENTHUSIAST, "Olympic Fan")

    # 3. PLAYER / ACADEMY ROLE CARD
    with col3:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; height: 320px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div>
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">Player / Academy</div>
                <div style="font-size: 0.8rem; color: #10b981; font-weight: 600; text-transform: uppercase; margin-bottom: 0.75rem;">Performance Analytics</div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.4;">
                    Discipline competitiveness, competitor nation depth, global performance tiers, and historical Olympic benchmark trajectories.
                </div>
            </div>
            <div style="font-size: 0.75rem; color: #64748b; border-top: 1px solid #334155; padding-top: 0.5rem;">
                Default user: <code>coach</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🏆 Enter as Player / Academy", key="btn_quick_coach", use_container_width=True):
            login("coach", ROLE_ACADEMY, "Elite Sports Academy")

    # Custom Credential Login Option
    with st.expander("🔐 Sign In with Custom Credentials"):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            uname = st.text_input("Username", placeholder="e.g. admin, fan, coach", key="login_uname")
        with c2:
            pword = st.text_input("Password", type="password", placeholder="Enter password", key="login_pword")
        with c3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Sign In", key="btn_submit_login", use_container_width=True):
                uname_clean = uname.strip().lower()
                if uname_clean in DEMO_ACCOUNTS and DEMO_ACCOUNTS[uname_clean]["password"] == pword:
                    acc = DEMO_ACCOUNTS[uname_clean]
                    login(uname_clean, acc["role"], acc["name"])
                else:
                    st.error("Invalid username or password. Demo accounts: admin / olympia2024, fan / olympia, coach / olympia")

    # Footer Metadata
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 0.8rem; color: #64748b; border-top: 1px solid #1e293b; padding-top: 1.5rem;">
        OLYMPIA System Architecture: Python 3 • SQLite3 Star Schema • Scikit-Learn • Plotly • Streamlit<br>
        20,240 Verified Podium Medal Records (1896 Athens – 2024 Paris)
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_user_badge():
    """Renders user info and a logout button in the sidebar."""
    if not is_authenticated():
        return

    user = get_current_user()
    role = user["role"]
    role_meta = ROLE_DISPLAY.get(role, ROLE_DISPLAY[ROLE_ENTHUSIAST])

    st.sidebar.markdown(f"""
    <div style="
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">
                {role_meta['icon']} {user['fullname']}
            </div>
            <span style="
                background: {role_meta['badge_color']};
                color: #ffffff;
                font-size: 0.65rem;
                font-weight: 800;
                padding: 0.15rem 0.45rem;
                border-radius: 9999px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            ">
                {role_meta['title']}
            </span>
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
            User ID: <code>{user['username']}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🚪 Log Out", key="sidebar_logout_btn", use_container_width=True):
        logout()
