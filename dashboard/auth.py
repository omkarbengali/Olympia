"""
Authentication and Role-Based Access Control for OLYMPIA.
High-Contrast Modern Dark-Themed Authentication Portal.
Palette:
- #0F0F0F (Pitch Black Canvas)
- #202020 (Charcoal Surface Container)
- #5DD62C (Electric Lime Accent & CTA)
- #337418 (Forest Green Secondary Accent)
- #F8F8F8 (Off-White Primary Typography)
"""

import os
import base64
import textwrap
import streamlit as st
from typing import Optional, Tuple

# Role Constants (Normalized as requested: admin, enthusiast, professional)
ROLE_ADMIN = "admin"
ROLE_ENTHUSIAST = "enthusiast"
ROLE_PROFESSIONAL = "professional"
ROLE_ACADEMY = ROLE_PROFESSIONAL  # Alias for backwards compatibility

ROLE_DISPLAY = {
    ROLE_ADMIN: {
        "title": "Administrator",
        "icon": "🛡️",
        "badge_color": "#5DD62C",
        "badge_bg": "rgba(93, 214, 44, 0.15)",
        "description": "Full access to Data Warehouse, dataset editor, system stats, and administrative controls."
    },
    ROLE_ENTHUSIAST: {
        "title": "Sport Enthusiast",
        "icon": "🎉",
        "badge_color": "#A3E635",
        "badge_bg": "rgba(163, 230, 53, 0.15)",
        "description": "The Podium, OLAP Cube, Data Mining Arena, and Athlete Hall of Fame."
    },
    ROLE_PROFESSIONAL: {
        "title": "Professional / Academy",
        "icon": "🎯",
        "badge_color": "#5DD62C",
        "badge_bg": "rgba(93, 214, 44, 0.15)",
        "description": "Advanced physical telemetry, sports science deep dives, and Predict Your Potential."
    },
    # Backwards compatibility keys
    "ADMIN": {
        "title": "Administrator",
        "icon": "🛡️",
        "badge_color": "#5DD62C",
        "badge_bg": "rgba(93, 214, 44, 0.15)",
        "description": "Full access to Data Warehouse, dataset editor, system stats, and administrative controls."
    },
    "SPORT_ENTHUSIAST": {
        "title": "Sport Enthusiast",
        "icon": "🎉",
        "badge_color": "#A3E635",
        "badge_bg": "rgba(163, 230, 53, 0.15)",
        "description": "The Podium, OLAP Cube, Data Mining Arena, and Athlete Hall of Fame."
    },
    "PLAYER_ACADEMY": {
        "title": "Professional / Academy",
        "icon": "🎯",
        "badge_color": "#5DD62C",
        "badge_bg": "rgba(93, 214, 44, 0.15)",
        "description": "Advanced physical telemetry, sports science deep dives, and Predict Your Potential."
    }
}

DEMO_ACCOUNTS = {
    "admin": {"password": "olympia2024", "role": ROLE_ADMIN, "name": "System Administrator"},
    "fan": {"password": "olympia", "role": ROLE_ENTHUSIAST, "name": "Olympic Fan"},
    "coach": {"password": "olympia", "role": ROLE_PROFESSIONAL, "name": "Elite Sports Academy"},
    "professional": {"password": "olympia", "role": ROLE_PROFESSIONAL, "name": "Performance Analyst"},
    "enthusiast": {"password": "olympia", "role": ROLE_ENTHUSIAST, "name": "Olympic Enthusiast"}
}


def _safe_html(html: str):
    """Renders HTML safely without markdown code-block indentation interpretation."""
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def init_auth():
    """Initializes session state keys for authentication."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
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
    # Support query parameter login convenience (e.g., ?role=admin, ?role=fan, ?role=coach)
    role_param = st.query_params.get("role")
    if role_param and not st.session_state.get("authenticated", False):
        role_map = {
            "admin": ("admin", ROLE_ADMIN, "System Administrator"),
            "fan": ("fan", ROLE_ENTHUSIAST, "Olympic Fan"),
            "enthusiast": ("enthusiast", ROLE_ENTHUSIAST, "Olympic Enthusiast"),
            "coach": ("coach", ROLE_PROFESSIONAL, "Elite Sports Academy"),
            "professional": ("professional", ROLE_PROFESSIONAL, "Performance Analyst"),
            "academy": ("coach", ROLE_PROFESSIONAL, "Elite Sports Academy"),
        }
        if role_param.lower() in role_map:
            u, r, f = role_map[role_param.lower()]
            st.session_state["authenticated"] = True
            st.session_state["logged_in"] = True
            st.session_state["username"] = u
            st.session_state["user_role"] = r
            st.session_state["user_fullname"] = f
            return True

    is_auth = st.session_state.get("authenticated", False)
    st.session_state["logged_in"] = is_auth
    return is_auth


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
    st.session_state["logged_in"] = True
    st.session_state["username"] = username
    st.session_state["user_role"] = role
    st.session_state["user_fullname"] = fullname or DEMO_ACCOUNTS.get(username, {}).get("name", username)
    st.session_state["selected_role_intent"] = None
    if "role" in st.query_params:
        del st.query_params["role"]
    st.rerun()


def logout():
    """Clears authentication session state."""
    init_auth()
    st.session_state["authenticated"] = False
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["user_role"] = None
    st.session_state["user_fullname"] = None
    st.session_state["selected_role_intent"] = None
    if "role" in st.query_params:
        del st.query_params["role"]
    st.rerun()


@st.cache_data
def get_login_background_images_b64() -> list:
    """Reads Olympic background images and returns them as Base64 data URIs."""
    assets_dir = os.path.join(os.path.dirname(__file__), "assets", "backgrounds")
    filenames = [
        "olympic_stadium.png",
        "usain_bolt.png",
        "olympic_flags.png",
        "ancient_olympics.png",
        "olympic_rings.png"
    ]
    b64_list = []
    for fn in filenames:
        p = os.path.join(assets_dir, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                b64_list.append(f"data:image/png;base64,{b64}")
    return b64_list


def render_login_scrolling_background():
    """
    Renders an infinite 3-column continuous scrolling image background
    with heavy blur and opacity, positioned strictly behind the login page.
    Exclusively rendered when the user is unauthenticated.
    """
    if st.session_state.get("logged_in", False):
        return

    images = get_login_background_images_b64()
    if not images:
        return

    # Create 3 staggered lists (duplicated to create a seamless infinite loop at -50% translateY)
    col1_imgs = (images + images)
    col2_imgs = (images[1:] + images[:1]) + (images[1:] + images[:1])
    col3_imgs = (images[2:] + images[:2]) + (images[2:] + images[:2])

    col1_html = "".join([f'<img src="{src}" alt="Olympic Asset">' for src in col1_imgs])
    col2_html = "".join([f'<img src="{src}" alt="Olympic Asset">' for src in col2_imgs])
    col3_html = "".join([f'<img src="{src}" alt="Olympic Asset">' for src in col3_imgs])

    bg_html = f"""
    <style>
        /* 1. 3-COLUMN VERTICAL SCROLL: z-index: -1, position: fixed */
        .login-bg-container {{
            position: fixed;
            top: -5%;
            left: -5%;
            width: 110vw;
            height: 110vh;
            z-index: -1;
            overflow: hidden;
            pointer-events: none;
            display: flex;
            justify-content: space-between;
            gap: 1.8rem;
            /* 3. HEAVY BLUR & OPACITY */
            filter: blur(10px) brightness(0.4);
            opacity: 0.9;
            box-sizing: border-box;
        }}

        .login-bg-col {{
            flex: 1;
            height: 100%;
            overflow: hidden;
            position: relative;
        }}

        .login-scroll-track {{
            display: flex;
            flex-direction: column;
            gap: 1.8rem;
            width: 100%;
            will-change: transform;
        }}

        .login-scroll-track img {{
            width: 100%;
            height: 380px;
            object-fit: cover;
            border-radius: 14px;
            display: block;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.7);
        }}

        /* 2. INFINITE CSS ANIMATION (Staggered 20s, 25s, 30s & Directions Up, Down, Up) */
        .col-scroll-up-1 {{
            animation: scrollInfiniteUp 20s linear infinite;
        }}

        .col-scroll-down {{
            animation: scrollInfiniteDown 25s linear infinite;
        }}

        .col-scroll-up-2 {{
            animation: scrollInfiniteUp 30s linear infinite;
        }}

        @keyframes scrollInfiniteUp {{
            0% {{
                transform: translateY(0);
            }}
            100% {{
                transform: translateY(-50%);
            }}
        }}

        @keyframes scrollInfiniteDown {{
            0% {{
                transform: translateY(-50%);
            }}
            100% {{
                transform: translateY(0%);
            }}
        }}

        /* Ensure .stApp background transparency ONLY during login state so z-index: -1 is visible */
        .stApp {{
            background-color: transparent !important;
        }}
        html, body {{
            background-color: #0F0F0F !important;
        }}
        /* Ensure login cards stay firmly interactive and prominent */
        .main .block-container, div[data-testid="stMainBlockContainer"] {{
            position: relative !important;
            z-index: 10 !important;
        }}
    </style>

    <div class="login-bg-container">
        <div class="login-bg-col">
            <div class="login-scroll-track col-scroll-up-1">
                {col1_html}
            </div>
        </div>
        <div class="login-bg-col">
            <div class="login-scroll-track col-scroll-down">
                {col2_html}
            </div>
        </div>
        <div class="login-bg-col">
            <div class="login-scroll-track col-scroll-up-2">
                {col3_html}
            </div>
        </div>
    </div>
    """

    st.markdown(bg_html, unsafe_allow_html=True)


def render_landing_page():
    """
    Renders the modern, 3-role high-contrast dark authentication portal.
    Users must authenticate before accessing the main dashboard.
    """
    init_auth()

    # Dynamic blurred continuous scrolling background (exclusively for login)
    if not st.session_state.get("logged_in", False):
        render_login_scrolling_background()

    # Hero Banner
    _safe_html("""
    <div style="text-align: center; padding: 2rem 1rem 1.5rem 1rem;">
        <div style="font-size: 3.2rem; line-height: 1;">⚡</div>
        <h1 style="font-size: 2.8rem; font-weight: 800; margin-top: 0.5rem; margin-bottom: 0.2rem; color: #F8F8F8; letter-spacing: -0.03em;">
            OLYMPIA
        </h1>
        <div style="font-size: 1.15rem; color: #5DD62C; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
            Olympic Performance Analytics & Intelligence System
        </div>
        <p style="max-width: 680px; margin: 0.8rem auto 0 auto; color: #A0A0A0; font-size: 1rem; line-height: 1.5;">
            Explore 128 years of modern Olympic history (1896 – 2024) powered by a verified Star Schema Data Warehouse,
            unsupervised clustering, predictive modeling, and multi-dimensional OLAP analytics.
        </p>
    </div>
    """)

    st.markdown("<h3 style='text-align: center; color: #F8F8F8; margin-bottom: 1.5rem;'>Choose Your Portal Experience</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    # 1. ADMIN ROLE CARD
    with col1:
        _safe_html("""
        <div style="background: #202020; border: 1px solid #337418; border-radius: 14px; padding: 1.5rem; height: 320px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div>
                <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🛡️</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8F8F8;">Administrator</div>
                <div style="font-size: 0.8rem; color: #5DD62C; font-weight: 700; text-transform: uppercase; margin-bottom: 0.75rem;">Full System Access</div>
                <div style="font-size: 0.9rem; color: #A0A0A0; line-height: 1.45;">
                    Data Warehouse inspection, ETL health, data quality audits, OLAP cube laboratory, and machine learning models.
                </div>
            </div>
            <div style="font-size: 0.75rem; color: #707070; border-top: 1px solid #2C2C2C; padding-top: 0.5rem;">
                Default user: <code style="color: #5DD62C;">admin</code>
            </div>
        </div>
        """)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Enter as Admin", key="btn_quick_admin", use_container_width=True):
            login("admin", ROLE_ADMIN, "System Administrator")

    # 2. SPORT ENTHUSIAST ROLE CARD
    with col2:
        _safe_html("""
        <div style="background: #202020; border: 1px solid #337418; border-radius: 14px; padding: 1.5rem; height: 320px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div>
                <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🎉</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8F8F8;">Sport Enthusiast</div>
                <div style="font-size: 0.8rem; color: #A3E635; font-weight: 700; text-transform: uppercase; margin-bottom: 0.75rem;">Visual & Intuitive</div>
                <div style="font-size: 0.9rem; color: #A0A0A0; line-height: 1.45;">
                    Engaging Olympic storyboards, country highlights, sport milestones, head-to-head nation showdowns, and medal trends.
                </div>
            </div>
            <div style="font-size: 0.75rem; color: #707070; border-top: 1px solid #2C2C2C; padding-top: 0.5rem;">
                Default user: <code style="color: #A3E635;">fan</code>
            </div>
        </div>
        """)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🌟 Enter as Enthusiast", key="btn_quick_fan", use_container_width=True):
            login("fan", ROLE_ENTHUSIAST, "Olympic Fan")

    # 3. PLAYER / ACADEMY ROLE CARD
    with col3:
        _safe_html("""
        <div style="background: #202020; border: 1px solid #337418; border-radius: 14px; padding: 1.5rem; height: 320px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div>
                <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🎯</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8F8F8;">Player / Academy</div>
                <div style="font-size: 0.8rem; color: #5DD62C; font-weight: 700; text-transform: uppercase; margin-bottom: 0.75rem;">Performance Analytics</div>
                <div style="font-size: 0.9rem; color: #A0A0A0; line-height: 1.45;">
                    Discipline competitiveness, competitor nation depth, global performance tiers, and historical Olympic benchmark trajectories.
                </div>
            </div>
            <div style="font-size: 0.75rem; color: #707070; border-top: 1px solid #2C2C2C; padding-top: 0.5rem;">
                Default user: <code style="color: #5DD62C;">coach</code>
            </div>
        </div>
        """)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🏆 Enter as Player / Academy", key="btn_quick_coach", use_container_width=True):
            login("coach", ROLE_ACADEMY, "Elite Sports Academy")

    # Custom Credential Login Option
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
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
                    st.error("Invalid credentials. Demo accounts: admin / olympia2024, fan / olympia, coach / olympia")

    # Footer Metadata
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    _safe_html("""
    <div style="text-align: center; font-size: 0.8rem; color: #707070; border-top: 1px solid #2C2C2C; padding-top: 1.5rem;">
        OLYMPIA System Architecture: Python 3 • SQLite3 Star Schema • Scikit-Learn • Plotly • Streamlit<br>
        20,240 Verified Podium Medal Records (1896 Athens – 2024 Paris)
    </div>
    """)


def render_sidebar_user_badge():
    """Renders user info and a logout button in the sidebar."""
    if not is_authenticated():
        return

    user = get_current_user()
    role = user["role"]
    role_meta = ROLE_DISPLAY.get(role, ROLE_DISPLAY[ROLE_ENTHUSIAST])

    _safe_html(f"""
    <div style="
        background: #202020;
        border: 1px solid #337418;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-weight: 800; color: #F8F8F8; font-size: 0.95rem;">
                {role_meta['icon']} {user['fullname']}
            </div>
            <span style="
                background: {role_meta.get('badge_bg', 'rgba(93, 214, 44, 0.15)')};
                color: {role_meta['badge_color']};
                border: 1px solid {role_meta['badge_color']};
                font-size: 0.65rem;
                font-weight: 800;
                padding: 0.15rem 0.5rem;
                border-radius: 9999px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            ">
                {role_meta['title']}
            </span>
        </div>
        <div style="font-size: 0.75rem; color: #A0A0A0; margin-top: 0.3rem;">
            User ID: <code style="color: #5DD62C;">{user['username']}</code>
        </div>
    </div>
    """)

    if st.sidebar.button("🚪 Log Out", key="sidebar_logout_btn", use_container_width=True):
        logout()
