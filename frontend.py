import os
from urllib.parse import quote

import streamlit as st
from pypdf import PdfReader
from groq import Groq
from app.retriever import load_knowledge_base, hybrid_retrieve
from app.ingestion import ingest_verified_file
from app.config import ADMIN_API_KEY, GROQ_MODEL


# --- Browser-tab icon (drawn in code so no emoji / image file is needed) ---
def _make_favicon():
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, 63, 63), radius=16, fill=(14, 124, 134, 255))
        d.rectangle((27, 14, 36, 49), fill="white")
        d.rectangle((14, 27, 49, 36), fill="white")
        return img
    except Exception:
        return None


# --- APP CONFIGURATION (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Medibot — AI Specialist & Health Assistant",
    page_icon=_make_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

# API Key Resolution (Streamlit Secrets on Cloud, .env fallback locally)
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


@st.cache_resource(show_spinner="Mounting clinical knowledge base...")
def initialize_knowledge_base():
    load_knowledge_base()
    return True


initialize_knowledge_base()


# Local streaming generator replacing the FastAPI backend endpoint
def generate_clinical_response(query: str, report_text: str = ""):
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        yield "**Configuration error:** GROQ_API_KEY is not configured in environment or Streamlit Secrets."
        return

    search_context = query
    user_report_prompt_text = ""

    if report_text:
        user_report_prompt_text = f"\n[User Attached Lab Report Details]:\n{report_text}\n"
        search_context = f"{query} {report_text[:250]}"

    retrieved_facts = hybrid_retrieve(search_context, top_n=3)

    if not retrieved_facts or not retrieved_facts.strip():
        yield "The verified clinical knowledge base appears unindexed. Please verify storage."
        return

    prompt = (
        f"You are Dr. Medibot, an expert clinical AI assistant.\n"
        f"Analyze the patient's inquiry using the retrieved medical excerpts provided below.\n"
        f"Synthesize the literature and correlate clinical terms.\n"
        f"Only if the topic is completely unaddressed in the text, state that the verified literature does not cover this topic.\n\n"
        f"--- VERIFIED MEDICAL EXCERPTS ---\n{retrieved_facts}\n---------------------------------\n"
        f"{user_report_prompt_text}\n"
        f"Patient Question / Symptoms: {query}\n\n"
        f"Provide a structured clinical response with these sections:\n"
        f"• Potential Clinical Causes (based on retrieved literature)\n"
        f"• Home Care & Next Steps\n"
        f"• Red Flag Warning Signs (When to seek immediate emergency care)\n"
    )

    client = Groq(api_key=GROQ_API_KEY)
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a clinical decision-support AI. Provide safe, literature-grounded medical guidance. Do not use emojis.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# =====================================================================
#  BACKGROUND
#  Default: a built-in medical-themed SVG (soft teal wash, faint crosses,
#  heartbeat line). To use your own photo instead, paste an image URL below.
# =====================================================================
BACKGROUND_IMAGE_URL = ""

_BG_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 900' preserveAspectRatio='xMidYMid slice'>"
    "<defs>"
    "<linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
    "<stop offset='0' stop-color='#F8FCFD'/><stop offset='.55' stop-color='#EBF5F8'/><stop offset='1' stop-color='#DFEFF2'/>"
    "</linearGradient>"
    "<pattern id='p' width='72' height='72' patternUnits='userSpaceOnUse'>"
    "<path d='M36 28v16M28 36h16' stroke='#0E7C86' stroke-width='2' stroke-linecap='round' opacity='.07'/>"
    "</pattern>"
    "</defs>"
    "<rect width='1600' height='900' fill='url(#g)'/>"
    "<rect width='1600' height='900' fill='url(#p)'/>"
    "<circle cx='1400' cy='120' r='330' fill='#0E7C86' opacity='.05'/>"
    "<circle cx='110' cy='830' r='370' fill='#1D6FA5' opacity='.05'/>"
    "<path d='M0 650 H430 L470 650 L500 570 L545 740 L585 620 L615 650 H980 L1020 650 L1050 600 "
    "L1085 700 L1115 650 H1600' fill='none' stroke='#0E7C86' stroke-width='2.5' opacity='.13' "
    "stroke-linejoin='round' stroke-linecap='round'/>"
    "</svg>"
)

if BACKGROUND_IMAGE_URL:
    BG_CSS = (
        "linear-gradient(rgba(243,248,250,.90), rgba(243,248,250,.94)), "
        f'url("{BACKGROUND_IMAGE_URL}") center / cover fixed no-repeat'
    )
else:
    BG_CSS = f'url("data:image/svg+xml,{quote(_BG_SVG)}") center / cover fixed no-repeat, #F3F8FA'


# =====================================================================
#  DESIGN SYSTEM
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:wght@500;600;700&display=swap');

:root {
    --navy: #0B2A43;
    --teal: #0E7C86;
    --teal-dark: #0A5F67;
    --blue: #1D6FA5;
    --text: #24384B;
    --muted: #5B6E80;
    --line: #D3E1E8;
}

/* ---------- Base ---------- */
html, body, .stApp, .stMarkdown, button, input, textarea {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--text);
}
.stApp { background: __BG__; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    max-width: 980px !important;
    padding: 1.4rem 1.2rem 9rem 1.2rem !important;
}

/* =====================================================================
   RESPONSIVE HYBRID NAVBAR
   ===================================================================== */
.navbar-container {
    position: sticky;
    top: 10px;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 6px 20px;
    box-shadow: 0 2px 14px rgba(11, 42, 67, .07);
    margin-bottom: 28px;
}

.brand-section {
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none !important;
}

.brand-mark {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--teal), var(--blue));
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.brand-name {
    font-family: 'Source Serif 4', serif !important;
    font-size: 22px;
    font-weight: 700;
    color: var(--navy);
    white-space: nowrap;
}

/* Desktop text navigation links */
.desktop-links {
    display: flex;
    align-items: center;
    gap: 24px;
}

.nav-link {
    font-size: 14.5px;
    font-weight: 500;
    color: var(--text);
    text-decoration: none !important;
    position: relative;
    padding: 4px 0;
    transition: color 0.2s ease;
}

.nav-link:hover {
    color: var(--teal);
}

.nav-link.active {
    color: var(--teal);
    font-weight: 600;
}

.nav-link.active::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -4px;
    height: 2px;
    background: var(--teal);
    border-radius: 2px;
}

/* Mobile Hamburger Menu */
.mobile-menu {
    display: none;
    position: relative;
}

.mobile-menu summary {
    list-style: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    padding: 0;
    outline: none;
}
.mobile-menu summary::-webkit-details-marker {
    display: none;
}

.hamburger-lines {
    width: 22px;
    height: 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.hamburger-lines span {
    display: block;
    height: 2.2px;
    width: 100%;
    background-color: var(--navy);
    border-radius: 2px;
    transition: background-color 0.2s;
}
.mobile-menu summary:hover .hamburger-lines span {
    background-color: var(--teal);
}

.mobile-dropdown {
    position: absolute;
    top: 42px;
    right: 0;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 12px;
    box-shadow: 0 10px 25px rgba(11, 42, 67, 0.15);
    min-width: 170px;
    padding: 8px 0;
    display: flex;
    flex-direction: column;
    z-index: 10000;
}

.mobile-dropdown a {
    padding: 10px 18px;
    font-size: 14.5px;
    font-weight: 500;
    color: var(--text);
    text-decoration: none !important;
    transition: background 0.15s;
}

.mobile-dropdown a:hover {
    background: #F2F9FD;
    color: var(--teal);
}

.mobile-dropdown a.active {
    color: var(--teal);
    font-weight: 600;
    background: #EAF5F8;
}

/* Switch from desktop links to hamburger below 768px */
@media (max-width: 768px) {
    .desktop-links {
        display: none !important;
    }
    .mobile-menu {
        display: block !important;
    }
}

/* ---------- Home hero ---------- */
.hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 34px 12px 6px 12px;
}
.hero-mark {
    width: 62px; height: 62px;
    border-radius: 16px;
    background: linear-gradient(135deg, var(--teal), var(--blue));
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 24px;
    box-shadow: 0 10px 24px rgba(14, 124, 134, .26);
}
.hero-title {
    font-family: 'Source Serif 4', serif !important;
    font-size: 42px;
    font-weight: 600;
    line-height: 1.15;
    letter-spacing: -.5px;
    color: var(--navy);
    margin: 0 0 14px 0;
}
.hero-sub {
    max-width: 500px;
    font-size: 16px;
    line-height: 1.65;
    color: var(--muted);
    margin: 0;
    text-align: center;
}
.hero-gap { height: 26px; }

/* ---------- Conversation toolbar ---------- */
.tool-label {
    display: flex; align-items: center;
    height: 42px;
    font-size: 15px; font-weight: 600;
    color: var(--navy);
}
.attach-chip {
    display: inline-flex; align-items: center; gap: 8px;
    height: 42px;
    max-width: 100%;
    padding: 0 14px;
    font-size: 13.5px;
    color: var(--teal-dark);
    background: #E4F2F3;
    border: 1px solid #BFDDE0;
    border-radius: 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ---------- Chat ---------- */
.stChatMessage {
    max-width: 820px;
    margin: 0 auto 12px auto;
    background: #FFFFFF !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 4px rgba(11, 42, 67, .05) !important;
    line-height: 1.7;
}
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]),
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) { background: #EEF6F8 !important; }
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-assistant"] { background: var(--teal) !important; }
.verified-chip {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--teal-dark);
    background: #E4F2F3;
    border: 1px solid #BFDDE0;
    border-radius: 999px;
}

[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stBottomBlockContainer"] { padding-bottom: 46px !important; }
[data-testid="stChatInput"] {
    max-width: 820px;
    margin: 0 auto;
    border-radius: 16px;
    border: 1px solid #BCD0DA;
    background: #FFFFFF;
    box-shadow: 0 6px 24px rgba(11, 42, 67, .10);
    transition: border-color .2s ease, box-shadow .2s ease;
}
[data-testid="stChatInput"] > div { background: #FFFFFF; border-radius: 16px; }
[data-testid="stChatInput"]:focus-within {
    border-color: var(--teal);
    box-shadow: 0 0 0 3px rgba(14, 124, 134, .16), 0 6px 24px rgba(11, 42, 67, .12);
}
[data-testid="stChatInput"] textarea { font-size: 15px; background: transparent; }

/* ---------- Inputs / containers ---------- */
div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--line) !important;
    border-radius: 12px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-radius: 14px;
    box-shadow: 0 1px 4px rgba(11, 42, 67, .05);
}
div[data-testid="stTextInput"] input { border-radius: 8px; }

/* ---------- Inner pages ---------- */
.page-head { margin: 4px 0 22px 0; }
.page-title {
    font-family: 'Source Serif 4', serif !important;
    font-size: 36px;
    font-weight: 600;
    letter-spacing: -.5px;
    line-height: 1.15;
    color: var(--navy);
    margin: 0 0 10px 0;
}
.page-sub { font-size: 15.5px; line-height: 1.65; color: var(--muted); max-width: 600px; margin: 0; }
.page-rule { width: 64px; height: 3px; border-radius: 2px; margin-top: 18px; background: linear-gradient(90deg, var(--teal), var(--blue)); }

.panel {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 26px 30px;
    margin-bottom: 18px;
    box-shadow: 0 1px 4px rgba(11, 42, 67, .05);
}
.panel-title {
    font-family: 'Source Serif 4', serif !important;
    font-size: 21px;
    font-weight: 600;
    color: var(--navy);
    margin: 0 0 6px 0;
}
.panel-text { font-size: 15px; line-height: 1.75; color: var(--muted); margin: 0; }

.step {
    display: grid;
    grid-template-columns: 40px 1fr;
    gap: 16px;
    padding: 16px 0;
    border-top: 1px solid var(--line);
}
.step:first-of-type { margin-top: 14px; }
.step-no {
    width: 32px; height: 32px;
    border: 1.5px solid var(--teal);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 600;
    color: var(--teal);
}
.step-title { font-size: 16px; font-weight: 600; color: var(--navy); margin: 2px 0 4px 0; }
.step-text { font-size: 14.5px; line-height: 1.65; color: var(--muted); margin: 0; }

.facts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 26px; margin-top: 18px; }
.fact { border-top: 3px solid var(--teal); padding-top: 14px; }
.fact-title { font-size: 15.5px; font-weight: 600; color: var(--navy); margin: 0 0 6px 0; }
.fact-text { font-size: 14px; line-height: 1.65; color: var(--muted); margin: 0; }

.contact-row {
    display: grid;
    grid-template-columns: 190px 1fr;
    gap: 16px;
    align-items: center;
    padding: 18px 0;
    border-top: 1px solid var(--line);
}
.contact-row:first-of-type { margin-top: 14px; }
.contact-label { font-size: 13.5px; font-weight: 600; color: var(--muted); }
.contact-value { font-size: 16px; color: var(--navy); font-weight: 500; word-break: break-word; }
.contact-value a { color: var(--teal-dark); text-decoration: none; border-bottom: 1px solid rgba(14, 124, 134, .35); }
.contact-value a:hover { border-bottom-color: var(--teal); }

/* ---------- Footer ---------- */
.site-footer {
    text-align: center;
    font-size: 12px;
    line-height: 1.6;
    color: var(--muted);
    margin-top: 44px;
    padding-top: 18px;
    border-top: 1px solid var(--line);
}
.site-footer.fixed {
    position: fixed;
    left: 0; right: 0; bottom: 0;
    margin: 0;
    padding: 9px 14px;
    background: rgba(255, 255, 255, .97);
    z-index: 1000;
}

/* ---------- Tablet ---------- */
@media (max-width: 992px) {
    .block-container { padding: 1rem 1rem 9rem 1rem !important; }
    .hero-title { font-size: 36px; }
    .page-title { font-size: 31px; }
    .facts { gap: 18px; }
}

/* ---------- Mobile & Responsive Navbar ---------- */
/* By default on desktop: hide the hamburger wrapper */
div.mobile-menu-wrapper {
    display: none !important;
}

@media (max-width: 768px) {
    .block-container { padding: .7rem .75rem 9rem .75rem !important; }

    /* Single clean row for header */
    div[data-testid="stHorizontalBlock"]:has(.brand-logo) {
        position: sticky;
        top: 8px;
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        align-items: center !important;
        flex-wrap: nowrap !important;
        padding: 6px 14px !important;
        margin-bottom: 20px !important;
    }

    /* Logo stays on the left */
    div[data-testid="stHorizontalBlock"]:has(.brand-logo) > div:first-child {
        flex: 1 1 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }

    /* Hide the 4 inline desktop buttons on small screens */
    div[data-testid="stHorizontalBlock"]:has(.brand-logo) > div.desktop-nav-link {
        display: none !important;
    }

    /* Show the hamburger menu on the right end */
    div.mobile-menu-wrapper {
        display: block !important;
        flex: 0 0 auto !important;
    }

    /* Style the hamburger popover trigger button */
    div.mobile-menu-wrapper div[data-testid="stPopover"] > button {
        background: transparent !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        min-height: 38px !important;
        padding: 4px 10px !important;
        box-shadow: none !important;
        color: var(--navy) !important;
    }
    div.mobile-menu-wrapper div[data-testid="stPopover"] > button:hover {
        border-color: var(--teal) !important;
        background: #F2F9FD !important;
    }

    /* Stacking for cards and panels */
    div[data-testid="stHorizontalBlock"]:not(:has(.brand-logo)) { flex-wrap: wrap; gap: .75rem; }
    div[data-testid="stHorizontalBlock"]:not(:has(.brand-logo)) > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"]:not(:has(.brand-logo)) > div[data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    .hero { padding-top: 14px; }
    .hero-mark { width: 54px; height: 54px; margin-bottom: 18px; }
    .hero-title { font-size: 28px; letter-spacing: -.3px; }
    .hero-sub { font-size: 14.5px; }
    [class*="st-key-sg_"] button { min-height: 0; padding: 14px 16px; }
    .stChatMessage { padding: 12px 14px !important; }
    .page-title { font-size: 26px; }
    .page-sub { font-size: 14.5px; }
    .panel { padding: 20px 18px; }
    .facts { grid-template-columns: 1fr; gap: 20px; }
    .contact-row { grid-template-columns: 1fr; gap: 4px; padding: 14px 0; }
    .site-footer.fixed { font-size: 10.5px; padding: 7px 10px; }
    [data-testid="stBottomBlockContainer"] { padding-bottom: 56px !important; }
}

@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
"""
st.markdown(CSS.replace("__BG__", BG_CSS), unsafe_allow_html=True)

# =====================================================================
#  SESSION STATE
# =====================================================================
PAGES = ["Home", "About", "Admin Portal", "Contact"]
PLACEHOLDER = "Ask a clinical question or describe your symptoms..."

st.session_state.setdefault("nav_page", PAGES[0])
st.session_state.setdefault("messages", [])
st.session_state.setdefault("user_report_text", "")
st.session_state.setdefault("active_report_name", "")
st.session_state.setdefault("uploader_key", 0)


def go(page: str):
    st.session_state.nav_page = page


def ask(query: str):
    st.session_state.pending_query = query


def clear_report():
    st.session_state.user_report_text = ""
    st.session_state.active_report_name = ""
    st.session_state.uploader_key += 1


def new_chat():
    st.session_state.messages = []
    clear_report()


# =====================================================================
#  NAVBAR (Native Responsive HTML with Auto-Switching Hamburger)
# =====================================================================
# Sync active page with URL query parameters for clean page navigation
qp = st.query_params.get("page", PAGES[0])
if qp in PAGES:
    st.session_state.nav_page = qp
current_page = st.session_state.nav_page

# Build Desktop Links HTML
desktop_links_html = "".join([
    f'<a href="?page={p}" target="_self" class="nav-link {"active" if p == current_page else ""}">{p}</a>'
    for p in PAGES
])

# Build Mobile Dropdown Links HTML
mobile_links_html = "".join([
    f'<a href="?page={p}" target="_self" class="{"active" if p == current_page else ""}">{p}</a>'
    for p in PAGES
])

# Render the Single-Row Navbar Container
st.markdown(
    f"""
    <div class="navbar-container">
        <!-- Left: Brand Logo & Title -->
        <a href="?page=Home" target="_self" class="brand-section">
            <div class="brand-mark">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3.2" stroke-linecap="round">
                    <path d="M12 4v16M4 12h16"/>
                </svg>
            </div>
            <span class="brand-name">Medibot</span>
        </a>

        <!-- Right (Desktop): Clean Text Links -->
        <nav class="desktop-links">
            {desktop_links_html}
        </nav>

        <!-- Right (Mobile only): Borderless 3-Line Hamburger Dropdown -->
        <div class="mobile-menu">
            <details>
                <summary aria-label="Toggle menu">
                    <div class="hamburger-lines">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </summary>
                <div class="mobile-dropdown">
                    {mobile_links_html}
                </div>
            </details>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================================
#  SHARED HELPERS
# =====================================================================
def page_header(title: str, subtitle: str):
    st.markdown(
        f'<div class="page-head"><div class="page-title">{title}</div>'
        f'<div class="page-sub">{subtitle}</div><div class="page-rule"></div></div>',
        unsafe_allow_html=True,
    )


def footer(fixed: bool = False):
    cls = "site-footer fixed" if fixed else "site-footer"
    st.markdown(
        f'<div class="{cls}">© 2026 <b>Medibot Healthcare Systems Inc.</b> All rights reserved. '
        'Clinical content is provided for educational decision support.</div>',
        unsafe_allow_html=True,
    )


def _field(obj, name, default=None):
    try:
        return getattr(obj, name)
    except AttributeError:
        try:
            return obj[name]
        except Exception:
            return default


def attach_report(uploaded):
    """Read a PDF lab report into session memory."""
    try:
        reader = PdfReader(uploaded)
        st.session_state.user_report_text = "".join(
            [page.extract_text() or "" for page in reader.pages]
        )
        st.session_state.active_report_name = uploaded.name
    except Exception as ex:
        st.error(f"Could not read that PDF: {ex}")


CLIP_SVG = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M21.4 11.1l-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2'
    'a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5"/></svg>'
)


# =====================================================================
#  PAGE: HOME  (chat, in the style of a standard LLM assistant)
# =====================================================================
def page_home():
    # Chat bar. On current Streamlit it has a built-in paperclip for attaching a PDF.
    supports_files = True
    try:
        submission = st.chat_input(PLACEHOLDER, accept_file=True, file_type=["pdf"])
    except TypeError:
        supports_files = False
        submission = st.chat_input(PLACEHOLDER)

    typed_text, files = "", []
    if submission:
        if supports_files:
            typed_text = _field(submission, "text", "") or ""
            files = _field(submission, "files", []) or []
        else:
            typed_text = submission
    typed_text = typed_text.strip()

    if files:
        attach_report(files[0])

    query = typed_text or st.session_state.pop("pending_query", None)
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

    has_chat = bool(st.session_state.messages)
    has_report = bool(st.session_state.active_report_name)

    # Older Streamlit versions: fall back to an upload box
    if not supports_files:
        with st.expander("Attach a lab report (PDF, optional)", expanded=False):
            uploaded_doc = st.file_uploader(
                "Upload lab report (PDF):",
                type=["pdf"],
                key=f"report_pdf_{st.session_state.uploader_key}",
            )
            if uploaded_doc and uploaded_doc.name != st.session_state.active_report_name:
                attach_report(uploaded_doc)

    # Attached-report status
    if has_report:
        c_chip, c_rm = st.columns([4, 1.2])
        with c_chip:
            st.markdown(
                f'<div class="attach-chip">{CLIP_SVG}<span>{st.session_state.active_report_name}</span></div>',
                unsafe_allow_html=True,
            )
        with c_rm:
            st.button("Remove report", key="rm_report", on_click=clear_report)

    # Conversation header
    if has_chat:
        c_lbl, c_new = st.columns([4, 1.2])
        with c_lbl:
            st.markdown('<div class="tool-label">Consultation</div>', unsafe_allow_html=True)
        with c_new:
            st.button("New chat", key="new_chat", on_click=new_chat)

    # Empty state
    if not has_chat:
        st.markdown(
            '<div class="hero">'
            '<div class="hero-mark"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" '
            'stroke-width="3" stroke-linecap="round"><path d="M12 4v16M4 12h16"/></svg></div>'
            '<div class="hero-title">How are you feeling today?</div>'
            '<div class="hero-sub">Describe your symptoms or ask a health question. Medibot answers from '
            'verified clinical books, not random web pages.</div>'
            '</div><div class="hero-gap"></div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button(
                "Severe one-sided headache with light sensitivity",
                key="sg_1",
                on_click=ask,
                args=("Severe throbbing unilateral headache with photophobia",),
            )
        with c2:
            st.button(
                "How to care for a minor burn at home",
                key="sg_2",
                on_click=ask,
                args=("Home care and management for minor thermal burn",),
            )
        with c3:
            st.button(
                "What do high HbA1c levels mean?",
                key="sg_3",
                on_click=ask,
                args=("Explain high HbA1c levels and recommended diet",),
            )

    # History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # New answer
    if query:
        with st.chat_message("assistant"):
            response_box = st.empty()
            full_response = ""
            report_context = st.session_state.user_report_text or ""
            try:
                for token in generate_clinical_response(query=query, report_text=report_context):
                    full_response += token
                    response_box.markdown(full_response + " ▌")

                final_display = full_response + "<br><span class='verified-chip'>Verified reference match</span>"
                response_box.markdown(final_display, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": final_display})
            except Exception as e:
                response_box.error(f"Clinical inference error: {e}")

    footer(fixed=True)


# =====================================================================
#  PAGE: ABOUT
# =====================================================================
def page_about():
    page_header(
        "About Medibot",
        "Bridging the gap between medical textbooks and everyday patient questions.",
    )
    st.markdown(
        '<div class="panel"><div class="panel-text">'
        'Medibot compares your question directly against verified clinical volumes instead of scouring random '
        'web articles. It then explains the potential causes, safe home care, and the warning signs that call '
        'for urgent medical attention.</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel"><div class="panel-title">How it works</div>'
        '<div class="step"><div class="step-no">1</div><div>'
        '<div class="step-title">You ask</div>'
        '<div class="step-text">Type your symptoms or question. You can also attach a lab report as a PDF.</div></div></div>'
        '<div class="step"><div class="step-no">2</div><div>'
        '<div class="step-title">Medibot searches the library</div>'
        '<div class="step-text">It finds the most relevant passages in the verified medical reference books.</div></div></div>'
        '<div class="step"><div class="step-no">3</div><div>'
        '<div class="step-title">You receive a structured answer</div>'
        '<div class="step-text">The answer streams in, written from those passages in plain language.</div></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel"><div class="panel-title">Every answer includes</div>'
        '<div class="facts">'
        '<div class="fact"><div class="fact-title">Potential causes</div>'
        '<div class="fact-text">What the literature says could explain the symptoms.</div></div>'
        '<div class="fact"><div class="fact-title">Home care and next steps</div>'
        '<div class="fact-text">Safe measures you can take now, and what to do next.</div></div>'
        '<div class="fact"><div class="fact-title">Red flag warning signs</div>'
        '<div class="fact-text">When to seek emergency care immediately.</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.button("Start a consultation", key="about_cta", on_click=go, args=("Home",))
    footer()


# =====================================================================
#  PAGE: ADMIN PORTAL
# =====================================================================
def page_admin():
    page_header(
        "Admin Portal",
        "Authorized personnel only. Upload official medical textbooks to update the clinical knowledge repository. "
        "Uploaded books are stored permanently and used in all future consultations.",
    )

    with st.container(border=True):
        admin_pass = st.text_input("Admin secret key", type="password", placeholder="Enter your admin key")
        admin_file = st.file_uploader("Clinical textbook (PDF):", type=["pdf"], key="admin_reference_file")

        if st.button("Upload to knowledge base", key="admin_upload"):
            if not admin_pass:
                st.error("Admin credentials required.")
            elif admin_pass != ADMIN_API_KEY:
                st.error("Error 403: Forbidden - Invalid Admin Token.")
            elif not admin_file:
                st.warning("Please choose a medical textbook PDF.")
            else:
                temp_path = f"temp_{admin_file.name}"
                with open(temp_path, "wb") as buffer:
                    buffer.write(admin_file.getbuffer())

                try:
                    with st.spinner("Processing and indexing textbook into the verified knowledge repository..."):
                        chunks = ingest_verified_file(temp_path)
                        load_knowledge_base()
                    st.success(
                        f"Success! Book '{admin_file.name}' permanently saved to disk and loaded into RAM. "
                        f"(Indexed chunks: {chunks})"
                    )
                except Exception as ex:
                    st.error(f"Failed to ingest document: {ex}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    footer()


# =====================================================================
#  PAGE: CONTACT
# =====================================================================
def page_contact():
    page_header(
        "Contact",
        "For clinical inquiries, partnership discussions, or support regarding the Medibot intelligence system.",
    )
    st.markdown(
        '<div class="panel">'
        '<div class="contact-row"><div class="contact-label">Email</div>'
        '<div class="contact-value"><a href="mailto:priyanshuchand7037@gmail.com">priyanshuchand7037@gmail.com</a></div></div>'
        '<div class="contact-row"><div class="contact-label">Headquarters</div>'
        '<div class="contact-value">Noida</div></div>'
        '<div class="contact-row"><div class="contact-label">Phone</div>'
        '<div class="contact-value"><a href="tel:+917037524005">+91 7037524005</a></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.button("Back to chat", key="contact_cta", on_click=go, args=("Home",))
    footer()


# =====================================================================
#  ROUTER
# =====================================================================
if current_page == "Home":
    page_home()
elif current_page == "About":
    page_about()
elif current_page == "Admin Portal":
    page_admin()
else:
    page_contact()