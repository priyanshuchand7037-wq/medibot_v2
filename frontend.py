import os
import streamlit as st
from pypdf import PdfReader
from groq import Groq
from app.retriever import load_knowledge_base, hybrid_retrieve
from app.ingestion import ingest_verified_file
from app.config import ADMIN_API_KEY, GROQ_MODEL

# --- APP CONFIGURATION (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Medibot — AI Specialist & Health Assistant",
    page_icon="🩺",
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
        yield "❌ **Configuration Error:** GROQ_API_KEY is not configured in environment or Streamlit Secrets."
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
            {"role": "system", "content": "You are a clinical decision-support AI. Provide safe, literature-grounded medical guidance."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# =====================================================================
#  DESIGN SYSTEM  (gradients, glass cards, responsive media queries)
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #1B2140;
    --muted: #66708F;
    --blue: #3DB4FE;
    --violet: #7B6BFF;
    --mint: #4CECB8;
    --line: rgba(110, 130, 200, 0.20);
    --glass: rgba(255, 255, 255, 0.74);
    --grad-main: linear-gradient(120deg, #2E8BFF 0%, #7B6BFF 55%, #12A6C9 100%);
    --grad-soft: linear-gradient(120deg, rgba(61,180,254,.16), rgba(123,107,255,.14), rgba(76,236,184,.18));
    --shadow: 0 10px 34px rgba(60, 80, 160, 0.10);
}

/* ---------- Base ---------- */
html, body, .stApp, .stMarkdown, button, input, textarea {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--ink);
}
.stApp {
    background: linear-gradient(125deg, #E7F5FF 0%, #F1EEFF 38%, #E4FFF5 70%, #EAF4FF 100%);
    background-size: 300% 300%;
    animation: bgdrift 22s ease infinite;
}
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    z-index: 0;
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
}
.stApp::before { width: 420px; height: 420px; top: -120px; right: -100px; background: rgba(123,107,255,.28); }
.stApp::after  { width: 380px; height: 380px; bottom: -120px; left: -100px; background: rgba(76,236,184,.30); }
@keyframes bgdrift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    position: relative;
    z-index: 1;
    max-width: 980px !important;
    padding: 1rem 1.2rem 6rem 1.2rem !important;
}

/* ---------- Navbar ---------- */
div[data-testid="stHorizontalBlock"]:has(.brand-logo) {
    position: sticky;
    top: 10px;
    z-index: 999;
    align-items: center;
    background: rgba(255,255,255,.80);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 8px 18px;
    box-shadow: var(--shadow);
    margin-bottom: 26px;
}
.brand-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -.4px;
    white-space: nowrap;
}
.brand-mark {
    width: 34px; height: 34px;
    border-radius: 11px;
    background: var(--grad-main);
    display: grid; place-items: center;
    box-shadow: 0 6px 16px rgba(80,100,255,.35);
}
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 6px;
    justify-content: flex-end;
    flex-wrap: nowrap;
    overflow-x: auto;
    scrollbar-width: none;
}
div[data-testid="stRadio"] div[role="radiogroup"]::-webkit-scrollbar { display: none; }
div[data-testid="stRadio"] div[role="radiogroup"] > label {
    margin: 0;
    padding: 9px 18px;
    border-radius: 999px;
    cursor: pointer;
    white-space: nowrap;
    transition: background .2s ease, transform .2s ease, box-shadow .2s ease;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child { display: none; }
div[data-testid="stRadio"] div[role="radiogroup"] > label p {
    font-size: 14px; font-weight: 600; color: var(--muted); margin: 0;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: rgba(61,180,254,.14);
    transform: translateY(-1px);
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: var(--grad-main);
    box-shadow: 0 6px 16px rgba(80,100,255,.32);
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p { color: #fff; }

/* ---------- Home: LLM-style hero ---------- */
.hero { text-align: center; padding: 34px 10px 10px 10px; }
.orb {
    width: 84px; height: 84px;
    margin: 0 auto 22px auto;
    border-radius: 50%;
    display: grid; place-items: center;
    font-size: 38px;
    background: conic-gradient(from 0deg, #3DB4FE, #7B6BFF, #4CECB8, #3DB4FE);
    box-shadow: 0 0 0 10px rgba(123,107,255,.10), 0 0 50px rgba(61,180,254,.45);
    animation: orbspin 9s linear infinite, orbpulse 3.2s ease-in-out infinite;
}
.orb span { display: block; animation: orbcounter 9s linear infinite; }
@keyframes orbspin { to { transform: rotate(360deg); } }
@keyframes orbcounter { to { transform: rotate(-360deg); } }
@keyframes orbpulse { 0%,100%{ filter: saturate(1);} 50%{ filter: saturate(1.35) brightness(1.05);} }
.hero-title {
    font-size: 44px; font-weight: 800; letter-spacing: -1.3px; line-height: 1.12;
    margin: 0 0 12px 0;
    background: linear-gradient(100deg, #2E8BFF 0%, #7B6BFF 50%, #0EA88A 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub { font-size: 16px; color: var(--muted); max-width: 540px; margin: 0 auto 6px auto; line-height: 1.65; }
.disclaimer { font-size: 12px; color: var(--muted); text-align: center; margin: 18px auto 0 auto; max-width: 560px; }

/* ---------- Buttons ---------- */
.stButton > button {
    width: 100%;
    border-radius: 16px;
    border: 1px solid var(--line);
    background: var(--glass);
    color: var(--ink);
    padding: 16px 18px;
    min-height: 78px;
    font-weight: 600;
    text-align: left;
    box-shadow: var(--shadow);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease, background .2s ease;
}
.stButton > button p { font-size: 14px; line-height: 1.45; white-space: normal; }
.stButton > button:hover {
    transform: translateY(-3px);
    border-color: var(--blue);
    background: var(--grad-soft);
    box-shadow: 0 14px 30px rgba(61,130,255,.18);
    color: var(--ink);
}
.stButton > button[kind="primary"] {
    min-height: 0;
    padding: 11px 24px;
    border: none;
    border-radius: 999px;
    background: var(--grad-main);
    color: #fff;
    text-align: center;
    box-shadow: 0 8px 22px rgba(80,100,255,.35);
}
.stButton > button[kind="primary"] p { color: #fff; font-weight: 700; }
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.01);
    background: var(--grad-main);
    box-shadow: 0 12px 28px rgba(80,100,255,.45);
}

/* ---------- Chat ---------- */
.stChatMessage {
    max-width: 820px;
    margin: 0 auto 14px auto;
    background: var(--glass) !important;
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    padding: 16px 20px !important;
    box-shadow: var(--shadow) !important;
}
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]),
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(120deg, rgba(61,180,254,.20), rgba(123,107,255,.18)) !important;
}
.verified-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(76,236,184,.20);
    color: #0b7a55;
    font-size: 12px; font-weight: 700;
    padding: 4px 11px; border-radius: 999px;
    margin-top: 10px;
}
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stChatInput"] {
    max-width: 820px;
    margin: 0 auto;
    border-radius: 28px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,.92);
    box-shadow: 0 12px 40px rgba(70,90,200,.18);
    transition: box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--violet);
    box-shadow: 0 0 0 4px rgba(123,107,255,.16), 0 12px 40px rgba(70,90,200,.22);
}
[data-testid="stChatInput"] textarea { font-size: 15px; }

/* ---------- Expander / bordered containers / inputs ---------- */
div[data-testid="stExpander"] {
    background: var(--glass);
    border: 1px solid var(--line) !important;
    border-radius: 16px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--glass);
    border-radius: 22px;
    box-shadow: var(--shadow);
}
div[data-testid="stTextInput"] input { border-radius: 12px; }

/* ---------- Inner pages ---------- */
.page-head {
    background: var(--grad-main);
    color: #fff;
    border-radius: 26px;
    padding: 36px 38px;
    margin-bottom: 22px;
    box-shadow: 0 18px 44px rgba(80,100,255,.30);
    position: relative;
    overflow: hidden;
}
.page-head::after {
    content: ""; position: absolute; right: -60px; top: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,.16);
}
.page-head h1 { margin: 0 0 8px 0; font-size: 34px; font-weight: 800; letter-spacing: -.8px; color: #fff; }
.page-head p { margin: 0; font-size: 15px; opacity: .95; max-width: 560px; line-height: 1.6; color: #fff; }

.card {
    background: var(--glass);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 26px 28px;
    box-shadow: var(--shadow);
    margin-bottom: 18px;
    transition: transform .25s ease, box-shadow .25s ease;
}
.card:hover { transform: translateY(-4px); box-shadow: 0 18px 40px rgba(70,100,220,.16); }
.card h3 { margin: 0 0 8px 0; font-size: 18px; font-weight: 800; }
.card p  { margin: 0; font-size: 14px; color: var(--muted); line-height: 1.7; }
.card a  { text-decoration: none; font-weight: 700; color: #2E7BFF; word-break: break-all; }
.step-no {
    width: 38px; height: 38px; border-radius: 12px;
    background: var(--grad-main); color: #fff;
    display: grid; place-items: center;
    font-weight: 800; margin-bottom: 14px;
}
.icon-bubble {
    width: 46px; height: 46px; border-radius: 15px;
    background: var(--grad-main);
    display: grid; place-items: center;
    margin-bottom: 14px;
    box-shadow: 0 8px 18px rgba(80,100,255,.30);
}
.section-title { font-size: 20px; font-weight: 800; margin: 26px 0 14px 0; letter-spacing: -.3px; }

.site-footer {
    text-align: center;
    margin-top: 48px;
    font-size: 12.5px;
    color: var(--muted);
    line-height: 1.7;
}

/* ---------- Tablet ---------- */
@media (max-width: 992px) {
    .block-container { padding: .8rem 1rem 6rem 1rem !important; }
    .hero-title { font-size: 36px; }
    .page-head { padding: 30px 28px; }
    .page-head h1 { font-size: 29px; }
    div[data-testid="stHorizontalBlock"]:not(:has(.brand-logo)) { flex-wrap: wrap; gap: 1rem; }
    div[data-testid="stHorizontalBlock"]:not(:has(.brand-logo)) > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"]:not(:has(.brand-logo)) > div[data-testid="column"] {
        flex: 1 1 calc(50% - 1rem) !important;
        min-width: calc(50% - 1rem) !important;
    }
}

/* ---------- Mobile ---------- */
@media (max-width: 640px) {
    .block-container { padding: .6rem .7rem 6rem .7rem !important; }
    div[data-testid="stHorizontalBlock"]:has(.brand-logo) {
        position: static;
        flex-wrap: wrap;
        gap: .4rem;
        padding: 10px 12px;
        border-radius: 18px;
    }
    div[data-testid="stHorizontalBlock"]:has(.brand-logo) > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"]:has(.brand-logo) > div[data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] { justify-content: flex-start; }
    div[data-testid="stRadio"] div[role="radiogroup"] > label { padding: 8px 14px; }
    div[data-testid="stHorizontalBlock"]:not(:has(.brand-logo)) > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"]:not(:has(.brand-logo)) > div[data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    .hero { padding-top: 14px; }
    .orb { width: 68px; height: 68px; font-size: 30px; }
    .hero-title { font-size: 28px; letter-spacing: -.8px; }
    .hero-sub { font-size: 14px; }
    .stButton > button { min-height: 0; padding: 14px 16px; }
    .stChatMessage { padding: 12px 14px !important; border-radius: 16px !important; }
    .page-head { padding: 24px 20px; border-radius: 20px; }
    .page-head h1 { font-size: 24px; }
    .card { padding: 20px; border-radius: 18px; }
    .brand-logo { font-size: 19px; }
}

@media (prefers-reduced-motion: reduce) {
    .stApp, .orb, .orb span { animation: none !important; }
    * { transition: none !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =====================================================================
#  SESSION STATE
# =====================================================================
PAGES = ["🏠 Home", "ℹ️ About", "🛡️ Admin Portal", "✉️ Contact"]

st.session_state.setdefault("nav_page", PAGES[0])
st.session_state.setdefault("messages", [])
st.session_state.setdefault("user_report_text", "")
st.session_state.setdefault("active_report_name", "")
st.session_state.setdefault("uploader_key", 0)


def go(page: str):
    """Callback used by in-page buttons to switch the navbar page."""
    st.session_state.nav_page = page


def ask(query: str):
    """Callback used by the suggestion cards."""
    st.session_state.pending_query = query


def new_chat():
    st.session_state.messages = []
    st.session_state.user_report_text = ""
    st.session_state.active_report_name = ""
    st.session_state.uploader_key += 1


# =====================================================================
#  NAVBAR  (each button is its own page)
# =====================================================================
nav_brand, nav_menu = st.columns([1, 3.2])
with nav_brand:
    st.markdown(
        '<div class="brand-logo"><div class="brand-mark">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round">'
        '<path d="M12 4v16M4 12h16"/></svg></div>Medibot</div>',
        unsafe_allow_html=True,
    )
with nav_menu:
    st.radio(
        "Navigate",
        PAGES,
        horizontal=True,
        key="nav_page",
        label_visibility="collapsed",
    )


def page_header(title: str, subtitle: str):
    st.markdown(
        f'<div class="page-head"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        '<div class="site-footer">© 2026 <b>Medibot Healthcare Systems Inc.</b> All rights reserved.<br>'
        'Clinical content is provided for educational decision support.</div>',
        unsafe_allow_html=True,
    )


# =====================================================================
#  PAGE 1 — HOME  (standard LLM-style chat)
# =====================================================================
def page_home():
    typed = st.chat_input("Ask a clinical question or describe your symptoms...")
    query = typed or st.session_state.pop("pending_query", None)

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

    has_chat = bool(st.session_state.messages)

    # Toolbar: optional lab report + new chat
    tool_l, tool_r = st.columns([3, 1])
    with tool_l:
        label = (
            f"📎 Report attached: {st.session_state.active_report_name}"
            if st.session_state.active_report_name
            else "📎 Attach a lab report (optional)"
        )
        with st.expander(label, expanded=False):
            uploaded_doc = st.file_uploader(
                "Upload lab report (PDF):",
                type=["pdf"],
                key=f"report_pdf_{st.session_state.uploader_key}",
            )
            if uploaded_doc and uploaded_doc.name != st.session_state.active_report_name:
                reader = PdfReader(uploaded_doc)
                st.session_state.user_report_text = "".join(
                    [page.extract_text() or "" for page in reader.pages]
                )
                st.session_state.active_report_name = uploaded_doc.name
                st.success(f"Attached: {uploaded_doc.name}")
    with tool_r:
        if has_chat:
            st.button("✨ New chat", type="primary", on_click=new_chat)

    # Empty state: greeting + suggestion cards
    if not has_chat:
        st.markdown(
            '<div class="hero">'
            '<div class="orb"><span>🩺</span></div>'
            '<h1 class="hero-title">How are you feeling today?</h1>'
            '<p class="hero-sub">Describe your symptoms or ask a health question. '
            'Medibot answers from verified clinical books, not random web pages.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.button(
                "🤕 Severe one-sided headache with light sensitivity",
                key="q1",
                on_click=ask,
                args=("Severe throbbing unilateral headache with photophobia",),
            )
        with c2:
            st.button(
                "🔥 How to care for a minor burn at home",
                key="q2",
                on_click=ask,
                args=("Home care and management for minor thermal burn",),
            )
        with c3:
            st.button(
                "🩸 What do high HbA1c levels mean?",
                key="q3",
                on_click=ask,
                args=("Explain high HbA1c levels and recommended diet",),
            )

        st.markdown(
            '<p class="disclaimer">Medibot shares verified reference information and does not replace a doctor. '
            'In an emergency, call your local emergency number (112 in India).</p>',
            unsafe_allow_html=True,
        )

    # Conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🩺"):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Generate the reply for a new question
    if query:
        with st.chat_message("assistant", avatar="🩺"):
            response_box = st.empty()
            full_response = ""
            report_context = st.session_state.user_report_text or ""

            try:
                # Direct in-memory RAG generation
                for token in generate_clinical_response(query=query, report_text=report_context):
                    full_response += token
                    response_box.markdown(full_response + " ▌")

                final_display = full_response + "<br><span class='verified-chip'>✓ Verified Reference Match</span>"
                response_box.markdown(final_display, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": final_display})

            except Exception as e:
                response_box.error(f"Clinical inference error: {e}")


# =====================================================================
#  PAGE 2 — ABOUT
# =====================================================================
def page_about():
    page_header(
        "About Medibot",
        "Bridging complex medical textbooks and everyday patient questions.",
    )
    st.markdown(
        '<div class="card"><p style="font-size:15px;">'
        'Medibot compares your question directly against verified clinical volumes instead of scouring random web articles. '
        'It then explains the potential causes, safe home care, and the warning signs that need urgent medical attention.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            '<div class="card"><div class="step-no">1</div><h3>You ask</h3>'
            '<p>Type your symptoms or question. You can also attach a lab report PDF.</p></div>',
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            '<div class="card"><div class="step-no">2</div><h3>Medibot searches</h3>'
            '<p>It finds the most relevant passages in the verified medical reference library.</p></div>',
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            '<div class="card"><div class="step-no">3</div><h3>You get a clear answer</h3>'
            '<p>The answer streams in, written from those passages in plain language.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Every answer includes</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            '<div class="card"><div class="icon-bubble" style="font-size:22px;">🔍</div><h3>Potential causes</h3>'
            '<p>What the literature says could explain the symptoms.</p></div>',
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            '<div class="card"><div class="icon-bubble" style="font-size:22px;">🏠</div><h3>Home care and next steps</h3>'
            '<p>Safe things you can do now and what to do next.</p></div>',
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            '<div class="card"><div class="icon-bubble" style="font-size:22px;">🚨</div><h3>Red flag signs</h3>'
            '<p>When to seek emergency care immediately.</p></div>',
            unsafe_allow_html=True,
        )

    st.button("Start a consultation", type="primary", on_click=go, args=(PAGES[0],), key="about_cta")


# =====================================================================
#  PAGE 3 — ADMIN PORTAL
# =====================================================================
def page_admin():
    page_header(
        "Admin Portal",
        "Authorized personnel only. Upload official medical textbooks to update the clinical knowledge base.",
    )

    col_form, col_info = st.columns([1.6, 1])
    with col_form:
        with st.container(border=True):
            admin_pass = st.text_input("Admin secret key", type="password", placeholder="Enter your admin key")
            admin_file = st.file_uploader(
                "Clinical textbook (PDF):", type=["pdf"], key="admin_reference_file"
            )

            if st.button("Upload to knowledge base", type="primary", key="admin_upload"):
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

    with col_info:
        st.markdown(
            '<div class="card"><div class="icon-bubble" style="font-size:22px;">📚</div>'
            '<h3>Permanent storage</h3>'
            '<p>Textbooks uploaded here are stored permanently and used as references in all future consultations.</p></div>',
            unsafe_allow_html=True,
        )


# =====================================================================
#  PAGE 4 — CONTACT
# =====================================================================
def page_contact():
    page_header(
        "Get in touch",
        "For clinical inquiries, partnership discussions, or support with the Medibot system.",
    )

    icon_open = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    ico_mail = icon_open + '<path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>'
    ico_pin = icon_open + '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>'
    ico_phone = icon_open + '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>'

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f'<div class="card"><div class="icon-bubble">{ico_mail}</div><h3>Email</h3>'
            '<p><a href="mailto:priyanshuchand7037@gmail.com">priyanshuchand7037@gmail.com</a></p></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="card"><div class="icon-bubble">{ico_pin}</div><h3>Headquarters</h3>'
            '<p>Noida</p></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="card"><div class="icon-bubble">{ico_phone}</div><h3>Phone</h3>'
            '<p><a href="tel:+917037524005">+91 7037524005</a></p></div>',
            unsafe_allow_html=True,
        )

    st.button("Back to chat", type="primary", on_click=go, args=(PAGES[0],), key="contact_cta")


# =====================================================================
#  ROUTER
# =====================================================================
current = st.session_state.nav_page
if current == PAGES[0]:
    page_home()
elif current == PAGES[1]:
    page_about()
elif current == PAGES[2]:
    page_admin()
else:
    page_contact()

footer()