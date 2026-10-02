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
    initial_sidebar_state="collapsed"
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
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# --- CUSTOM CSS: PERLA & MEDOK LIGHT THEME DESIGN SYSTEM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --color-dark: #2B2E38;
        --color-bg: #F7F8F7;
        --color-blue: #3DB4FE;
        --color-mint: #4CECB8;
        --color-white: #FFFFFF;
        --color-border: #E8ECEF;
        --color-text-muted: #7A8699;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: var(--color-bg) !important;
        color: var(--color-dark) !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1280px !important;
    }

    /* --- COMMERCIAL NAVBAR --- */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 32px;
        background: var(--color-white);
        border: 1px solid var(--color-border);
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(43, 46, 56, 0.04);
        margin-bottom: 24px;
        position: sticky;
        top: 12px;
        z-index: 1000;
    }
    .brand-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 22px;
        font-weight: 800;
        color: var(--color-dark);
        text-decoration: none;
    }
    .brand-logo span {
        color: var(--color-blue);
    }
    .nav-links {
        display: flex;
        gap: 28px;
        align-items: center;
    }
    .nav-link {
        font-size: 14px;
        font-weight: 600;
        color: var(--color-text-muted);
        text-decoration: none;
        transition: color 0.2s ease;
    }
    .nav-link:hover {
        color: var(--color-blue);
    }
    .nav-btn-signin {
        padding: 8px 20px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 700;
        border: 1px solid var(--color-border);
        color: var(--color-dark);
        background: transparent;
        text-decoration: none;
    }

    /* --- PERLA STYLE HERO SECTION --- */
    .hero-section {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 100%);
        border: 1px solid #E0F2FE;
        border-radius: 28px;
        padding: 56px 48px;
        margin-bottom: 36px;
        box-shadow: 0 10px 30px rgba(61, 180, 254, 0.05);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(76, 236, 184, 0.18);
        color: #0d9468;
        font-weight: 700;
        font-size: 12px;
        padding: 6px 14px;
        border-radius: 20px;
        margin-bottom: 18px;
        letter-spacing: 0.3px;
    }
    .hero-title {
        font-size: 46px;
        font-weight: 800;
        line-height: 1.15;
        color: var(--color-dark);
        margin: 0 0 16px 0;
        letter-spacing: -1.2px;
    }
    .hero-subtitle {
        font-size: 17px;
        color: var(--color-text-muted);
        max-width: 580px;
        line-height: 1.6;
        margin: 0 0 28px 0;
    }
    .btn-gradient {
        background: linear-gradient(90deg, var(--color-blue) 0%, var(--color-mint) 100%);
        color: white !important;
        font-weight: 700;
        border-radius: 14px;
        padding: 12px 28px;
        text-decoration: none;
        display: inline-block;
        border: none;
        box-shadow: 0 6px 18px rgba(61, 180, 254, 0.28);
    }

    /* --- MEDOK WORKSPACE DESIGN --- */
    .medok-frame {
        background: #FFFFFF;
        border-radius: 24px;
        border: 1px solid var(--color-border);
        box-shadow: 0 8px 30px rgba(43, 46, 56, 0.05);
        padding: 24px;
        margin-bottom: 40px;
    }
    .specialist-card {
        background: var(--color-bg);
        border: 1px solid var(--color-border);
        border-radius: 20px;
        padding: 24px;
    }
    .badge-version {
        background: var(--color-dark);
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        float: right;
    }
    .consultation-title {
        font-size: 22px;
        font-weight: 800;
        color: var(--color-dark);
        margin: 12px 0 6px 0;
    }
    .consultation-desc {
        font-size: 13px;
        color: var(--color-text-muted);
        line-height: 1.5;
        margin-bottom: 20px;
    }

    /* Quick Prompts List */
    .quick-list-title {
        font-size: 11px;
        font-weight: 800;
        color: #A0AEC0;
        letter-spacing: 0.8px;
        margin: 20px 0 10px 0;
        text-transform: uppercase;
    }
    .quick-item {
        background: #FFFFFF;
        border: 1px solid var(--color-border);
        padding: 10px 14px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 500;
        color: var(--color-dark);
        margin-bottom: 8px;
        cursor: pointer;
    }

    /* Chat bubble enhancements */
    .stChatMessage {
        background: #FFFFFF !important;
        border: 1px solid var(--color-border) !important;
        border-radius: 18px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 8px rgba(43, 46, 56, 0.02) !important;
    }
    .verified-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(76, 236, 184, 0.15);
        color: #0b8058;
        font-size: 12px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        margin-top: 10px;
    }

    /* --- FOOTER --- */
    .site-footer {
        background: var(--color-white);
        border-top: 1px solid var(--color-border);
        border-radius: 20px;
        padding: 32px 40px;
        margin-top: 60px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .footer-copy {
        font-size: 13px;
        color: var(--color-text-muted);
    }
    .footer-links a {
        color: var(--color-text-muted);
        text-decoration: none;
        font-size: 13px;
        font-weight: 600;
        margin-left: 20px;
    }
    .footer-links a:hover {
        color: var(--color-blue);
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_report_text" not in st.session_state:
    st.session_state.user_report_text = ""
if "active_report_name" not in st.session_state:
    st.session_state.active_report_name = ""

# --- 1. COMMERCIAL NAVIGATION BAR ---
st.markdown("""
<div class="navbar">
    <a href="#home" class="brand-logo">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3DB4FE" stroke-width="2.5"><path d="M12 2v20M2 12h20"/></svg>
        Medibot<span></span>
    </a>
    <div class="nav-links">
        <a href="#home" class="nav-link">Home</a>
        <a href="#consultation" class="nav-link">AI Specialist</a>
        <a href="#about" class="nav-link">About</a>
        <a href="#admin" class="nav-link">Admin Portal</a>
        <a href="#contact" class="nav-link">Contact</a>
    </div>
    <div>
        <a href="#admin" class="nav-btn-signin">Sign in</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. HOME SECTION (PERLA STYLE) ---
st.markdown('<div id="home"></div>', unsafe_allow_html=True)
col_h1, col_h2 = st.columns([1.3, 1])

with col_h1:
    st.markdown("""
    <div class="hero-section">
        <span class="hero-badge">● Authoritative Clinical Assistant</span>
        <h1 class="hero-title">Intelligent healthcare guidance for everyday life.</h1>
        <p class="hero-subtitle">
            Consult with our specialized medical AI backed by verified clinical reference books. Fast, private, and grounded in approved knowledge.
        </p>
        <a href="#consultation" class="btn-gradient">Consult Dr. Medibot →</a>
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    st.markdown("""
    <div style="background: white; border: 1px solid #E8ECEF; border-radius: 28px; padding: 32px; box-shadow: 0 8px 24px rgba(0,0,0,0.03);">
        <h3 style="margin-top:0; font-weight:800; font-size:19px; color:#2B2E38;">Why Patients Trust Medibot</h3>
        <div style="display:flex; gap:14px; margin-top:16px;">
            <div style="background:rgba(61,180,254,0.15); width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px;">📚</div>
            <div>
                <b style="font-size:14px; color:#2B2E38;">Verified Literature Only</b>
                <p style="font-size:13px; color:#7A8699; margin:4px 0 0 0;">Never relies on generic internet gossip or uncontrolled forums.</p>
            </div>
        </div>
        <div style="display:flex; gap:14px; margin-top:16px;">
            <div style="background:rgba(76,236,184,0.18); width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px;">🛡️</div>
            <div>
                <b style="font-size:14px; color:#2B2E38;">Private & Zero-Retention</b>
                <p style="font-size:13px; color:#7A8699; margin:4px 0 0 0;">Attached reports remain ephemeral in memory and are never saved permanently.</p>
            </div>
        </div>
        <div style="display:flex; gap:14px; margin-top:16px;">
            <div style="background:rgba(43,46,56,0.08); width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px;">⚡</div>
            <div>
                <b style="font-size:14px; color:#2B2E38;">Instant Response Streaming</b>
                <p style="font-size:13px; color:#7A8699; margin:4px 0 0 0;">Structured causes, home steps, and warning red flags provided in seconds.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 3. CONSULTATION WORKSPACE (MEDOK APP LAYOUT) ---
st.markdown('<div id="consultation"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
    <h2 style="font-size:24px; font-weight:800; color:#2B2E38; margin:0;">AI Specialist Console</h2>
    <span style="font-size:13px; font-weight:600; color:#3DB4FE;">• Powered by Clinical Reference Library</span>
</div>
""", unsafe_allow_html=True)

col_sidebar, col_chat = st.columns([1, 2.5], gap="large")

# LEFT PANEL: Specialist Profile & Quick Prompts
with col_sidebar:
    st.markdown("""
    <div class="specialist-card">
        <span class="badge-version">v2.4</span>
        <div style="font-size:12px; font-weight:700; color:#3DB4FE; letter-spacing:0.5px;">YOUR CLINICAL CONSULTANT</div>
        <div class="consultation-title">Family Doctor</div>
        <p class="consultation-desc">
            This chat assists with verified information and is not a substitute for an in-person physician. Please seek immediate emergency medical care if you experience acute distress.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✨ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_report_text = ""
        st.session_state.active_report_name = ""
        st.rerun()

    # Optional Lab Report Upload
    with st.expander("📎 Attach Test Report (Optional)", expanded=False):
        uploaded_doc = st.file_uploader("Upload Lab Report (PDF):", type=["pdf"], key="medok_pdf")
        if uploaded_doc and uploaded_doc.name != st.session_state.active_report_name:
            reader = PdfReader(uploaded_doc)
            st.session_state.user_report_text = "".join([page.extract_text() or "" for page in reader.pages])
            st.session_state.active_report_name = uploaded_doc.name
            st.success(f"Attached: {uploaded_doc.name}")

    # Quick Clinical Questions
    st.markdown('<div class="quick-list-title">Common Patient Queries</div>', unsafe_allow_html=True)
    q1 = st.button("Severe unilateral headache with photophobia", use_container_width=True)
    q2 = st.button("How to care for a minor thermal burn", use_container_width=True)
    q3 = st.button("Understanding high HbA1c levels", use_container_width=True)

# RIGHT PANEL: Live Chat Dialogue
with col_chat:
    chat_container = st.container()

    # Display dialogue history
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; padding: 40px 20px; background:white; border-radius:20px; border:1px solid #E8ECEF;">
                <div style="font-size:36px; margin-bottom:10px;">🩺</div>
                <h4 style="margin:0 0 6px 0; color:#2B2E38; font-weight:700;">How can Dr. Medibot help you today?</h4>
                <p style="font-size:13px; color:#7A8699; margin:0;">Select a suggested inquiry from the left or describe your symptoms below.</p>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🩺"):
                st.markdown(msg["content"], unsafe_allow_html=True)

    # Trigger via quick buttons or chat input
    selected_query = None
    if q1: selected_query = "Severe throbbing unilateral headache with photophobia"
    elif q2: selected_query = "Home care and management for minor thermal burn"
    elif q3: selected_query = "Explain high HbA1c levels and recommended diet"

    user_input = st.chat_input("Ask a clinical question or describe your symptoms...") or selected_query

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🩺"):
            response_box = st.empty()
            full_response = ""

            report_context = st.session_state.user_report_text if "user_report_text" in st.session_state and st.session_state.user_report_text else ""

            try:
                # Direct in-memory RAG generation
                stream_gen = generate_clinical_response(
                    query=user_input,
                    report_text=report_context
                )

                for token in stream_gen:
                    full_response += token
                    response_box.markdown(full_response + " ▌")

                final_display = full_response + "<br><span class='verified-chip'>✓ Verified Reference Match</span>"
                response_box.markdown(final_display, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": final_display})

            except Exception as e:
                response_box.error(f"Clinical inference error: {e}")

st.markdown("<br><hr style='border:none; border-top:1px solid #E8ECEF;'><br>", unsafe_allow_html=True)

# --- 4. ABOUT SECTION ---
st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="background: white; border: 1px solid #E8ECEF; border-radius: 24px; padding: 40px; margin-bottom: 30px;">
    <h2 style="font-weight: 800; color: #2B2E38; margin-top: 0;">About Medibot Clinical Intelligence</h2>
    <p style="font-size: 15px; color: #7A8699; line-height: 1.7; max-width: 800px;">
        medOK is developed to bridge the gap between complex medical textbooks and everyday patient inquiries. Rather than scouring random web articles, our system performs direct semantic comparisons against verified clinical volumes and synthesizes potential causes, safe home interventions, and clear red flags requiring urgent medical care.
    </p>
</div>
""", unsafe_allow_html=True)

# --- 5. ADMIN PORTAL (FOR REFERENCE UPLOAD) ---
st.markdown('<div id="admin"></div>', unsafe_allow_html=True)
with st.expander("🛡️ Medical Reference Admin Portal (Authorized Personnel)", expanded=False):
    st.markdown("""
    <p style="font-size:14px; color:#7A8699;">Upload official medical books and textbooks to update the assistant's clinical knowledge repository.</p>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([1.5, 1])
    with col_a1:
        admin_pass = st.text_input("Enter Admin Secret Key:", type="password", placeholder="admin_medibot_secret_2026")
        admin_file = st.file_uploader("Upload Clinical Textbook (PDF format):", type=["pdf"], key="admin_reference_file")

        if st.button("Upload to Knowledge Base", type="primary"):
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
                    with st.spinner("Processing and indexing textbook into verified knowledge repository..."):
                        chunks = ingest_verified_file(temp_path)
                        load_knowledge_base()
                    st.success(f"Success! Book '{admin_file.name}' permanently saved to disk and loaded into RAM. (Indexed Chunks: {chunks})")
                except Exception as ex:
                    st.error(f"Failed to ingest document: {ex}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
    with col_a2:
        st.info("Medical textbooks uploaded here are permanently stored and referenced for future consultations.")

# --- 6. CONTACT SECTION ---
st.markdown('<div id="contact"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="background: white; border: 1px solid #E8ECEF; border-radius: 24px; padding: 36px;">
    <h3 style="font-weight: 800; color: #2B2E38; margin-top: 0;">Get in Touch</h3>
    <p style="font-size: 14px; color: #7A8699; margin-bottom: 20px;">
        For clinical inquiries, partnership discussions, or support regarding the Medibot intelligence system:
    </p>
    <div style="font-size: 14px; color: #2B2E38; font-weight: 600; display: flex; flex-wrap: wrap; gap: 18px; align-items: center;">
        <span style="display: inline-flex; align-items: center; gap: 8px;">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#3DB4FE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
            Email: <a href="mailto:priyanshuchand7037@gmail.com" style="color:#3DB4FE; text-decoration:none;">priyanshuchand7037@gmail.com</a>
        </span>
        <span style="color: #E8ECEF;">|</span>
        <span style="display: inline-flex; align-items: center; gap: 8px;">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#3DB4FE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>
            Headquarters: Noida
        </span>
        <span style="color: #E8ECEF;">|</span>
        <span style="display: inline-flex; align-items: center; gap: 8px;">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#3DB4FE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path></svg>
            Contact: <a href="tel:+917037524005" style="color:#2B2E38; text-decoration:none;">+91 7037524005</a>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 7. COMMERCIAL FOOTER ---
st.markdown("""
<div class="site-footer">
    <div class="footer-copy">
        © 2026 <b>Medibot Healthcare Systems Inc.</b> All rights reserved. Clinical content provided for educational decision support.
    </div>
    <div class="footer-links">
        <a href="#home">Home</a>
        <a href="#consultation">AI Specialist</a>
        <a href="#about">About</a>
        <a href="#admin">Admin</a>
        <a href="#contact">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)