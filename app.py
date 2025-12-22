import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import Libraries
from tools_library import (
    fetch_ipo_details, download_pdf_logic, build_vs_logic, 
    get_all_ipo_names, get_concurrent_ipos
)
from brain import execute_brain
from report_engine import generate_deep_dive_report
from comparison_engine import execute_peer_comparison

load_dotenv()

# --- CONFIG & CSS ---
st.set_page_config(page_title="IPO Smooth Operator", page_icon="🚀", layout="wide")
st.markdown("""
<style>
    .stApp { background: linear-gradient(to bottom right, #0e1117, #151922); }
    section[data-testid="stSidebar"] { background-color: #11141d; border-right: 1px solid #2b313e; }
    .stButton>button { background: linear-gradient(45deg, #FF4B4B, #FF914D); color: white; border: none; border-radius: 8px; }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4); }
</style>
""", unsafe_allow_html=True)

# --- STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "active_ipo" not in st.session_state: st.session_state.active_ipo = None
if "active_category" not in st.session_state: st.session_state.active_category = "Mainboard"
if "last_report" not in st.session_state: st.session_state.last_report = ""

@st.cache_data
def load_data(): return get_all_ipo_names()
ipo_data = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Control Panel")
    category = st.radio("Category:", ["Mainboard", "SME"], horizontal=True)
    available = ipo_data.get(category, [])
    if not available: available = ["No IPOs found"]
    
    selected_ipo = st.selectbox("Choose IPO:", available)
    st.divider()
    
    if st.button("🚀 Initialize System", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_ipo = selected_ipo
        st.session_state.active_category = category
        st.session_state.vector_store = None
        st.session_state.last_report = ""
        
        with st.status("Booting Systems...", expanded=True):
            st.write(f"Target: **{selected_ipo}**")
            details = fetch_ipo_details(selected_ipo)
            
            if "id" in details:
                st.write("📥 Fetching RHP...")
                pdf = download_pdf_logic(details)
                if pdf:
                    st.write("🧠 Building Vector Brain...")
                    st.session_state.vector_store = build_vs_logic(pdf)
                    st.success("System Online")
                else:
                    st.warning("⚠️ RHP Missing. Web Fallback Active.")
            else:
                st.error("❌ Critical: ID Not Found.")

# --- MAIN ---
if not st.session_state.active_ipo:
    st.info("👈 Please Select an IPO and Click Initialize.")
    st.stop()

st.subheader(f"🎯 Analysis: {st.session_state.active_ipo}")

tab_chat, tab_report, tab_compare = st.tabs(["💬 Chat", "📑 360° Report", "⚔️ Peer Comparison"])

with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"): st.write(m.content)

    if prompt := st.chat_input("Ask about Sentiment, Peers, GMP..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"):
            status_container = st.status("Thinking...", expanded=True)
            final_ans = ""
            for chunk in execute_brain(prompt, st.session_state.active_ipo, st.session_state.vector_store):
                if "Executing" in chunk or "Synthesizing" in chunk: status_container.write(chunk)
                else: final_ans = chunk
            status_container.update(label="Done", state="complete", expanded=False)
            st.markdown(final_ans)
            st.session_state.messages.append(AIMessage(content=final_ans))

with tab_report:
    if st.button("Generate Report", type="primary"):
        with st.status("Compiling...", expanded=True):
            full_text = ""
            for chunk in generate_deep_dive_report(st.session_state.active_ipo, st.session_state.vector_store):
                if "**Phase" in chunk: st.write(chunk)
                else: full_text = chunk
            st.session_state.last_report = full_text
    if st.session_state.last_report: st.markdown(st.session_state.last_report)

with tab_compare:
    col_filter, _ = st.columns([1, 2])
    with col_filter:
        peer_cat = st.radio("Filter Peers:", ["Mainboard", "SME", "All"], horizontal=True, index=0 if st.session_state.active_category=="Mainboard" else 1)
    
    peers = get_concurrent_ipos(st.session_state.active_ipo, category_filter=peer_cat)
    if not peers: st.info("No active peers found.")
    else:
        selected_peers = st.multiselect("Select Opponents:", peers)
        if st.button("⚔️ Run Comparison", disabled=not selected_peers):
            with st.status("Analyzing...", expanded=True):
                full_analysis = ""
                for chunk in execute_peer_comparison(st.session_state.active_ipo, selected_peers, st.session_state.vector_store):
                    if "Phase" in chunk or "Fetching" in chunk: st.write(chunk)
                    else: full_analysis = chunk
                st.markdown(full_analysis)
