import streamlit as st
import sys
from pathlib import Path

# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.rag.service import RAGService
from frontend.components import render_sidebar

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Edgar Analyst AI", page_icon="📈", layout="wide")
st.title("🤖 Edgar RAG Multi-Analyst")
st.markdown("Query SEC financial statements (Form 10-K) with Mistral and Reranking.")

# --- STATE INITIALIZATION ---
if "rag_app" not in st.session_state:
    with st.spinner("LLM engine initialization (may take several minutes)..."):
        st.session_state.rag_app = RAGService()
        
if "messages" not in st.session_state:
    st.session_state.messages = []

if "company_loaded" not in st.session_state:
    st.session_state.company_loaded = False

# --- INTERFACE RENDERING ---
# Draw the sidebar by calling the component
render_sidebar()

# Chat Management
if not st.session_state.company_loaded:
    st.info("👈 Use the sidebar to select and upload a company's financial statement.")
else:
    # Show history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("❓ Ask a question about the budget..."):
        
        # Show the question on the screen
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate the response
        with st.chat_message("assistant"):
            with st.spinner("🧠 I'm thinking..."):
                # We only pass the last 3 messages as history
                history_for_rag = [
                    (st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"])
                    for i in range(0, len(st.session_state.messages)-1, 2)
                ][-3:] 

                response = st.session_state.rag_app.ask(query=prompt, history=history_for_rag)
                st.markdown(response)
        
        # Save the answer
        st.session_state.messages.append({"role": "assistant", "content": response})