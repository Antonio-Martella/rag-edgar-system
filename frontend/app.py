import streamlit as st
import sys
from pathlib import Path

# --- CONFIGURAZIONE PATH CRITICA ---
# Siccome app.py è in /frontend, il root del progetto è due livelli sopra
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Ora possiamo importare i nostri moduli senza errori
from src.rag.service import RAGService
from frontend.components import render_sidebar

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Edgar Analyst AI", page_icon="📈", layout="wide")
st.title("🤖 Edgar RAG Multi-Analyst")
st.markdown("Interroga i bilanci SEC (Form 10-K) con Llama/Mistral e Reranking.")

# --- INIZIALIZZAZIONE STATO ---
if "rag_app" not in st.session_state:
    with st.spinner("Inizializzazione del motore LLM (può richiedere un minuto)..."):
        st.session_state.rag_app = RAGService()
        
if "messages" not in st.session_state:
    st.session_state.messages = []

if "company_loaded" not in st.session_state:
    st.session_state.company_loaded = False

# --- RENDER DELL'INTERFACCIA ---
# 1. Disegna la sidebar chiamando il componente
render_sidebar()

# 2. Gestione della Chat
if not st.session_state.company_loaded:
    st.info("👈 Usa la barra laterale per selezionare e caricare il bilancio di un'azienda.")
else:
    # Mostra la cronologia
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input della chat
    if prompt := st.chat_input("Fai una domanda sul bilancio..."):
        
        # Mostra la domanda a schermo
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Genera la risposta
        with st.chat_message("assistant"):
            with st.spinner("Ricerca vettoriale, Reranking e generazione in corso..."):
                # Passiamo solo gli ultimi 3 messaggi come history
                history_for_rag = [
                    (st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"])
                    for i in range(0, len(st.session_state.messages)-1, 2)
                ][-3:] 

                response = st.session_state.rag_app.ask(query=prompt, history=history_for_rag)
                st.markdown(response)
        
        # Salva la risposta
        st.session_state.messages.append({"role": "assistant", "content": response})