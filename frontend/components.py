import streamlit as st

# Importiamo le tue pipeline per l'automazione!
# (Assicurati che i percorsi e i nomi delle classi corrispondano ai tuoi file reali)
from src.ingestion.pipeline import run_ingestion_pipeline
from src.embedding import run_embedding_pipeline 

def render_sidebar():
    """Disegna la barra laterale e gestisce il caricamento/creazione dei dati."""
    with st.sidebar:
        st.header("⚙️ Configurazione Dati")
        
        # Input utente
        ticker = st.text_input("Ticker Azienda (es. TSLA, AAPL):", value="TSLA").upper()
        year = st.text_input("Anno Fiscale:", value="2025")
        
        st.markdown("---")
        
        # Pulsante di caricamento e Auto-Pipeline
        if st.button("Carica Dati nel RAG", type="primary", use_container_width=True):
            
            # Proviamo prima a caricare i dati se esistono già
            try:
                with st.spinner(f"Cerco i dati locali per {ticker} ({year})..."):
                    # 🟢 FIX ERRORE: Passiamo gli argomenti esplicitamente con il loro nome!
                    st.session_state.rag_app.load_company_data(ticker=ticker, year=year, report_type="10-K")
                    
                st.session_state.company_loaded = True
                st.session_state.messages = [] 
                st.success(f"✅ Analista pronto per {ticker}!")
                
            except Exception as e: # Se i dati non esistono, parte l'automazione!
                st.warning(f"⚠️ Dati non trovati in locale. Avvio auto-pipeline per {ticker}...")
                
                try:
                    # 1. Fase di Ingestion (Download, Parse, Chunk)
                    with st.spinner("📥 Scaricamento da SEC e Chunking in corso... (potrebbe volerci un po')"):
                        # Inserisci qui la chiamata esatta alla tua classe di ingestion
                        ingestion = run_ingestion_pipeline(ticker=ticker, year=year, report_type="10-K")
                        #ingestion.run(ticker=ticker, year=year, report_type="10-K")
                    
                    # 2. Fase di Embedding (Vector Storage FAISS)
                    with st.spinner("🧠 Creazione Indice Vettoriale (FAISS)..."):
                        # Inserisci qui la chiamata esatta alla tua classe di embedding
                        embedding = run_embedding_pipeline(ticker=ticker, year=year, report_type="10-K")
                        #embedding.run(ticker=ticker, year=year, report_type="10-K")
                    
                    # 3. Ricarichiamo il RAG ora che i dati esistono!
                    with st.spinner("🔄 Caricamento nel motore RAG..."):
                        st.session_state.rag_app.load_company_data(ticker=ticker, year=year, report_type="10-K")
                        
                    st.session_state.company_loaded = True
                    st.session_state.messages = []
                    st.success(f"🎉 Pipeline completata! Analista pronto per {ticker}!")
                    
                except Exception as pipeline_error:
                    st.error(f"❌ Errore critico durante la creazione dei dati: {pipeline_error}")