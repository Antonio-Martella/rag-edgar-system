import streamlit as st

from src.ingestion import run_ingestion_pipeline
from src.embedding import run_embedding_pipeline 

def render_sidebar():
    """
    Designs the sidebar and manages data loading/creation.
    """
    with st.sidebar:
        st.header("⚙️ Data Configuration for 10-K reports")
        
        # User input
        ticker = st.text_input("Company Ticker (e.g. TSLA, AAPL):", value="TSLA").upper()
        year = st.text_input("Fiscal year:", value="2025")
        
        st.markdown("---")
        
        # Load Button and Auto-Pipeline
        if st.button("Load Data into RAG", type="primary", use_container_width=True):
            
            # Let's try to load the data first if it already exists
            try:
                with st.spinner(f"Looking for local data for {ticker} ({year})..."):
                    # Let's pass arguments explicitly by name!
                    st.session_state.rag_app.load_company_data(ticker=ticker, year=year, report_type="10-K")
                    
                st.session_state.company_loaded = True
                st.session_state.messages = [] 
                st.success(f"✅ Analyst ready for {ticker}!")

            # If the data doesn't exist, automation starts!
            except Exception as e: 
                st.warning(f"⚠️ Data not found locally. Starting auto-pipeline for {ticker}...")
                
                try:
                    # Ingestion Phase (Download, Parse, Chunk)
                    with st.spinner("📥 Downloading from SEC and Chunking... (may take a while)"):
                        # Enter the exact call to your ingestion class here
                        ingestion = run_ingestion_pipeline(ticker=ticker, year=year, report_type="10-K")
                    
                    # Embedding Phase (Vector Storage FAISS)
                    with st.spinner("🧠 Vector Index Creation (FAISS)..."):
                        # Enter the exact call to your embedding class here
                        embedding = run_embedding_pipeline(ticker=ticker, year=year, report_type="10-K")
                    
                    # Let's reload the RAG now that the data exists!
                    with st.spinner("🔄 Loading into the RAG engine..."):
                        st.session_state.rag_app.load_company_data(ticker=ticker, year=year, report_type="10-K")
                        
                    st.session_state.company_loaded = True
                    st.session_state.messages = []
                    st.success(f"🎉 Pipeline complete! Analyst ready for {ticker}!")
                    
                except Exception as pipeline_error:
                    st.error(f"❌ Critical error while creating data: {pipeline_error}")