import os
import sys
from src.retrieval.retriever import Retriever
from src.llm.generator import LLMGenerator
from src.utils import config

def main():
    print("🚀 Welcome to the Edgar RAG Multi-Analyst (Local Version)!")
    
    # --- OTTIMIZZAZIONE 1: Carichiamo l'LLM UNA SOLA VOLTA ---
    # Caricare 15GB di modello ogni volta che cambi azienda è un suicidio di tempo/risorse.
    # Lo facciamo all'inizio, fuori dal ciclo di switch.
    print("🧠 Initializing LLM (Mistral-7B) from local storage...")
    try:
        generator = LLMGenerator() # Usa il default LOCAL_LLM_PATH definito nel config
    except Exception as e:
        print(f"❌ Error loading LLM: {e}. Did you run setup_models.py?")
        return

    while True:
        # Ticker selection
        print("\n" + "="*50)
        ticker = input("Which company do you want to analyze? (e.g., TSLA, AMZN, AAPL) or 'exit': ").strip().upper() or "TSLA"
        
        if ticker == 'exit':
            break
            
        report_type = input("What type of report? (e.g., 10-K, 10-Q) [default 10-K]: ").strip().upper() or "10-K"
        
        paths = config.get_paths(ticker, report_type) 
        
        # Check if data files exist
        if not os.path.exists(paths["index"]) or not os.path.exists(paths["chunks"]):
            print(f"❌ Error: Local data not found for {ticker} ({report_type}).")
            print(f"   Please run ingestion and indexing for this ticker first.")
            continue # Torna all'inizio del loop senza crashare

        # --- OTTIMIZZAZIONE 2: Cambio solo i dati, non il cervello ---
        print(f"📂 Loading FAISS index and chunks for {ticker}...")
        retriever = Retriever(paths["index"], paths["chunks"])
        
        print(f"✅ Analyst ready for {ticker}!")
        
        # Chat cycle per il ticker selezionato
        while True:
            query = input(f"\n[{ticker.upper()}] Question (type 'switch' to change company): ")
            
            if query.lower() == 'exit':
                print("Goodbye!")
                sys.exit() # Chiude tutto il programma
            
            if query.lower() == 'switch':
                break # Esce dal ciclo chat e torna alla selezione ticker
                
            print("🔍 Searching context...")
            chunks = retriever.search(query, k=4)
            
            print("✍️ Generating answer...")
            answer = generator.generate_answer(query, chunks)
            
            print(f"\n[Analista]: {answer}\n")
            print("-" * 50)

if __name__ == "__main__":
    main()