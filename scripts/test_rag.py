import os
from src.retrieval.retriever import Retriever
from src.llm.generator import LLMGenerator
from src.utils import config

def main():
    print("🚀 Welcome to the Edgar RAG Multi-Analyst!")
    
    # Ticker selection
    ticker = input("Which company do you want to analyze? (e.g., TSLA, AMZN, AAPL): ").strip().lower() or "tsla"
    report_type = input("What type of report? (e.g., 10-K, 10-Q): ").strip().upper() or "10-K"
    
    
    paths = config.get_paths(ticker, report_type) 
    
    # Check if files exist before loading the LLM (which is slow)
    if not os.path.exists(paths["index"]) or not os.path.exists(paths["chunks"]):
        print(f"❌ Error: Data not found for {ticker}. Please run the ingestion and indexing scripts first!")
        return

    # Engine initialization
    print(f"📂 Loading data for {ticker}...")
    retriever = Retriever(paths["index"], paths["chunks"])
    
    print("🧠 LLM (Mistral) initialization...")
    generator = LLMGenerator()

    # Chat cycle
    print(f"\n--- Analyst ready for {ticker}! (type 'switch' to change or 'exit' to quit) ---")
    
    while True:
        query = input(f"\n[{ticker}] Question: ")
        
        if query.lower() == 'exit':
            break
        if query.lower() == 'switch':
            return main() # Ricorsione semplice per cambiare azienda
            
        chunks = retriever.search(query, k=4)
        answer = generator.generate_answer(query, chunks)
        
        print(f"\n[Analista]: {answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()