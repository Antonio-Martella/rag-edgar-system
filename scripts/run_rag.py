import os
import sys
from src.retrieval.retriever import Retriever
from src.llm.generator import LLMGenerator
from src.utils import config

def main():

    print("🚀 Welcome to the Edgar RAG Multi-Analyst (Local Version)!")

    # Uploading the LLM from local storage (already quantized for efficiency)
    print("🧠 Initializing LLM (Mistral-7B) from local storage...")
    try:
        generator = LLMGenerator()
    except Exception as e:
        print(f"❌ Error loading LLM: {e}. Did you run setup_models.py?")
        return

    # Creating a loop to allow users to select different tickers and ask questions without restarting the program
    while True:

        print("\n" + "="*50)

        # Dynamic ticker and report selection, with file control
        ticker = input("Which company do you want to analyze? (e.g., TSLA, AMZN, AAPL) or 'exit': ").strip().upper() or "TSLA"

        # If the user types 'exit', we break the loop and end the program gracefully
        if ticker == 'exit':
            break
        
        # Ask for report type, with a default value of 10-K
        report_type = input("What type of report? (e.g., 10-K, 10-Q) [default 10-K]: ").strip().upper() or "10-K"
        
        # Dynamic file checking for each ticker/report
        paths = config.get_paths(ticker, report_type) 
        
        # Check if data files exist
        if not os.path.exists(paths["index"]) or not os.path.exists(paths["chunks"]):
            print(f"❌ Error: Local data not found for {ticker} ({report_type}).")
            print(f"   Please run ingestion and indexing for this ticker first.")
            continue 

        # If files exist, we proceed to load the retriever and start the chat loop
        print(f"📂 Loading FAISS index and chunks for {ticker}...")
        retriever = Retriever(paths["index"], paths["chunks"])
        
        print(f"✅ Analyst ready for {ticker}!")
        
        # Chat cycle per il ticker selezionato
        while True:
            # User input for questions, with an option to switch ticker or exit
            query = input(f"\n[{ticker.upper()}] Question (type 'switch' to change company): ")

            # If the user types 'exit', we break the loop and end the program gracefully
            if query.lower() == 'exit':
                print("Goodbye!")
                sys.exit() 
            
            # If the user types 'switch', we break the inner loop to allow selecting a different ticker
            if query.lower() == 'switch':
                break 
            
            # Search for relevant chunks in the FAISS index and generate an answer using the LLM
            print("🔍 Searching context...")
            chunks = retriever.search(query, k=5)
            
            # If no relevant chunks are found, we can skip the generation step and inform the user
            print("✍️ Generating answer...")
            answer = generator.generate_answer(query, chunks)
            
            # Display the answer in a clear format
            print(f"\n[Analista]: {answer}\n")
            print("-" * 50)

if __name__ == "__main__":
    main()