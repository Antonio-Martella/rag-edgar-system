import os
import sys
from src.retrieval.retriever import Retriever
from src.llm.generator import LLMGenerator
from src.utils import config

def main():
    print("🚀 Welcome to the Edgar RAG Multi-Analyst (Local Version)!")

    print("🧠 Initializing LLM (Mistral-7B) from local storage...")
    try:
        generator = LLMGenerator()
    except Exception as e:
        print(f"❌ Error loading LLM: {e}.")
        return

    while True:
        print("\n" + "="*50)
        ticker = input("Which company do you want to analyze? (e.g., TSLA, AMZN, AAPL) or 'exit': ").strip().upper() or "TSLA"

        if ticker == 'EXIT':
            break
        
        report_type = input("What type of report? [default 10-K]: ").strip().upper() or "10-K"
        paths = config.get_paths(ticker, report_type) 
        
        if not os.path.exists(paths["index"]) or not os.path.exists(paths["chunks"]):
            print(f"❌ Error: Local data not found for {ticker}.")
            continue 

        print(f"📂 Loading FAISS index for {ticker}...")
        retriever = Retriever(paths["index"], paths["chunks"])
        
        # Let's remember the conversation (let's choose the last 3 questions and answers to avoid saturating the prompt)
        chat_history = [] 
        MAX_HISTORY = 3         # memory length
        
        print(f"✅ Analyst ready for {ticker}!")
        
        # Main interaction loop for the selected company
        while True:

            # Ask the user for a question about the company
            query = input(f"\n[{ticker}] Question (type 'switch' to change company): ")

            # If the user types 'exit', we exit the entire program. If they type 'switch', we break out of this loop and go back to selecting a company, which will reset the chat history.
            if query.lower() == 'exit':
                sys.exit() 
            if query.lower() == 'switch':
                break                       # By leaving here, chat_history will be reset to the next ticker
            
            # Search for relevant context chunks using the retriever
            print("🔍 Searching context...")
            chunks = retriever.search(query, k=20) # Retrieve the top 10 most relevant chunks from the index based on the user's query
            
            # Generate the answer using the LLM, passing the retrieved context and the conversation history
            print("✍️ Generating answer...")
            answer = generator.generate_answer(query, chunks, history=chat_history)
            
            print(f"\n[Analista]: {answer}\n")
            print("-" * 50)

            # Update the conversation history with the new question and answer, and ensure we only keep the last MAX_HISTORY interactions to avoid saturating the prompt
            chat_history.append((query, answer))
            if len(chat_history) > MAX_HISTORY:
                chat_history.pop(0) # Remove the oldest interaction if we exceed the maximum history length

if __name__ == "__main__":
    main()