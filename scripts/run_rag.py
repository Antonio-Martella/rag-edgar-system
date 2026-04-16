import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.rag import RAGService

def main():
    print("🚀 =======================================")
    print("🚀  EDGAR RAG MULTI-ANALYST")
    print("🚀 =======================================\n")

    try:
        rag_app = RAGService()
    except Exception as e:
        print(f"❌ LLM loading error: {e}")
        return

    while True:
        print("\n" + "="*50)

        ticker = input("🏢 Which company do you want to analyze? (e.g., TSLA, AAPL) or 'exit':").strip().upper() or "TSLA"
        if ticker == 'EXIT': break

        ticket_format = input("📄 Report Type? [default 10-K]:").strip().upper() or "10-K"
        
        year = input("📅 Fiscal Year (e.g. 2025):").strip() or "2025"
        
        try:
            rag_app.load_company_data(ticker, year, ticket_format)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        chat_history = []
        MAX_HISTORY = 5

        while True:
            query = input(f"\n[{ticker}] Question (or type 'switch' to change companies):").strip()
            
            if query.lower() == 'exit': sys.exit()
            if query.lower() == 'switch': break
            if not query: continue

            print("⏳ Analysis in progress (Semantic Search + Generation)...")
            
            answer = rag_app.ask(query, history=chat_history)
            
            print(f"\n💡 [Analyst Answer]:\n{answer}\n")
            print("-" * 50)

            chat_history.append((query, answer))
            if len(chat_history) > MAX_HISTORY:
                chat_history.pop(0)

if __name__ == "__main__":
    main()