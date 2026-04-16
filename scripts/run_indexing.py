import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.embedding import run_embedding_pipeline

def main():
    print("🚀 =======================================")
    print("🚀  FAISS INDEXING PIPELINE")
    print("🚀 =======================================\n")
    # Prompt the user for ticker, year, and report type with defaults
    ticker = input("📈 Enter the Ticker (e.g. TSLA):").strip().upper() or "TSLA"
    year = input("📅 Enter the fiscal year (e.g. 2025): ").strip() or "2025"
    report_type = input("📄 Enter the report type (e.g. 10-K): ").strip().upper() or "10-K"
    # Run the embedding pipeline and handle any exceptions that may occur
    try:
        run_embedding_pipeline(ticker, year, report_type)
        print("\n🎉 INDEXING COMPLETED! The vector database is ready.")
    except Exception as e:
        print(f"\n❌ ERROR DURING INDEXING: {e}")

if __name__ == "__main__":
    main()