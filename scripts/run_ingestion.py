import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ingestion import run_ingestion_pipeline

def main() -> None:
    print("🚀 =======================================")
    print("🚀  TEST: SEC RAG INGESTION PIPELINE")
    print("🚀 =======================================\n")
    
    # We prompt the user to input the ticker, report type, and fiscal year for the SEC filing they want to ingest.
    ticker = input("📈 Enter the Ticker (e.g. TSLA): ").strip().upper()
    if not ticker: ticker = "TSLA"

    # We allow the user to specify the report type (e.g., 10-K, 10-Q), but we default to "10-K" if they don't provide one.
    report_type = input("📄 Enter the report type (e.g. 10-K): ").strip().upper()
    if not report_type: report_type = "10-K"

    # We prompt the user to enter the fiscal year for the report, defaulting to "2023" if they don't provide one.
    year = input("📅 Enter the fiscal year (e.g. 2023): ").strip()
    if not year: year = "2023"
    
    # We run the ingestion pipeline with the provided inputs and handle any exceptions that may occur during the process, 
    # printing an error message if something goes wrong.
    try:
        run_ingestion_pipeline(ticker.upper(), year, report_type.upper())
        print("\n🎉 TEST PASSED! All modules are communicating correctly.")
    except Exception as e:
        print(f"\n❌ ERROR DURING TEST: {e}")

if __name__ == "__main__":
    main()