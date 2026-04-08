import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import config
from src.ingestion.downloader import EdgarDownloader
from src.ingestion.parser import clean_sec_text, refined_clean

def run_ingestion(ticker="TSLA", report_type="10-K", data_after="2024-01-01"):
    """
    Organizes the downloading, cleaning, and splitting of SEC documents.
    """

    # Allows you to override parameters via input prompts, but defaults to the function arguments if left blank
    ticker = input("Enter ticker (e.g. TSLA): ").upper() or ticker
    report_type = input("Enter report type (e.g. 10-K): ") or report_type
    data_after = input("Enter date after (e.g. 2024-01-01): ") or data_after

    # Download the document
    print(60*f"=")
    print(f"🚀 Starting ingestion process for: {ticker}")
    downloader = EdgarDownloader() # Automatically gets the SEC_USER_AGENT from the .env
    downloader.fetch_10k(format=report_type, ticker=ticker, limit=1, date_after=data_after)

    # Let's build the path based on the structure created by sec-edgar-downloader
    base_path = Path(config.RAW_DATA_DIR) / "sec-edgar-filings" / ticker / report_type 
    
    # Find the full-submission.txt file (there is usually only one folder for the most recent filing)
    try:
        submission_dir = next(base_path.iterdir())
        file_path = submission_dir / "full-submission.txt"
    except (StopIteration, FileNotFoundError):
        print(f"❌ Error: I couldn't find a file for {ticker} in {base_path}")
        return

    # Reading and Cleaning with your parser functions
    print(f"🧹 Cleaning file: {file_path.name}...")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    text_cleaned = clean_sec_text(raw_content)
    text_refined = refined_clean(text_cleaned)

    # Chunking the text
    print(f"✂️ Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # Maximum size of each chunk
        chunk_overlap=200,        # Overlap between chunks to maintain context
        length_function=len,      # Function to calculate the length of the text (in this case, number of characters)
        is_separator_regex=False, # We don't use a regex separator, but only the chunk size
    )
    chunks = text_splitter.split_text(text_refined)

    # Let's create a clean JSON filename, e.g.: tsla_10k_chunks.json
    output_filename = f"{ticker.lower()}_{report_type.lower()}_chunks.json"
    output_path = Path(config.CHUNKS_DIR) / output_filename
    
    # Let's save the chunks to a JSON file with some metadata
    data_to_save = {
        "ticker": ticker,            # Ticker information
        "report_type": report_type,  # Report type (e.g. 10-K)
        "total_chunks": len(chunks), # Total number of chunks created
        "chunks": chunks             # List of text chunks
    }

    # Saving the JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    print(f"✅ Ingestion completed! {len(chunks)} chunks saved to: {output_path}")

if __name__ == "__main__":
    # You can test it with other tickers, e.g.: run_ingestion("AAPL")
    run_ingestion()