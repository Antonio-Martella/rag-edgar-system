import json
from pathlib import Path
from src.ingestion import EdgarDownloader, extract_pure, clean_sec_text, create_chunks
from src.utils import config


def run_ingestion_pipeline(ticker: str, year: str, report_type: str = "10-K") -> Path:
    """ 
    Orchestrates the complete ingestion flow for SEC filings. 

    This function integrates the downloader, parser, and chunker modules to 
    transform a raw SEC submission into a structured JSON file. It handles: 
    1. Downloading the 'full-submission.txt' based on the fiscal year. 
    2. Extracting the primary 10-K HTML document. 
    3. Cleaning text and linearizing complex financial tables. 
    4. Generating smart chunks with contextual metadata. 
    5. Saving the final dataset to the path specified in the configuration. 

    Args: 
        ticker (str): The stock symbol of the company (e.g., 'TSLA'). 
        year (str): The fiscal year of the report. 
        report_type (str): The type of SEC form (default: '10-K'). 

    Returns: 
        Path: The file system path to the generated JSON chunks. 

    Raises: 
        ValueError: If the SEC document cannot be retrieved or found.
    """
    print(f"⚙️ Initializing pipeline for {ticker} - Fiscal Year {year}...")
    # Download the raw HTML content of the specified SEC filing using our EdgarDownloader
    downloader = EdgarDownloader()
    raw_html = downloader.fetch_and_read(
        ticker=ticker, 
        target_year=year, 
        report_type=report_type
    )
    # We check if we successfully retrieved the raw HTML content
    if not raw_html:
        raise ValueError(f"❌ Critical failure: Unable to retrieve data for {ticker} {report_type} ({year}).")
    
    print(f"🛡️ Extracting the pure {report_type} document...")
    # Extract the pure HTML content from the raw submission
    pure_html = extract_pure(raw_submission = raw_html, report_type = report_type)

    print("🧹 Text parsing and table linearization in progress...")
    # We clean the SEC text by removing non-textual tags, linearizing tables, and performing additional cleaning to prepare it for chunking.
    clean_text = clean_sec_text(pure_html)

    # Creating smart chunks with contextual metadata to enhance RAG performance.
    print("✂️ Creating smart chunks...")
    chunks = create_chunks(clean_text, ticker, report_type, year)
    print(f"✅ Generated {len(chunks)} chunks!")

    # We assemble the final JSON with metadata and chunks, and save it to a file in the designated output folder.
    final_json_data = {
        "ticker": ticker,
        "report_type": report_type,
        "fiscal_year": year,
        "total_chunks": len(chunks),
        "chunks": chunks
    }
    
    # We determine the output path for the JSON file using our config utility, ensuring it follows our organized folder structure.
    paths = config.get_paths(ticker, report_type, year)
    # We create the output directory if it doesn't exist, and we save the final JSON data to a file named "chunks.json" within that directory.
    output_path = Path(paths["chunks"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # We save the final JSON data to the output path with proper formatting for readability.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_json_data, f, ensure_ascii=False, indent=4)
        
    print(f"💾 File JSON saved successfully in: {output_path}")
    return output_path