import json
from pathlib import Path

from src.utils import config
from src.ingestion.downloader import EdgarDownloader
# Importiamo il nuovo parser e il nuovo chunker
from src.ingestion.parser import clean_sec_text
from src.ingestion.chunking import create_smart_chunks

def run_ingestion(ticker="TSLA", report_type="10-K", data_after="2024-01-01"):
    """
    Organizza il download, il parsing strutturato e il chunking intelligente dei documenti SEC.
    """

    # Input utente con default
    ticker = input(f"Enter ticker (default {ticker}): ").upper() or ticker
    report_type = input(f"Enter report type (default {report_type}): ").upper() or report_type
    data_after = input(f"Enter date after (default {data_after}): ") or data_after

    # 1. Download del documento
    print(60*"=")
    print(f"🚀 Starting ingestion process for: {ticker}")
    downloader = EdgarDownloader() 
    downloader.fetch_10k(format=report_type, ticker=ticker, limit=1, date_after=data_after)

    # Definizione del path del file scaricato
    base_path = Path(config.RAW_DATA_DIR) / "sec-edgar-filings" / ticker / report_type 
    
    try:
        submission_dir = next(base_path.iterdir())
        file_path = submission_dir / "full-submission.txt"
    except (StopIteration, FileNotFoundError):
        print(f"❌ Error: I couldn't find a file for {ticker} in {base_path}")
        return

    # 2. Lettura e Parsing Strutturato
    print(f"🧹 Parsing and structuring tables: {file_path.name}...")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    # Utilizziamo la funzione ottimizzata per mantenere le tabelle leggibili
    text_refined = clean_sec_text(raw_content)

    # 3. Smart Chunking
    print(f"✂️ Creating smart text chunks with metadata...")
    # Utilizziamo la funzione dedicata in src/ingestion/chunking.py
    chunks = create_smart_chunks(text_refined, ticker, report_type)

    # 4. Salvataggio dei risultati
    output_filename = f"{ticker.lower()}_{report_type.lower()}_chunks.json"
    output_path = Path(config.CHUNKS_DIR) / output_filename
    
    # Prepariamo l'oggetto finale includendo i metadati per ogni chunk
    data_to_save = {
        "ticker": ticker,
        "report_type": report_type,
        "total_chunks": len(chunks),
        "chunks": chunks  # Ogni elemento qui è ora un dict con 'content' e 'metadata'
    }

    # Salvataggio su file JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    print(60*"=")
    print(f"✅ Ingestion completed!")
    print(f"📦 Total chunks created: {len(chunks)}")
    print(f"📂 Saved to: {output_path}")

if __name__ == "__main__":
    run_ingestion()