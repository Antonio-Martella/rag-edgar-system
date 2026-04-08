import json
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import config
from src.ingestion.downloader import EdgarDownloader
from src.ingestion.parser import clean_sec_text, refined_clean

def run_ingestion(ticker="TSLA", report_type="10-K"):
    """
    Orchestra il download, la pulizia e lo splitting dei documenti SEC.
    """
    # Permette di inserire il ticker e il tipo di report da terminale
    ticker = input("Inserisci il ticker (es: TSLA): ") or ticker
    report_type = input("Inserisci il tipo di report (es: 10-K): ") or report_type

    # Download del documento
    print(f"🚀 Avvio processo di Ingestion per: {ticker}")
    downloader = EdgarDownloader()      # Prende automaticamente il SEC_USER_AGENT dal .env
    downloader.fetch_10k(ticker, limit=1, date_after="2024-01-01")

    # Costruiamo il percorso basandoci sulla struttura creata da sec-edgar-downloader
    base_path = Path(config.RAW_DATA_DIR) / "sec-edgar-filings" / ticker / report_type
    
    # Trova il file full-submission.txt (di solito c'è una sola cartella per il filing più recente)
    try:
        submission_dir = next(base_path.iterdir())
        file_path = submission_dir / "full-submission.txt"
    except (StopIteration, FileNotFoundError):
        print(f"❌ Errore: Non ho trovato file per {ticker} in {base_path}")
        return

    # Lettura e Pulizia con le tue funzioni del parser
    print(f"🧹 Pulizia del file: {file_path.name}...")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    text_cleaned = clean_sec_text(raw_content)
    text_refined = refined_clean(text_cleaned)

    # Chunking (Suddivisione in pezzi)
    print(f"✂️ Creazione dei chunk di testo...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # Dimensione massima di ogni chunk
        chunk_overlap=200,        # Sovrapposizione tra i chunk per mantenere il contesto
        length_function=len,      # Funzione per calcolare la lunghezza del testo (in questo caso, numero di caratteri)
        is_separator_regex=False, # Non usiamo un separatore regex, ma solo la dimensione dei chunk
    )
    chunks = text_splitter.split_text(text_refined)

    # Creiamo un nome file JSON pulito, es: tsla_10k_chunks.json
    output_filename = f"{ticker.lower()}_{report_type.lower()}_chunks.json"
    output_path = Path(config.CHUNKS_DIR) / output_filename
    
    # Salviamo i chunk in un file JSON con un po' di metadati
    data_to_save = {
        "ticker": ticker,            # Informazione sul ticker
        "report_type": report_type,  # Tipo di report (es: 10-K)
        "total_chunks": len(chunks), # Numero totale di chunk creati
        "chunks": chunks             # Lista dei chunk di testo
    }

    # Salvataggio del file JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    print(f"✅ Ingestion completata! {len(chunks)} chunk salvati in: {output_path}")

if __name__ == "__main__":
    # Puoi testarlo anche con altri ticker, es: run_ingestion("AAPL")
    run_ingestion()