import json
from pathlib import Path

from src.utils import config
from src.embedding.embedder import Embedder
from src.embedding.storage import VectorStorage

def run_indexing(ticker="TSLA", report_type="10-K"):
    """
    Loads the text chunks, generates the embeddings, and saves the FAISS index.
    """

    ticker = input("Enter ticker (e.g. TSLA): ").upper() or ticker
    report_type = input("Enter report type (e.g. 10-K): ") or report_type
    date = input("Enter report date (e.g. 2023): ") or "2023"

    # Loading chunks saved by ingestion
    # We build the filename based on the ticker, e.g.: tsla_10k_chunks.json
    chunks_filename = f"{ticker.lower()}_{report_type.lower()}_chunks_{date}.json"
    print(chunks_filename)
    chunks_path = Path(config.CHUNKS_DIR) / chunks_filename
    
    if not chunks_path.exists():
        print(f"❌ Error: File {chunks_path} not found. Run ingestion first!")
        return

    print(f"📄 Loading chunks from: {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        chunks = data['chunks']

    # Generating Embeddings Using Your Embedder Class
    # Note: Embedder already uses config.EMBEDDING_MODEL_ID internally
    embedder = Embedder()
    print(f"🧠 Generating embeddings for {len(chunks)} text chunks...")
    embeddings = embedder.encode(chunks)

    # Creating and populating the FAISS index using VectorStorage
    storage = VectorStorage(dimension=embeddings.shape[1])
    storage.add_embeddings(embeddings)

    # Saving the index
    # We build the index filename, e.g.: tsla_10k_index.bin
    index_filename = f"{ticker.lower()}_{report_type.lower()}_{date}_index.bin"
    index_path = Path(config.EMBEDDINGS_DIR) / index_filename
    storage.save(str(index_path))
    print(f"💾 FAISS index saved to: {index_path}")

if __name__ == "__main__":
    run_indexing()