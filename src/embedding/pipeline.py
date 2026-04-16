import json
from pathlib import Path

from src.utils import config
from src.embedding import Embedder, VectorStorage

def run_embedding_pipeline(ticker: str, year: str, report_type: str = "10-K") -> Path:
    """
    This function runs the entire embedding pipeline for a given ticker, year, and report type.
    """
    print(f"\n🧠 Starting embedding pipeline for {ticker} format {report_type} ({year})...")
    # Generate dynamic paths for chunk and index files based on ticker, report type and year
    paths = config.get_paths(ticker, report_type, year)
    chunks_path = Path(paths["chunks"])
    index_path = Path(paths["index"])
    # 
    if not chunks_path.exists():
        raise FileNotFoundError(f"❌ Error: File {chunks_path} not found. Run run_ingestion.py first!")

    # Loading the text chunks
    print(f"📄 Loading chunks from: {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        chunks = data.get('chunks', [])

    if not chunks:
        raise ValueError("❌ No chunks found in the JSON file!")

    # Since the Embedder expects a list of texts, we extract the 'content' from the dictionaries
    if isinstance(chunks[0], dict) and "content" in chunks[0]:
        texts = [c["content"] for c in chunks]
    else:
        texts = chunks

    # Generating Embeddings Using Your Embedder Class
    embedder = Embedder()
    print(f"🪄 Generating vector embeddings for {len(texts)} chunks...")
    # Using the encode method of your Embedder class
    embeddings = embedder.encode(texts) 

    # Creating and populating the FAISS index using VectorStorage
    print("🗄️ Entering vectors into the FAISS database...")
    storage = VectorStorage(dimension=embeddings.shape[1])
    storage.add_embeddings(embeddings)

    # Saving the index
    storage.save(str(index_path))
    print(f"💾 FAISS index saved successfully to: {index_path}")

    return index_path