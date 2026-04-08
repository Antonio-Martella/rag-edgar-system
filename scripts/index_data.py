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
    
    # Loading chunks saved by ingestion
    # We build the filename based on the ticker, e.g.: tsla_10k_chunks.json
    chunks_filename = f"{ticker.lower()}_{report_type.lower()}_chunks.json"
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
    index_filename = f"{ticker.lower()}_{report_type.lower()}_index.bin"
    index_path = Path(config.EMBEDDINGS_DIR) / index_filename
    storage.save(str(index_path))

    # Quick search test (optional, but useful for verification)
    print("\n🔍 Running quick search test...")
    query = "What are the main risks mentioned regarding cyberattacks?"
    query_vector = embedder.encode([query])
    distances, indices = storage.index.search(query_vector, 3) # top 3
    
    print("\n"+60*"=")
    print(f"Top result found in chunk index: \033[34m{indices[0][0]} (Distance: {distances[0][0]:.4f})\033[0m")
    print(f"Query: \033[31m{query}\033[0m")
    print(f"Answer: \033[92m{chunks[indices[0][0]]}\033[0m") 
    print(f"✅ Indexing completed successfully!")
    print(60*f"=")

if __name__ == "__main__":
    run_indexing()