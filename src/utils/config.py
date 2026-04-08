from pathlib import Path

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" 

# Create all the necessary folders
for path in [RAW_DATA_DIR, CHUNKS_DIR, EMBEDDINGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Specific file paths
CHUNKS_JSON = CHUNKS_DIR / "tsla_10k_2025_chunks.json"
FAISS_INDEX = EMBEDDINGS_DIR / "tsla_index.bin"

def get_paths(ticker, report_type="10-K"):
    """Funzione helper per ottenere i percorsi corretti dato un ticker."""
    t = ticker.lower()
    return {
        "index": str(EMBEDDINGS_DIR / f"{t}_{report_type.lower()}_index.bin"),
        "chunks": str(CHUNKS_DIR / f"{t}_{report_type.lower()}_chunks.json")
    }