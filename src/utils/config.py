from pathlib import Path

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" 

# Local Models Folder
MODELS_DIR = PROJECT_ROOT / "models"

# Create all the necessary folders
for path in [RAW_DATA_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Model Configuration
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Dynamic paths for locally saved models
LOCAL_EMBEDDING_PATH = MODELS_DIR / EMBEDDING_MODEL_ID
LOCAL_RERANKER_PATH = MODELS_DIR / RERANKER_MODEL_ID
LOCAL_LLM_PATH = MODELS_DIR / LLM_MODEL_ID

# Percorsi file specifici (Default per Tesla)
#CHUNKS_JSON = CHUNKS_DIR / "tsla_10k_2025_chunks.json"
#FAISS_INDEX = EMBEDDINGS_DIR / "tsla_index.bin"

def get_paths(ticker, report_type="10-K"):
    """
    Helper per ottenere i percorsi dei file dati per un ticker specifico.
    """
    t = ticker.lower()
    return {
        "index": str(EMBEDDINGS_DIR / f"{t}_{report_type.lower()}_index.bin"),
        "chunks": str(CHUNKS_DIR / f"{t}_{report_type.lower()}_chunks.json")
    }