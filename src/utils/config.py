from pathlib import Path

# Radice del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Cartelle Dati
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" 

# Crea tutte le cartelle necessarie
for path in [RAW_DATA_DIR, CHUNKS_DIR, EMBEDDINGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Configurazione Modelli 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Percorsi file specifici 
CHUNKS_JSON = CHUNKS_DIR / "tsla_10k_2025_chunks.json"
FAISS_INDEX = EMBEDDINGS_DIR / "tsla_index.bin"