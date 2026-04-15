import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

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

# ------------------------------------
# ------- MODELS CONFIGURATION -------
# ------------------------------------
# Definiamo l'ID del modello embedding, del reranker e del LLM che vogliamo usare.
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Carichiamo il token di Hugging Face dalle variabili d'ambiente
HF_TOKEN = os.getenv("HF_TOKEN")

# Dynamic paths for locally saved models
LOCAL_EMBEDDING_PATH = MODELS_DIR / EMBEDDING_MODEL_ID
LOCAL_RERANKER_PATH = MODELS_DIR / RERANKER_MODEL_ID
LOCAL_LLM_PATH = MODELS_DIR / LLM_MODEL_ID

# Helper function to get file paths for a given ticker and report type
def get_paths(ticker, report_type="10-K", date=None):
    """
    Helper per ottenere i percorsi dei file dati per un ticker specifico.
    """
    t = ticker.lower()
    return {
        "index": str(EMBEDDINGS_DIR / t / f"{report_type.lower()}_{date}_index.bin"),
        "chunks": str(CHUNKS_DIR / t / f"{report_type.lower()}_{date}_chunks.json")
    }