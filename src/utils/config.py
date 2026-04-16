import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Load the Hugging Face token from the environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ------------------------------------
# ------- MODELS CONFIGURATION -------
# ------------------------------------
# Let's define the ID of the embedding model, the reranker and the LLM we want to use.
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# ------------------------------------

# Local Models Folder
MODELS_DIR = PROJECT_ROOT / "models"

# Dynamic paths for locally saved models
LOCAL_EMBEDDING_PATH = MODELS_DIR / EMBEDDING_MODEL_ID
LOCAL_RERANKER_PATH = MODELS_DIR / RERANKER_MODEL_ID
LOCAL_LLM_PATH = MODELS_DIR / LLM_MODEL_ID

# Data Folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" 

# Create all the necessary folders
for path in [RAW_DATA_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Helper function to get file paths for a given ticker and report type
def get_paths(ticker: str, report_type: str, date: str) -> dict:
    """
    Helper to get the paths to data files for a specific ticker.
    """
    t = ticker.lower()
    return {
        "index": str(EMBEDDINGS_DIR / t / f"{report_type.lower()}_{date}_index.bin"),
        "chunks": str(CHUNKS_DIR / t / f"{report_type.lower()}_{date}_chunks.json")
    }