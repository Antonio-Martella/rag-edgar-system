import sys
from pathlib import Path

# Ensure the project root is in the system path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.embedding import setup_embedding_model
from src.retrieval import setup_reranker_model
from src.llm import setup_llm
from src.utils import config

def main():
    print("🚀 ===================================================")
    print("🚀  SETUP: DOWNLOAD MODELS (EMBEDDING, RERANKING, LLM)")
    print("🚀 ===================================================\n")

    try:
        # Embedding Model 
        setup_embedding_model()
        # Reranker Model
        setup_reranker_model()
        # LLM Model
        setup_llm()
        print("Summary of downloaded templates: \n")
        print(f"✅ Embedding Model: {config.EMBEDDING_MODEL_ID} -> {config.LOCAL_EMBEDDING_PATH}"
              f"\n✅ Reranker Model: {config.RERANKER_MODEL_ID} -> {config.LOCAL_RERANKER_PATH}"
              f"\n✅ LLM Model: {config.LLM_MODEL_ID} -> {config.LOCAL_LLM_PATH}")

        print("\n🎉 SETUP COMPLETE! All templates are ready to use.")
    
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR DURING SETUP: {e}")

if __name__ == "__main__":
    main()