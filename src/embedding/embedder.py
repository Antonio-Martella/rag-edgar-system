from sentence_transformers import SentenceTransformer
from src.utils import config

def setup_embedding_model() -> None:
    """
    Download and save the embedding model locally if it doesn't exist.
    """
    # Check if the embedding model is already downloaded and saved locally, if not, download and save it.
    if not config.LOCAL_EMBEDDING_PATH.exists():
        print(f"📥 Downloading Embedding: {config.EMBEDDING_MODEL_ID}...")
        model = SentenceTransformer(config.EMBEDDING_MODEL_ID, trust_remote_code=True)
        model.save(str(config.LOCAL_EMBEDDING_PATH))
        print(f"✅ Embedding saved in {config.LOCAL_EMBEDDING_PATH}")
    else:
        print("✔️ Embedding already present locally.")


class Embedder:
    def __init__(self, model_path=str(config.LOCAL_EMBEDDING_PATH)):

        print(f"Loading LOCAL Embedding Model: {model_path} ---")
        # Check that the model exists locally before loading it (avoids slow errors)
        if not config.LOCAL_EMBEDDING_PATH.exists():
             raise FileNotFoundError(f"❌ Model not found in {model_path}. Run setup_models.py!")
             
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, text_list):
        return self.model.encode(text_list, show_progress_bar=False)