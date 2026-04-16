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
    """
    A class to generate vector embeddings from text using a local SentenceTransformer model.
     - The model is loaded from a local path specified in the config, which should be set up running run_setupmodels.py.
     - The encode method takes a list of texts and returns their corresponding embeddings as a NumPy array.
     - The dimension of the embeddings is determined by the model and can be accessed via the dimension attribute.
    """
    def __init__(self, model_path: str = str(config.LOCAL_EMBEDDING_PATH)):
        print(f"Loading LOCAL Embedding Model: {model_path} ---")
        # Check that the model exists locally before loading it (avoids slow errors)
        if not config.LOCAL_EMBEDDING_PATH.exists():
             raise FileNotFoundError(f"❌ Model not found in {model_path}. Run setup_models.py!")
        # Load the model from the local path
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        # Try to get the embedding dimension from the model, if not available, compute it using a dummy input
        try:
            self.dimension = self.model[1].word_embedding_dimension
        except (IndexError, AttributeError):
            dummy_embedding = self.model.encode("", show_progress_bar=False)
            self.dimension = dummy_embedding.shape[0]

    def encode(self, text_list: list) -> 'np.ndarray':
        # Generate embeddings for a list of texts using the loaded model and return them as a NumPy array.
        return self.model.encode(text_list, show_progress_bar=False)