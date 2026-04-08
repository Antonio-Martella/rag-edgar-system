from sentence_transformers import SentenceTransformer
from src.utils import config

class Embedder:
    def __init__(self, model_path=str(config.LOCAL_EMBEDDING_PATH)):

        print(f"Loading LOCAL Embedding Model: {model_path} ---")
        # Check that the model exists locally before loading it (avoids slow errors)
        if not config.LOCAL_EMBEDDING_PATH.exists():
             raise FileNotFoundError(f"❌ Model not found in {model_path}. Run setup_models.py!")
             
        self.model = SentenceTransformer(model_path)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, text_list):
        return self.model.encode(text_list, show_progress_bar=False)