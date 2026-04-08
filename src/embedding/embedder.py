from sentence_transformers import SentenceTransformer
from src.utils import config

class Embedder:
    def __init__(self, model_name=config.EMBEDDING_MODEL):
        print(f"--- Loading Embedding Model: {model_name} ---")
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def encode(self, text_list):
        """
        Transforms a string or a list of strings into vectors.
        """
        return self.model.encode(sentences=text_list, show_progress_bar=True)