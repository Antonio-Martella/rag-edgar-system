import torch
from sentence_transformers import CrossEncoder
from src.utils import config

def setup_reranker_model() -> None:
    """
    Download and save the reranker model locally if it doesn't exist.
    """
    # Check if the reranker model is already downloaded and saved locally, if not, download and save it.
    if not config.LOCAL_RERANKER_PATH.exists():
        print(f"📥 Downloading Reranker: {config.RERANKER_MODEL_ID}...")
        reranker = CrossEncoder(config.RERANKER_MODEL_ID)
        reranker.save(str(config.LOCAL_RERANKER_PATH))
        print(f"✅ Reranker saved in {config.LOCAL_RERANKER_PATH}")
    else:
        print("✔️ Reranker already present locally.")

class RAGReranker:
    """
    This class is responsible for re-ranking the candidates retrieved by FAISS.
    It uses a CrossEncoder model to score the relevance of each candidate with respect to the query.
    The candidates are expected to be in their original form (dictionaries) so that we can maintain all the metadata and structure for the final output.
    The Reranker will return the top N candidates sorted by their relevance score, but it will return the original dictionary objects, not just the text,
    to preserve all the information for downstream use.
    """
    def __init__(self, model_path = config.LOCAL_RERANKER_PATH):
        """
        Initialize the Reranker by loading it from the local path.
        If the local path doesn't exist, try using the model ID (fallback).
        """
        # Set the device for the model (GPU if available, otherwise CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Determine the model to load: prefer local path, but fallback to model ID if local file is missing
        model_to_load = model_path if model_path.exists() else config.RERANKER_MODEL_ID
        print(f"--- Loading Reranker: {model_to_load} on {self.device} ---")
        # Try to load the model, and handle any exceptions that may occur during loading
        try:
            self.model = CrossEncoder(model_to_load, device=self.device)
        except Exception as e:
            raise RuntimeError(f"❌ Error loading Reranker: {e}")

    def rerank(self, query: str, chunks: list, top_n: int = 5) -> list:
        """
        Re-rank the candidate chunks based on their relevance to the query.
        """
        # If there are no chunks to rerank, return an empty list immediately
        if not chunks:
            return []
        # Prepare the input for the CrossEncoder: we need pairs of (query, chunk_text).
        texts_to_score = [c['content'] if isinstance(c, dict) else c for c in chunks]
        # Let's create the pairs (query, text) for the CrossEncoder
        pairs = [[query, text] for text in texts_to_score]
        # Get the relevance scores from the model for each pair
        scores = self.model.predict(pairs)
        # Now we have the scores, we need to sort the chunks based on these scores.
        scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        return [chunk for chunk, score in scored_chunks[:top_n]]