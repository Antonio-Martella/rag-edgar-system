import torch
from sentence_transformers import CrossEncoder
from src.utils import config

class RAGReranker:
    def __init__(self, model_path=config.LOCAL_RERANKER_PATH):
        """
        Initialize the Reranker by loading it from the local path.
        If the local path doesn't exist, try using the model ID (fallback).
        """
        # Determine the device (cuda if available, otherwise cpu)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if the local path exists, otherwise use the remote ID
        model_to_load = str(model_path) if model_path.exists() else config.RERANKER_MODEL_ID
        
        print(f"--- Loading Reranker: {model_to_load} on {self.device} ---")
        
        try:
            self.model = CrossEncoder(model_to_load, device=self.device)
        except Exception as e:
            raise RuntimeError(f"❌ Error loading Reranker: {e}")

    # src/retrieval/reranker.py

    def rerank(self, query, chunks, top_n=5):
        if not chunks:
            return []

        # Estraiamo il testo per il modello, ma manteniamo i riferimenti agli oggetti
        texts_to_score = [c['content'] if isinstance(c, dict) else c for c in chunks]
        pairs = [[query, text] for text in texts_to_score]

        scores = self.model.predict(pairs)
        
        # Uniamo l'oggetto originale (dict) con il suo score
        scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        # Restituiamo i top_n oggetti ORIGINALI (dizionari)
        return [chunk for chunk, score in scored_chunks[:top_n]]