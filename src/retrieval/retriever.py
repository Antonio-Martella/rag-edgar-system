import json
from src.embedding import Embedder, VectorStorage
from src.retrieval import RAGReranker 

class Retriever:
    """
    This class is the heart of the retrieval system. 
    Models (Embedder/Reranker) are loaded ONLY ONCE in memory.
    Data (FAISS index/Chunks) are swapped dynamically on request.
    """
    def __init__(self):
        print("--- Initializing Retrieval Models (Embedder & Reranker) ---")
        # 1. LOAD HEAVY MODELS ONLY ONCE ON GPU
        self.embedder = Embedder()
        self.reranker = RAGReranker() 
        
        # Placeholders for dynamic data
        self.storage = VectorStorage()
        self.chunks = []

    def load_data(self, index_path: str, chunks_path: str):
        """
        Swaps out the FAISS index and JSON chunks without reloading ML models.
        Takes milliseconds and 0 extra VRAM.
        """
        # Reset storage to avoid mixing old data
        self.storage = VectorStorage() 
        self.storage.load(index_path)
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = data['chunks']

    def search(self, query: str, initial_k: int = 20, final_k: int = 5) -> list:
        """
        Two-stage search: FAISS extracts top 20, Reranker sorts top 5.
        """
        if not self.chunks:
            return []
            
        # Search in FAISS to get the initial candidates
        query_vector = self.embedder.encode([query])

        # Searching in FAISS returns candidate distances and indices
        _, indices = self.storage.index.search(query_vector.astype('float32'), initial_k)

        # We convert the indices returned by FAISS into the original chunks for the Reranker
        faiss_results = []
        for idx in indices[0]:
            if idx != -1: 
                faiss_results.append(self.chunks[idx])
        
        if not faiss_results:
            return []
        
        # Pass the candidates to the Reranker to get the top results
        top_results = self.reranker.rerank(query, faiss_results, top_n=final_k)
        
        return top_results