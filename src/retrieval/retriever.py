import json
from src.embedding import Embedder, VectorStorage
from src.retrieval import RAGReranker 

class Retriever:
    """
    This class is the heart of the retrieval system. It is responsible for:
    1. Loading chunks and the FAISS index.
    2. Performing the fast FAISS search to obtain candidates.
    3. Passing the candidates to the Reranker to obtain the final sorted results.
    """
    def __init__(self, index_path: str, chunks_path: str):
        self.embedder = Embedder()
        self.storage = VectorStorage()
        self.storage.load(index_path)
        # We load the chunks from the JSON file, keeping the original structure (dictionaries)
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = data['chunks']
        # Initialize the Reranker
        self.reranker = RAGReranker() 

    def search(self, query: str, initial_k: int = 20, final_k: int = 5) -> list:
        """
        It performs the two-stage search in a super-clean way:
        1. FAISS extracts the top 20 candidates.
        2. The Reranker sorts them and returns the top 5.
        """
        # Search in FAISS to get the initial candidates
        query_vector = self.embedder.encode([query])

        # Searching in FAISS returns candidate distances and indices
        _, indices = self.storage.index.search(query_vector.astype('float32'), initial_k)

        # We convert the indices returned by FAISS into the original chunks (dictionaries) for the Reranker
        faiss_results = []

        for idx in indices[0]:
            if idx != -1: 
                faiss_results.append(self.chunks[idx])
        
        if not faiss_results:
            return []
        
        # Pass the candidates to the Reranker to get the top results
        top_results = self.reranker.rerank(query, faiss_results, top_n=final_k)
        
        return top_results