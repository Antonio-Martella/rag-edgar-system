import json
from src.embedding.embedder import Embedder
from src.embedding.storage import VectorStorage

class Retriever:
    def __init__(self, index_path, chunks_path):
        # Inizializziamo il modello e carichiamo l'indice
        self.embedder = Embedder()
        self.storage = VectorStorage()
        self.storage.load(index_path)
        
        # Carichiamo i testi (i chunk) associati ai vettori
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = data['chunks']

    def search(self, query, k=3):
        """Esegue la ricerca semantica e restituisce i chunk più rilevanti."""
        query_vector = self.embedder.encode([query])
        distances, indices = self.storage.index.search(query_vector.astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1: # FAISS restituisce -1 se non trova nulla
                results.append(self.chunks[idx])
        
        return results