import faiss
import numpy as np
import os

# Classe per gestire lo storage dei vettori di embedding usando FAISS
class VectorStorage:
    def __init__(self, dimension=384):
        self.dimension = dimension                # Dimensione dei vettori di embedding (deve corrispondere al modello usato per generare gli embedding)
        self.index = faiss.IndexFlatL2(dimension) # Crea un indice FAISS per la ricerca efficiente dei vettori (L2 distance)

    def add_embeddings(self, embeddings):
        """Aggiunge i vettori all'indice FAISS."""
        self.index.add(np.array(embeddings).astype('float32'))

    def save(self, path):
        """Salva l'indice in un file .bin."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        print(f"--- Indice salvato in: {path} ---")

    def load(self, path):
        """Carica un indice esistente."""
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            print(f"--- Indice caricato da: {path} ---")
        else:
            raise FileNotFoundError(f"Nessun indice trovato a: {path}")