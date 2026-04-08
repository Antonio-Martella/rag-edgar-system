import faiss
import numpy as np
import os

# Class to manage the storage of embedding vectors using FAISS
class VectorStorage:
    def __init__(self, dimension=384):
        self.dimension = dimension                # Dimension of the embedding vectors (must correspond to the model used for generating embeddings)
        self.index = faiss.IndexFlatL2(dimension) # Creates a FAISS index for efficient vector search (L2 distance)

    def add_embeddings(self, embeddings):
        """Adds vectors to the FAISS index."""
        self.index.add(np.array(embeddings).astype('float32'))

    def save(self, path):
        """Saves the index to a .bin file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        print(f"--- Index saved to: {path} ---")

    def load(self, path):
        """Loads an existing index."""
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            print(f"--- Index loaded from: {path} ---")
        else:
            raise FileNotFoundError(f"No index found at: {path}")