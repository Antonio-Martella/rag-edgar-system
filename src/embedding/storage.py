import faiss
import numpy as np
import os

# Class to manage the storage of embedding vectors using FAISS
class VectorStorage:
    """
    A class to manage the storage of embedding vectors using FAISS.
    It allows adding vectors to the index, saving the index to disk, and loading an existing index. 
    The index is created for L2 distance, which is suitable for many embedding-based applications.
    """
    def __init__(self, dimension=384):
        self.dimension = dimension                # Dimension of the embedding vectors (must correspond to the model used for generating embeddings)
        self.index = faiss.IndexFlatL2(dimension) # Creates a FAISS index for efficient vector search (L2 distance)

    def add_embeddings(self, embeddings: list) -> None:
        """
        Adds vectors to the FAISS index.
        Args:
            embeddings (list of list of float): A list of embedding vectors to be added to the index.
        """
        # Convert the list of embeddings to a NumPy array of type float32 and add it to the index
        self.index.add(np.array(embeddings).astype('float32'))

    def save(self, path: str) -> None:
        """
        Saves the index to a .bin file.
        Args:
            path (str): The path where the index will be saved.
        """
        # Ensure the directory exists before saving the index
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save the FAISS index to the specified path
        faiss.write_index(self.index, path)
        print(f"--- Index saved to: {path} ---")

    def load(self, path: str) -> None:
        """
        Loads an existing index.
        Args:
            path (str): The path from which the index will be loaded.
        """
        # Check if the specified path exists and load the index, otherwise raise an error
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            print(f"--- Index loaded from: {path} ---")
        else:
            raise FileNotFoundError(f"No index found at: {path}")