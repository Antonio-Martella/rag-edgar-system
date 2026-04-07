from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Carica il modello una sola volta all'inizializzazione
        print(f"--- Caricamento modello: {model_name} ---")
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def encode(self, text_list):
        """Trasforma una stringa o una lista di stringhe in vettori."""
        return self.model.encode(sentences=text_list, show_progress_bar=True)