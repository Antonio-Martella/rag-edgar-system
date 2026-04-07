import sys
import os
from src.retrieval.retriever import Retriever
from src.llm.generator import LLMGenerator

def main():
    # Percorsi dei dati (assicurati che esistano!)
    index_path = "data/embeddings/tsla_index.bin"
    chunks_path = "data/chunks/tsla_10k_2025_chunks.json"

    # Verifica che i file necessari esistano prima di procedere
    if not os.path.exists(index_path):
        print("❌ Errore: Indice non trovato. Generalo prima col notebook!")
        return

    if not os.path.exists(chunks_path):
        print("❌ Errore: Chunk non trovati. Generali prima col notebook!")
        return

    # Inizializzazione dei moduli di Retrieval e Generazione
    print("🚀 Avvio del Sistema RAG (Tesla Analyst Bot)...")
    retriever = Retriever(index_path, chunks_path)
    generator = LLMGenerator()

    # Interfaccia via terminale
    print("\n--- Sistema Pronto! Digita 'exit' per uscire ---")
    
    # Loop principale per interagire con l'utente
    while True:
        # Chiedi la domanda dell'utente
        query = input("\nDomanda (es. 'What are the risks?'): ")

        # Permetti all'utente di uscire dal loop digitando 'exit' o 'quit'
        if query.lower() in ['exit', 'quit']:
            break
            
        print("\n🔍 Ricerca nei documenti in corso...")
        relevant_chunks = retriever.search(query, k=4)
        
        print("🧠 Elaborazione risposta con LLM...")
        answer = generator.generate_answer(query, relevant_chunks)
        
        print("\n" + "="*60)
        print(f"RISPOSTA:\n{answer}")
        print("="*60)

if __name__ == "__main__":
    main()