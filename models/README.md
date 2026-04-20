# 🧠 Models Directory

Questa directory funge da "cervello locale" per l'Edgar RAG Multi-Analyst. Ospita i pesi (weights) e le configurazioni dei modelli di Intelligenza Artificiale necessari per far funzionare la pipeline di Retrieval-Augmented Generation (RAG) interamente offline e sulla tua macchina.

Per motivi di spazio, i file dei modelli **non sono tracciati su Git** (sono inclusi nel `.gitignore`). Vanno scaricati localmente tramite l'apposito script di setup.

## 📂 Struttura della Directory

```text
models/
├── EMBEDDING/   # Modelli per la vettorializzazione del testo (Prima fase di ricerca)
├── LLM/         # Large Language Models per la generazione delle risposte
└── RERANKER/    # Modelli Cross-Encoder per il riordino semantico (Seconda fase di ricerca)
```
## ⚙️ Come scaricare e configurare i modelli
Il download dei modelli è completamente automatizzato tramite la libreria Hugging Face.

1. **Scaricare i modelli**: Esegui lo script di setup dalla cartella principale del progetto:
    ```bash
    python scripts/run_setup_model.py
    ```
    Nota: Il download iniziale può richiedere del tempo e scaricherà circa 16-17 GB se si sceglie di non quantizzare il modello LLM.

2. **Cambiare i modelli**: I modelli da utilizzare non sono hard-coded, ma centralizzati. Se desideri testare un nuovo modello, ti basta modificare le seguenti variabili nel file `src/utils/config.py`:
    ```python
    EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
    RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
    LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    ```

## 🔬 Architettura dell'AI: I modelli di Default
Il sistema utilizza un'architettura **Two-Stage Retrieval** (Ricerca a due fasi) supportata da un LLM generativo. Ecco cosa fa ciascun componente e perché sono stati scelti questi specifici modelli di default.
1. **Il Modello di Embedding**
    * **Il Ruolo**: L'Embedding è il "traduttore matematico". Prende i paragrafi di testo (chunk) dei bilanci aziendali e li converte in vettori (liste di migliaia di numeri) posizionandoli in uno spazio multidimensionale. Quando l'utente fa una domanda, anche questa viene vettorializzata per trovare i testi semanticamente più "vicini" usando algoritmi veloci come FAISS.

    * **Il Modello di Default** (`nomic-embed-text-v1.5`): Un modello open-source altamente efficiente basato su architettura transformer.

        * **Punto di forza**: A differenza dei modelli standard che supportano solo 512 token, Nomic gestisce nativamente una finestra di contesto di **8192 token**. Questo lo rende eccezionale per documenti finanziari lunghi, catturando il contesto di intere pagine senza tagliare informazioni cruciali.

        * **Peso**: Estremamente leggero (~0.5 GB).

2. **Il Modello di Reranking**
    * **Il Ruolo**: Il Reranker agisce come un "revisore al microscopio". Mentre l'Embedding estrae velocemente decine di documenti "probabilmente utili", il Reranker usa un'architettura *Cross-Encoder* per leggere simultaneamente la domanda dell'utente e ogni singolo documento estratto, assegnando un punteggio di pertinenza chirurgico e scartando i falsi positivi.

    * **Il Modello di Default** (`BAAI/bge-reranker-v2-m3`): Creato dalla Beijing Academy of Artificial Intelligence (BAAI), è attualmente uno dei modelli di reranking state-of-the-art a livello mondiale.
        
        * **Punto di forza**: Il suffisso "M3" indica la sua incredibile natura **Multi-lingua**. È capace di capire sfumature logiche complesse (es. la differenza causale in una sezione MD&A di un bilancio) con una precisione spaventosa.

        * **Peso**: Un modello denso e "pesante" per la sua categoria (~2.5 GB di VRAM), giustificato dalle sue prestazioni eccellenti.

3. **Il Large Language Model (LLM)**
    * **Il Ruolo**: L'LLM è "L'Analista e il Giudice". Riceve la domanda dell'utente e i documenti esatti filtrati dal Reranker. Non cerca su internet: utilizza esclusivamente i documenti forniti per ragionare, sintetizzare e formulare una risposta discorsiva chiara e finanziariamente accurata. Nel nostro sistema, agisce anche da Giudice Interno (Self-Evaluating RAG) per valutare se la propria risposta è completa rispetto alla domanda.

    * **Il Modello di Default** (`Mistral-7B-Instruct-v0.2`):
        * Sviluppato da Mistral AI in Francia, è un modello da 7 miliardi di parametri ottimizzato per seguire istruzioni (Instruct).

        * **Punto di forza**: Oltre alla finestra di contesto di 32k token, questa versione (v0.2) ha rimosso i vecchi limiti di sliding window attention, risultando in un ragionamento logico di livello superiore. Batte molti modelli di dimensioni doppie sulla precisione numerica ed è fenomenale nel formattare l'output in JSON per i task del nostro Giudice interno.

        * **Peso**: Richiede circa 14-15 GB di memoria per i pesi, arrivando a consumare fino a 21 GB di VRAM operativa (insieme agli altri modelli) durante l'inferenza e il calcolo della KV Cache. Questo spazio puo essere diminuito enormemente applicando in `src/utils/config.py` la quantizzazione a `4 bit` in `QUANTIZATION_SWITCH = True`.