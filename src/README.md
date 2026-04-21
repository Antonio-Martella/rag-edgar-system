# 🧩 Source Directory (`src/`)

Benvenuto nel cuore pulsante dell'Edgar RAG Multi-Analyst. Questa directory contiene tutto il codice sorgente (backend) dell'applicazione. 

Il sistema è stato architettato con un approccio **Modulare ed Enterprise-Grade**: i processi di estrazione dati, vettorializzazione, ricerca e generazione sono rigorosamente separati in sottomoduli. Questo garantisce un codice pulito, facilmente testabile e altamente scalabile.

---

## 🗺️ Mappa dell'Architettura

La directory è divisa in 6 moduli principali. Ognuno gestisce un passaggio specifico della pipeline RAG.

### 📥 1. `ingestion/` (Acquisizione Dati)
Questo modulo si occupa di prelevare i dati grezzi dal mondo esterno e prepararli per il sistema.
* **`downloader.py`**: Interagisce con le API della SEC (EDGAR) per scaricare i Form 10-K ufficiali dell'azienda richiesta.
* **`parser.py`**: Pulisce i documenti grezzi, rimuovendo tag HTML/XML o rumore inutile per estrarre solo il testo finanziario puro e linearizza le tabelle (essenziale!).
* **`chuncking.py`**: Applica algoritmi di text-splitting per tagliare il testo in "chunk" navigabili e compatibili con la finestra di contesto dei modelli AI.
* **`pipeline.py`**: Orchestra l'intero flusso di ingestion (Download -> Parse -> Chunk).

### 🧮 2. `embedding/` (Vettorializzazione)
Trasforma il testo umano in un formato comprensibile alle macchine (vettori matematici).
* **`embedder.py`**: Gestisce il modello di embedding (es. Nomic) per tradurre i chunk in vettori densi ad alta dimensionalità.
* **`storage.py`**: Gestisce il database vettoriale locale (FAISS). Crea, salva e carica l'indice per la ricerca ultra-rapida (`.bin`).
* **`pipeline.py`**: Orchestra il processo di indicizzazione (Prende i chunk -> Li embedda -> Salva il database FAISS).

### 🔎 3. `retrieval/` (Motore di Ricerca)
Implementa l'architettura **Two-Stage Retrieval** per garantire la massima precisione nell'estrazione dei documenti.
* **`retriever.py`**: Il "Cercatore". Interroga il database FAISS per recuperare velocemente i top-K documenti più simili (ricerca approssimata).
* **`reranker.py`**: Il "Revisore". Utilizza un modello Cross-Encoder (es. BAAI) per ri-analizzare i risultati di FAISS incrociandoli con la domanda dell'utente, riordinandoli e filtrando i falsi positivi con precisione.

### 🧠 4. `llm/` (Intelligenza Artificiale)
Gestisce l'inizializzazione e l'inferenza dei Large Language Models locali.
* **`model.py`**: Gestisce il caricamento fisico dei pesi in VRAM, configurando hardware, device map e opzioni di quantizzazione (4-bit, 8-bit).
* **`prompt.py`**: Custodisce i template di sistema (System Prompts). Struttura i messaggi RAG combinando history, contesto estratto e istruzioni della personalità (L'Analista Finanziario).
* **`generator.py`**: La classe esecutiva. Esegue il modello, applica i chat-template e genera il testo finale. Include metodi per l'inferenza standard e per query "nude" (raw prompts).

### ⚙️ 5. `rag/` (L'Orchestratore)
Il punto di convergenza di tutti i moduli precedenti.
* **`service.py`**: La classe `RAGService` funge da API interna per il frontend. Gestisce il **Front-Loading** dei modelli pesanti (evitando crash della GPU) e lo Swap istantaneo dei database. Soprattutto, ospita la logica del **Self-Evaluating RAG (LLM-as-a-Judge)**, valutando la completezza delle proprie risposte prima di inviarle all'utente.

### 🛠️ 6. `utils/` (Utility Generali)
* **`config.py`**: Il pannello di controllo dell'applicazione. Contiene le variabili globali, i path dinamici (risoluzione dei percorsi assoluti), gli ID dei modelli Hugging Face e gli interruttori (switch) per attivare o disattivare feature come la quantizzazione.

---

## 🌊 Flusso di Esecuzione (Il "Viaggio" di una Domanda)

Quando un utente fa una domanda dal frontend (es. *"Quali sono i ricavi di Tesla nel 2023?"*), ecco cosa succede dietro le quinte in `src/`:

1. **`rag.service`** riceve la query e la passa a **`retrieval.retriever`**.
2. Il retriever usa **`embedding.embedder`** per vettorializzare la domanda.
3. La domanda vettoriale viene cercata in **`embedding.storage`** (FAISS) che restituisce 20 documenti grezzi.
4. Questi 20 documenti passano per **`retrieval.reranker`**, che restituisce i 5 documenti assolutamente perfetti.
5. I 5 documenti e la domanda viaggiano verso **`llm.generator`**, passando per **`llm.prompt`** che li formatta elegantemente.
6. L'LLM genera la risposta finanziaria.
7. Prima di consegnarla, **`rag.service`** usa un metodo isolato di **`llm.generator`** per agire da *Giudice*, valutando la risposta da 1 a 5.
8. La risposta + il punteggio tornano al frontend.