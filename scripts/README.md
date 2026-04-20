# 🛠️ Scripts Directory (La Sala Macchine)

Questa directory contiene tutti gli script eseguibili per orchestrare il ciclo di vita dell'applicazione: dal download dei modelli di Intelligenza Artificiale, all'elaborazione dei bilanci finanziari (Form 10-K), fino all'esecuzione della chat e ai test di qualità.

Tutti gli script sono progettati per essere eseguiti dalla **root principale** del progetto (non da dentro la cartella `scripts/`).

---

## 🛤️ Ordine di Esecuzione (Pipeline)

Se stai avviando il progetto per la prima volta, l'ordine logico di esecuzione è il seguente:
1. `run_setup_models.py` (Setup AI)
2. `run_ingestion.py` (Download Dati)
3. `run_indexing.py` (Creazione Vettori)
4. `run_rag.py` (Chat) o `run_evaluate_rag.py` (Test)

---

## 📄 Dettaglio degli Script

### 1. `run_setup_models.py` (Inizializzazione AI)
* **Cosa fa:** Legge le variabili in `src/utils/config.py` e scarica i pesi dei modelli (LLM, Reranker, Embedding) da Hugging Face, salvandoli in locale nella cartella `models/`.
* **⚙️ Personalizzazione e Ottimizzazione VRAM:** Questo script è strettamente legato al file `src/utils/config.py`, dove puoi modificare radicalmente il comportamento del sistema prima del download:
  * **Cambio ID Modelli:** Puoi sostituire le stringhe `LLM_MODEL_ID`, `EMBEDDING_MODEL_ID` e `RERANKER_MODEL_ID` per scaricare e testare qualsiasi altro modello Open Source disponibile su Hugging Face (es. passare da Mistral a Llama-3).
  * **Switch di Quantizzazione:** Impostando `QUANTIZATION_SWITCH = True` nel config, istruirai lo script (e l'intera architettura RAG) a utilizzare tecniche di quantizzazione (4-bit o 8-bit). Questo comprime drasticamente il peso dell'LLM, permettendoti di farlo girare su GPU consumer con meno di 16GB di VRAM, sacrificando solo una percentuale di precisione.
* **Quando usarlo:** La prima volta che avvii il progetto, o ogni volta che modifichi i parametri in `config.py` per testare un nuovo setup AI.
* **Esecuzione:** 
```bash
python scripts/run_setup_models.py
```
* *Nota: Richiede una buona connessione internet (scarica fino a ~16GB di dati se la quantizzazione è disattivata).*

### 2. `run_ingestion.py` (Estrazione Dati e Chunking)
* **Cosa fa:** Si collega al database EDGAR della SEC (Securities and Exchange Commission), cerca il Ticker aziendale richiesto (es. TSLA) e scarica il testo del bilancio finanziario (Form 10-K). Oltre al download, si occupa del **Chunking**: "taglia" il documento grezzo in segmenti logici e navigabili, salvando i file elaborati (solitamente in formato `.json`) nella cartella `data/chunks/`.
* **Quando usarlo:** Quando vuoi analizzare una nuova azienda o aggiungere un nuovo anno fiscale al tuo database locale.
* **Esecuzione:** 
```bash
python scripts/run_ingestion.py
```

### 3. `run_indexing.py` (Creazione del "Cervello" Vettoriale)
* **Cosa fa:** Prende i segmenti di testo (chunk) precedentemente preparati e salvati in `data/chunks/`, li converte in vettori matematici utilizzando il modello di Embedding (es. Nomic) e costruisce l'indice di ricerca ad altissima velocità (FAISS). Salva il database vettoriale finale nella cartella `data/embeddings/`.
* **Quando usarlo:** Subito dopo aver eseguito l'ingestion di un nuovo documento, o se decidi di cambiare il tuo modello di Embedding e hai bisogno di ricreare i database vettoriali partendo dai chunk esistenti.
* **Esecuzione:** 
```bash
python scripts/run_indexing.py
```

### 4. `run_rag.py` (Interfaccia a riga di comando - CLI)
* **Cosa fa:** Avvia l'esperienza interattiva direttamente nel terminale, senza l'interfaccia web di Streamlit. Carica i modelli in VRAM (~21 GB), permette di scegliere l'azienda indicizzata e avvia il motore conversazionale. Include il **Self-Evaluating RAG**, mostrando in diretta il ragionamento e il punteggio (1-5) del Giudice interno per ogni risposta generata.
* **Quando usarlo:** Per chattare rapidamente con l'Analista o per il debugging del backend.
* **Esecuzione:** 
```bash
python scripts/run_rag.py
```

### 5. `run_evaluate_rag.py` (Suite di Test Enterprise)
* **Cosa fa:** È il banco di prova definitivo dell'architettura. Esegue un test automatizzato (Regression Testing) passando in rassegna dozzine di domande preimpostate per verificare l'accuratezza del sistema sui bilanci 10-K di Tesla degli anni 2023, 2024 e 2025.
* **Struttura dei Test:** Lo script legge e scrive all'interno della cartella `evaluation/`, che è strutturata rigorosamente in sottocartelle per anno (es. `eval_tsla_10-k_2023`, `eval_tsla_10-k_2024`, ecc.). Ogni cartella contiene:
  * `test_queries_202X.json`: Il file di input che contiene la *ground truth* (le domande e le risposte esatte attese).
  * `eval_report_202X.json`: Il file di output generato automaticamente da questo script a fine esecuzione, contenente il report dettagliato e la percentuale di accuratezza finale.
* **Caratteristiche Tecniche:** * Usa un'architettura a **Front-Loading**: carica l'artiglieria pesante (Modelli) una sola volta in VRAM.
  * Esegue uno **Swap Istantaneo** dei database FAISS passando da un anno all'altro in pochi millisecondi.
  * Utilizza un **Doppio Giudice**: registra il voto di completezza (1-5) del Giudice Interno e usa l'LLM come "Giudice Esterno" per verificare l'accuratezza dei numeri generati contro la *ground truth*.
* **Quando usarlo:** Per certificare le prestazioni del sistema, o per verificare che le modifiche al codice (es. un cambio di LLM o di chunking) non abbiano degradato la qualità delle risposte.
* **Esecuzione:**
```bash 
python scripts/run_evaluate_rag.py
```