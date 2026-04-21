# 🏛️ EDGAR RAG System: Local Financial Multi-Analyst

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![AI](https://img.shields.io/badge/AI-Local_LLM-orange.svg)
![RAG](https://img.shields.io/badge/Architecture-Two_Stage_RAG-success.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

Un sistema avanzato di **Retrieval-Augmented Generation (RAG) 100% in locale**, progettato per l'analisi complessa di documenti finanziari (SEC Form 10-K) senza compromettere la privacy dei dati.

---

## 🎯 L'Obiettivo del Progetto

Nel mondo della finanza aziendale e dell'auditing, la riservatezza dei dati è fondamentale. Inviare bilanci, report interni o documenti sensibili ad API proprietarie (come OpenAI, Anthropic o Google) rappresenta un rischio di sicurezza e conformità inaccettabile per molte aziende.

Questo progetto dimostra che **non servono modelli linguistici giganti in cloud per ottenere analisi finanziarie di livello Enterprise.** Abbiamo costruito un ecosistema RAG interamente locale che:
1. **Garantisce la Privacy Totale:** Nessun byte di dato lascia la tua macchina.
2. **Domina Documenti Complessi:** Utilizza i report **SEC EDGAR (Form 10-K)** come banco di prova. Questi documenti sono notoriamente difficili da processare a causa della loro lunghezza, del gergo legale e delle complesse tabelle finanziarie.
3. **Ottimizza l'Hardware:** Sfruttando la quantizzazione a 8-bit e un'architettura ottimizzata, il sistema gira fluidamente su hardware consumer o istanze cloud economiche (<24GB VRAM).

## ✨ Caratteristiche Principali

* **Two-Stage Retrieval (Ricerca a due fasi):** Combina la velocità della ricerca vettoriale (FAISS + Nomic Embeddings) con la precisione chirurgica del riordino semantico (BAAI Cross-Encoder Reranker).
* **Self-Evaluating RAG (LLM-as-a-Judge):** Il sistema non si limita a rispondere, ma valuta internamente (da 1 a 5) la completezza della propria risposta prima di mostrarla all'utente, garantendo trasparenza sui "vuoti" informativi.
* **Front-Loading VRAM Architecture:** I pesi dei modelli vengono caricati una sola volta in memoria, permettendo lo *swap* istantaneo tra i bilanci di diverse aziende o anni in frazioni di secondo.
* **UI Modulare:** Interfaccia web reattiva e pulita sviluppata in Streamlit.

---

## 📂 Struttura della Repository

Il progetto segue i principi della *Separation of Concerns*. **Ogni directory contiene un proprio `README.md` dettagliato.**

```text
rag-edgar-system/
├── data/               # (Generata dinamicamente) Dati vettoriali FAISS e Chunks JSON
├── evaluation/         # Suite di testing automatizzata e report di benchmark
├── frontend/           # Interfaccia Utente Streamlit (app.py, components.py)
├── models/             # Pesi scaricati localmente (LLM, Embedder, Reranker)
├── scripts/            # CLI Tools: Setup AI, Ingestion, Indexing, Testing
├── src/                # Backend Core: Logica RAG, Prompting, Gestione LLM
├── .gitignore          # File esclusi dal versioning (inclusi i modelli pesanti)
├── Dockerfile          # Configurazione per la containerizzazione dell'ambiente
└── requirements.txt    # Dipendenze Python necessarie
```

---

## 🚀 Guida all'Installazione
**Opzione 1: Installazione Locale**
1. **Clona la repository:**
    ```bash
    git clone [https://github.com/tuo-username/rag-edgar-system.git](https://github.com/tuo-username/rag-edgar-system.git)
    cd rag-edgar-system
    ```
2. **Installa le dipendenze:**
    Si consiglia di utilizzare un ambiente virtuale (es. venv o conda).
    ```bash
    pip install -r requirements.txt
    ```
3. Inizializza i Modelli AI:
    Scarica i modelli necessari da Hugging Face (circa 16GB). Puoi configurare l'uso della quantizzazione nel file src/utils/config.py prima di eseguire questo comando.
    ```bash
    python scripts/run_setup_models.py
    ```
**Opzione 2: Docker (Consigliata)**
Per evitare problemi di dipendenze (specialmente con PyTorch e CUDA), puoi buildare l'immagine Docker fornita:
```bash
docker build -t edgar-rag-system -f Dockerfile .
docker run --gpus all -p 8501:8501 edgar-rag-system
```
---

## 💻 Utilizzo Rapido (Pipeline)
Per iniziare a interrogare un bilancio basta avviare l'interfaccia web con il comando:
```bash
streamlit run frontend/app.py
```
da li è possibile selezionare il report (10-K) della società di interesse e il relativo anno.

---

## 📊 Valutazione e Benchmark
Il sistema include una suite di test automatizzata (`scripts/run_evaluate_rag.py`) che utilizza la tecnica LLM-as-a-Judge contro una Ground Truth verificata manualmente.

Negli stress-test sui Form 10-K di Tesla (2023-2025), operando con modelli quantizzati a 8-bit, **l'architettura ha mantenuto un'accuratezza media di superamento (PASS rate) superiore all'81%**, dimostrando un'eccellente capacità di estrarre e calcolare metriche finanziarie esatte da testo e tabelle non strutturate.

(Per i log dettagliati, consulta la directory `evaluation/`).

---

## 🔮 Sviluppi Futuri (Roadmap)
Il sistema getta basi solide, ma ci sono diverse aree di evoluzione già pianificate:

1. **Migliore Linearizzazione delle Tabelle**: Attualmente i PDF/HTML complessi vengono parsati in modo standard. L'integrazione di parser semantici avanzati (es. LlamaParse o layout-parser) permetterà una lettura spaziale delle tabelle finanziarie, riducendo le allucinazioni sui dati incollonnati.

2. **Iniezione di Metadati Avanzati**: Arricchire ogni chunk testuale con metadati dinamici (es. `{"Sezione": "MD&A", "Anno": 2023, "Topic": "Risk Factors"}`). Questo permetterà al Retriever di fare pre-filtraggio hard-coded prima della ricerca vettoriale, aumentando drasticamente la precisione.

3. **Agentic Loop (Self-Correction)**: Evolvere il "Giudice Interno" da un ruolo passivo (che avvisa l'utente se manca un dato) a un ruolo attivo, istruendo l'LLM a reiterare la ricerca in background finché il punteggio non raggiunge il 5/5.

4. **Multimodal RAG**: Espandere la pipeline di ingestion per supportare il riconoscimento e l'analisi dei grafici e dei chart presenti nei report annuali tramite modelli Vision-Language (VLM).

---

## 🤝 Contributi e Licenza
I contributi sono benvenuti! Se hai idee per migliorare il chunking finanziario o ottimizzare la pipeline, apri una Issue o invia una Pull Request.

Distribuito con licenza MIT. Vedi il file `LICENSE` per maggiori informazioni.