# 🏛️ EDGAR RAG System: Local Financial Multi-Analyst

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![AI](https://img.shields.io/badge/AI-Local_LLM-orange.svg)
![RAG](https://img.shields.io/badge/Architecture-Two_Stage_RAG-success.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

An advanced, 100% on-premises **Retrieval-Augmented Generation (RAG) system**, designed for complex financial document analysis (SEC Form 10-K) without compromising data privacy.

---

## 🎯 Project Goal

In the world of corporate finance and auditing, data confidentiality is paramount. Sending financial statements, internal reports, or sensitive documents to proprietary APIs (like OpenAI, Anthropic, or Google) represents an unacceptable security and compliance risk for many companies.

This project demonstrates that **you don't need giant language models in the cloud to achieve enterprise-level financial analytics.** We've built a fully local RAG ecosystem that:
1. **Guarantees Total Privacy:** Not a single byte of data leaves your machine.
2. **Master Complex Documents:** Use **SEC EDGAR (Form 10-K)** reports as a test bed. These documents are notoriously difficult to process due to their length, legal jargon, and complex financial tables.
3. **Optimize Hardware:** Leveraging 8-bit quantization and an optimized architecture, the system runs smoothly on consumer hardware or budget cloud instances (<24GB VRAM).

## ✨ Caratteristiche Principali

* **Two-Stage Retrieval:** Combines the speed of vector search (FAISS + Nomic Embeddings) with the surgical precision of semantic reordering (BAAI Cross-Encoder Reranker).
* **Self-Evaluating RAG (LLM-as-a-Judge):** The system not only responds, but internally evaluates (from 1 to 5) the completeness of its response before displaying it to the user, ensuring transparency on information gaps.
* **Front-Loading VRAM Architecture:** Model weights are loaded into memory only once, allowing for instant *swapping* between balance sheets of different companies or years in fractions of a second.
* **Modular UI:** Responsive and clean web interface developed in Streamlit.

---

## 📂 Repository Structure

The project follows the principles of *Separation of Concerns*. **Each directory contains its own detailed `README.md`.**

```text
rag-edgar-system/
├── data/               # (Dynamically generated) FAISS vector data and JSON chunks
├── evaluation/         # Suite di testing automatizzata e report di benchmark
├── frontend/           # Streamlit User Interface (app.py, components.py)
├── models/             # Locally downloaded weights (LLM, Embedder, Reranker)
├── scripts/            # CLI Tools: AI Setup, Ingestion, Indexing, Testing
├── src/                # Backend Core: RAG Logic, Prompting, LLM Management
├── .gitignore          # Files excluded from versioning (including large templates)
├── Dockerfile          # Configuration for containerization of the environment
├── LICENSE             # License to use
└── requirements.txt    # Necessary Python dependencies
```

---

## 🚀 Installation Guide
**Clone the repository:**
```bash
git clone https://github.com/Antonio-Martella/rag-edgar-system.git
cd rag-edgar-system
```
**Option 1: Local Installation**
1. **Install dependencies:**
    It is recommended to use a virtual environment (e.g. venv or conda).
    ```bash
    pip install -r requirements.txt
    ```
2. Initialize Models:
    Download the required models. You can configure quantization usage in the `src/utils/config.py` file before running this command.
    ```bash
    python scripts/run_setup_models.py
    ```
**Option 2: Docker (Recommended)**
To avoid dependency issues (especially with PyTorch and CUDA), you can build the provided Docker image:
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