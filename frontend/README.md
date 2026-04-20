# 🖥️ Frontend Directory

Questa directory contiene l'Interfaccia Utente (UI) dell'Edgar RAG Multi-Analyst. L'intero frontend è costruito utilizzando **Streamlit**, un framework Python ideale per trasformare script di data science e architetture di machine learning in applicazioni web interattive e reattive.

L'architettura visiva è stata progettata seguendo il principio della modularità, separando la logica di chat dai componenti di navigazione laterale.

## 📂 Struttura dei File

### 1. `app.py` (Main Entry Point)
È il cuore pulsante dell'applicazione web e gestisce la logica principale dell'interazione utente-AI.
* **State Management:** Inizializza e gestisce il `st.session_state` per preservare la cronologia della chat e i modelli RAG caricati in memoria durante i ricaricamenti della pagina.
* **Chat Interface:** Renderizza i messaggi dell'utente e le risposte dell'Analista. Prepara la *history* degli ultimi messaggi per fornire contesto al Large Language Model.
* **Self-Evaluating UI:** Spacchetta i dizionari generati dal backend per alimentare il sistema di "Live Quality Badge". Stampa dinamicamente badge verdi (✅ Verificato) o gialli (⚠️ Incompleto) sotto ogni risposta, rivelando all'utente il log di ragionamento del Giudice interno.

### 2. `components.py` (Modular UI Elements)
Questo file ospita i componenti visivi secondari per mantenere `app.py` pulito, leggibile e scalabile.
* **Sidebar (`render_sidebar`):** Contiene i widget e i form che permettono all'utente di configurare l'analisi. Gestisce l'input del Ticker aziendale (es. TSLA, AAPL), la selezione dell'anno fiscale e il tipo di documento (es. 10-K). Quando l'utente preme "Carica", comunica con il backend per effettuare lo swap istantaneo dei dati vettoriali (FAISS) senza pesare sulla GPU.

## 🚀 Come avviare l'Interfaccia

Assicurati di aver prima completato l'indicizzazione dei documenti tramite gli script di backend.
Per lanciare l'applicazione web, apri il terminale nella root principale del progetto ed esegui il seguente comando:

```bash
streamlit run frontend/app.py
```