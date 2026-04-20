# 📊 RAG Performance Evaluation
Questa directory contiene gli strumenti e i dati necessari per valutare la precisione, la fedeltà e l'affidabilità dell'Edgar RAG Multi-Analyst. L'obiettivo è confrontare le risposte generate dall'IA (Mistral-7B) con un set di dati reali ("Ground Truth") estratti manualmente dal report SEC 10-K di Tesla.

## 🛠 Pre-requisiti e Preparazione
Prima di eseguire la valutazione, è fondamentale che l'ambiente locale sia configurato correttamente e che i dati siano stati indicizzati. Segui questa sequenza di comandi:

### 1. Setup dei Modelli
Scarica il modello di embedding e il modello LLM (quantizzato in 4-bit) per l'esecuzione locale.

```bash
python3 scripts/run_setup_models.py
```

### 2. Ingestion dei Dati (SEC)
Scarica il report ufficiale dalla SEC. Quando richiesto dallo script, inserisci i seguenti valori:
- Azienda (Ticker): `TSLA`
- Tipo Report: `10-K`
- Data: `2025-01-01`

```bash
python3 scripts/run_ingestdata.py
```

### 3. Indicizzazione (Vettorializzazione)
Trasforma il testo del report in vettori numerici (file `.bin`) per permettere la ricerca semantica. Inserisci i valori:
- Azienda: `TSLA`
- Tipo Report: `10-K`

```bash
python3 scripts/run_indexing.py
```
## 🧪 Il Processo di Valutazione
### Il "Ground Truth" (La Verità)
Il file `ground_truth.json` funge da bussola per il nostro test. Contiene 10 domande chiave e le relative risposte certificate basate sul report Tesla 2024/2025.
Queste domande coprono diverse aree critiche:
* Financials: Ricavi totali, Net Income e CapEx.
* Operational: Consegne veicoli e performance del Model Y.
* Risk Factors: Dipendenza dai fornitori e policy commerciali.
* Future Tech: Capacità di calcolo AI e stoccaggio energetico.

### Esecuzione del Test
Per avviare la valutazione automatizzata, esegui:
```bash
python3 evaluation/evaluate_rag.py
```
Lo script interrogherà il RAG, confronterà la risposta generata con quella del `test_queries.json` e salverà un report finale in `eval_report.json`.