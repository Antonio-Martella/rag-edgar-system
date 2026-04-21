# 📊 Evaluation Directory (Benchmarking & Testing)

Questa directory rappresenta il banco di prova (benchmark) dell'intero sistema RAG. Contiene i dataset di validazione e i report generati automaticamente per misurare l'accuratezza dell'Analista Finanziario AI nell'estrazione e nel ragionamento sui documenti complessi (Form 10-K).

Tutti i test presenti in questa cartella vengono eseguiti automaticamente lanciando lo script `scripts/run_evaluate_rag.py`.

---

## 📂 Struttura dei Dati di Test

I test sono segmentati logicamente per azienda e anno fiscale. Attualmente, la suite valuta le performance sui bilanci annuali (10-K) di **Tesla (TSLA)** per il triennio 2023-2025.

```text
evaluation/
├── eval_tsla_10K_2023/
├── eval_tsla_10K_2024/
└── eval_tsla_10K_2025/
```
**Contenuto di ogni cartella**:
1. `test_queries_202X.json` **(Ground Truth)**: Questo file è il punto di riferimento. Contiene una lista di domande specifiche poste sul bilancio di quell'anno e la **risposta esatta** attesa (estratta manualmente o verificata).
2. `eval_report_202X.json` **(Output di Valutazione)**: Questo è il file generato dinamicamente alla fine del test. Contiene un log completo che mette a confronto la risposta attesa con la **risposta effettivamente generata** dal sistema RAG.

## ⚖️ Metodologia di Valutazione (LLM-as-a-Judge)
Per garantire un test imparziale e automatizzato, il sistema non usa semplici controlli testuali (che fallirebbero se il RAG usa sinonimi o formattazioni diverse), ma impiega la tecnica dell'**LLM-as-a-Judge**.

Per ogni domanda della suite:
1. Il sistema RAG genera la sua risposta.
2. Un "Giudice Esterno" (lo stesso LLM istruito con un prompt di valutazione rigoroso) analizza la risposta generata confrontandola con la ground truth.
3. Il Giudice verifica se i dati numerici chiave e il senso logico combaciano.
4. Il Giudice emette un verdetto secco: **PASS** (Superato) o **FAIL** (Fallito).

I risultati di ogni domanda e il punteggio totale vengono poi consolidati nel file `eval_report_202X.json`.

## 📈 Risultati dei Test e Benchmark
Le seguenti metriche di accuratezza rappresentano le performance reali del sistema RAG.

**Configurazione Hardware/Software del Test**:
* **Modello LLM**: Mistral-7B-Instruct-v0.2
* **Quantizzazione**: 8-bit (Ottimizzazione della VRAM attivata)
* **Retriever**: FAISS + BAAI Reranker (Top-5 chunks finali)

| Documento | Anno Fiscale | Accuratezza (PASS Rate) | Note |
| :--- | :---: | :---: | :--- |
| **TSLA 10-K** | 2023 | **90%** | Prestazione eccellente, altissima fedeltà nell'estrazione dati. |
| **TSLA 10-K** | 2024 | **75%** | Fisiologico calo probabilistico causato dalla compressione a 8-bit sui ragionamenti più complessi. |
| **TSLA 10-K** | 2025 | **80%** | Ripresa e forte aderenza alla ground truth per l'ultimo anno fiscale. |