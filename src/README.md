# 🧩 Source Directory (`src/`)

Welcome to the beating heart of Edgar RAG Multi-Analyst. This directory contains all the application's source code (backend).

The system was architected with a **Modular and Enterprise-Grade** approach: the data extraction, vectorization, search, and generation processes are strictly separated into submodules. This ensures clean, easily testable, and highly scalable code.

---

## 🗺️ Architecture Map

The directory is divided into six main modules. Each manages a specific step in the RAG pipeline.

### 📥 1. `ingestion/` (Data Acquisition)
This module is responsible for taking raw data from the outside world and preparing it for the system.
* **`downloader.py`**: Interacts with the SEC (EDGAR) API to download the requested company's official Form 10-K.
* **`parser.py`**: It cleans raw documents, removing HTML/XML tags or unnecessary noise to extract only pure financial text and linearizes tables (essential!).
* **`chuncking.py`**: It applies text-splitting algorithms to cut text into navigable "chunks" that are compatible with the context window of AI models.
* **`pipeline.py`**: Orchestrate the entire ingestion flow (Download -> Parse -> Chunk).

### 🧮 2. `embedding/` (Vectorization)
Transform human text into machine-readable format (mathematical vectors).
* **`embedder.py`**: It manages the embedding model (e.g. Nomic) to translate chunks into dense high-dimensional vectors.
* **`storage.py`**: Manages the local vector database (FAISS). Creates, saves, and loads the ultra-fast search index (`.bin`).
* **`pipeline.py`**: Orchestrates the indexing process (Gets chunks -> Embeds them -> Saves to FAISS database).

### 🔎 3. `retrieval/` (Search Engine)
It implements **Two-Stage Retrieval** architecture to ensure maximum accuracy in document extraction.
* **`retriever.py`**: The "Seeker". Query the FAISS database to quickly retrieve the top-K most similar documents (approximate search).
* **`reranker.py`**: The "Reviewer". It uses a Cross-Encoder model (e.g., BAAI) to reanalyze the FAISS results by cross-referencing them with the user's query, reordering them, and accurately filtering out false positives.

### 🧠 4. `llm/` (Artificial intelligence)
Handles the initialization and inference of local Large Language Models.
* **`model.py`**: Manages the physical loading of weights into VRAM, configuring hardware, device map and quantization options (4-bit).
* **`prompt.py`**: Stores system templates (System Prompts). Structures RAG messages by combining history, extracted context, and personality instructions (The Financial Analyst).
* **`generator.py`**: The executive class. It executes the model, applies chat templates, and generates the final text. It includes methods for standard inference and "naked" queries (raw prompts).

### ⚙️ 5. `rag/` (The Orchestrator)
The point of convergence of all the previous modules.
* **`service.py`**: The RAGService class serves as an internal API for the frontend. It handles front-loading of large models (avoiding GPU crashes) and instant database swapping. Most importantly, it hosts the RAG Self-Evaluating (LLM-as-a-Judge) logic, evaluating the completeness of its responses before sending them to the user.

### 🛠️ 6. `utils/` (General Utilities)
* **`config.py`**: The application control panel. It contains global variables, dynamic paths (absolute path resolution), Hugging Face model IDs, and switches to enable or disable features like quantization.

---

## 🌊 Execution Flow (The "Journey" of a Question)

When a user asks a question from the frontend (e.g. *"What are Tesla's revenues in 2023?"*), here's what happens behind the scenes in `src/`:

1. **`rag.service`** receives the query and passes it to **`retrieval.retriever`**.
2. The retriever uses **`embedding.embedder`** to vectorize the query.
3. The vector query is looked up in **`embedding.storage`** (FAISS), which returns 20 raw documents.
4. These 20 documents are passed to **`retrieval.reranker`**, which returns the 5 absolutely perfect documents.
5. The 5 documents and the query travel to **`llm.generator`**, passing through **`llm.prompt`**, which formats them elegantly.
6. The LLM generates the financial answer.
7. Before submitting it, **`rag.service`** uses an isolated method of **`llm.generator`** to act as a *Judge*, evaluating the answer from 1 to 5.
8. The answer + score are returned to the frontend.