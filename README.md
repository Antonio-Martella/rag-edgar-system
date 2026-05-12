# 🏛️ EDGAR RAG System: Local Financial Multi-Analyst

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![AI](https://img.shields.io/badge/AI-Local_LLM-orange.svg)
![RAG](https://img.shields.io/badge/Architecture-Two_Stage_RAG-success.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![NVIDIA](https://img.shields.io/badge/Hardware-NVIDIA_GPU-76B900.svg)

An advanced, 100% on-premises **Retrieval-Augmented Generation (RAG) system**, designed for complex financial document analysis (SEC Form 10-K) without compromising data privacy.

---

## 🎯 Project Goal

In the world of corporate finance and auditing, data confidentiality is paramount. Sending financial statements, internal reports, or sensitive documents to proprietary APIs (like OpenAI, Anthropic, or Google) represents an unacceptable security and compliance risk for many companies.

This project demonstrates that **you don't need giant language models in the cloud to achieve enterprise-level financial analytics.** We've built a fully local RAG ecosystem that:
1. **Guarantees Total Privacy:** Not a single byte of data leaves your machine.
2. **Master Complex Documents:** Use **SEC EDGAR (Form 10-K)** reports as a test bed. These documents are notoriously difficult to process due to their length, legal jargon, and complex financial tables.
3. **Optimize Hardware:** Leveraging 8-bit quantization and an optimized architecture, the system runs smoothly on consumer hardware or budget cloud instances (<24GB VRAM).

## ✨ Main Features

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
2. **Initialize Models:**
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

## 💻 Quick Use (Pipeline)
To start querying a balance sheet, simply launch the web interface with the command:
```bash
streamlit run frontend/app.py
```
from there you can select the report (10-K) of the company of interest and of a specific year.

---

## 📊 Valutazione e Benchmark
The system includes an automated test suite (`scripts/run_evaluate_rag.py`) that uses the *LLM-as-a-Judge* technique against a manually verified Ground Truth.

In stress tests on Tesla’s Form 10-K (2023-2025), operating with 8-bit quantized models, **the architecture maintained an average pass rate above 81%**, demonstrating excellent ability to extract and calculate accurate financial metrics from text and unstructured tables.

(For detailed logs, see the `evaluation/` directory.)

---

## 🔮 Future Developments (Roadmap)
The system lays solid foundations, but there are several areas of evolution already planned:

1. **Hybrid Search (Metadata Pre-filtering):** Currently, the *ingestion* process already surgically enriches each document with structured metadata and in-text tags (e.g., `[COMPANY: AAPL | FY: 2023 | SECTION: MD&A]`). The next architectural step involves actively using these fields to enable **Hybrid Search**: applying precise deterministic filters (e.g., searching *only* in chunks where `year == 2023` and `section == "Risk Factors"`) *before* launching the vector search on FAISS. This will further reduce calculation times and eliminate the risk of cross-contamination between different years.

2. **Agentic Loop (Self-Correction)**: Evolve the “Internal Judge” from a passive role (which alerts the user if a piece of data is missing) to an active role, instructing the LLM to repeat the search in the background until the score reaches 5/5.

3. **Multimodal RAG**: Expand the ingestion pipeline to support the recognition and analysis of graphs and charts in annual reports using Vision-Language (VLM) models.

---

## 🖥️ Hardware Requirements (Important)
This project makes heavy use of **PyTorch** and CUDA acceleration.

To run the models locally (LLM, Embedder, Reranker) in a reasonable time, an NVIDIA GPU is **strictly required**.

* **Development and Test Environment:** The system was developed, optimized, and certified on a **Google Colab Pro environment (NVIDIA L4 GPU - 24GB VRAM)**.
* **Minimum Requirements:** Thanks to the 8-bit quantization option, the system can run on consumer NVIDIA GPUs with at least 16GB of VRAM.
* *Note for Mac/AMD users:* Running purely on CPUs or via MPS (Apple Silicon) is not currently supported or optimized and would result in extremely long inference times.

---

## 🤝 Contributions and License
Contributions are welcome! If you have ideas for improving financial chunking or optimizing the pipeline, open an Issue or submit a Pull Request.

Distributed under the MIT License. See the `LICENSE` file for more information.
