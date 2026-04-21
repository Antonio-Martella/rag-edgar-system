# 🧠 Models Directory

This directory serves as the "local brain" for Edgar RAG Multi-Analyst. It hosts the weights and configurations of the AI ​​models needed to run the Retrieval-Augmented Generation (RAG) pipeline entirely offline and on your machine.

Due to space constraints, the model files are not tracked on Git (they are included in `.gitignore`). They must be downloaded locally using the appropriate setup script.

## 📂 Directory Structure

```text
models/
├── EMBEDDING/   
├── LLM/        
└── RERANKER/    
```
## ⚙️ How to download and configure models
Model downloading is fully automated using the Hugging Face library.

1. **Download models**: Run the setup script from the project's root folder:
    ```bash
    python scripts/run_setup_model.py
    ```
    Note: The initial download may take some time and will download approximately 16-17 GB if you choose not to quantize the LLM model.

2. **Changing Models**: The models to be used are not hard-coded, but centralized. If you want to test a new model, simply change the following variables in the `src/utils/config.py` file:
    ```python
    EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
    RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
    LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    ```

## 🔬 AI Architecture: Default Models
The system uses a **Two-Stage Retrieval** architecture supported by a generative LLM. Here's what each component does and why these specific default models were chosen.
1. **The Embedding Model**
    * **The Role**: The Embedding is the "mathematical translator." It takes paragraphs of text (chunks) from company financial statements and converts them into vectors (lists of thousands of numbers), positioning them in a multidimensional space. When the user asks a question, it is also vectorized to find the semantically closest texts using fast algorithms like FAISS.

    * **The Default Template** (`nomic-embed-text-v1.5`): A highly efficient open-source template based on a transformer architecture.

        * **Strength**: Unlike standard templates that only support 512 tokens, Nomic natively handles a context window of **8192 tokens**. This makes it exceptional for long financial documents, capturing the context of entire pages without cutting off crucial information.

        * **Weight**: Extremely lightweight (~0.5 GB).

2. **The Reranking Model**
    * **The Role**: The Reranker acts as a "microscopic reviewer." While the Embedding quickly extracts dozens of "likely useful" documents, the Reranker uses a *Cross-Encoder* architecture to simultaneously read the user's query and each individual extracted document, assigning a surgical relevance score and discarding false positives.

    * **The Default Model** (`BAAI/bge-reranker-v2-m3`): Created by the Beijing Academy of Artificial Intelligence (BAAI), it is currently one of the world's state-of-the-art reranking models.

        * **Strength**: The "M3" suffix indicates its incredible **Multilingual** nature. It is capable of understanding complex logical nuances (e.g., the causal difference in an MD&A section of a balance sheet) with frightening accuracy.

        * **Weight**: A dense and "heavy" model for its category (~2.5 GB of VRAM), justified by its excellent performance.

3. **The Large Language Model (LLM)**
    * **The Role**: The LLM is "The Analyst and Judge." It receives the user's question and the exact documents filtered by the Reranker. It does not search the internet: it uses exclusively the documents provided to reason, synthesize, and formulate a clear and financially accurate discursive response. In our system, it also acts as an Internal Judge (Self-Evaluating RAG) to assess whether its response is complete with respect to the question.

    * **The Default Model** (`Mistral-7B-Instruct-v0.2`):
        * Developed by Mistral AI in France, it is a 7 billion parameter model optimized for following instructions (Instructs).

        * **Strength**: In addition to the 32k token context window, this version (v0.2) has removed the old sliding window attention limitations, resulting in higher-level logical reasoning. It beats many double-sized models on numerical precision and is phenomenal at formatting output into JSON for our internal Adjudicator tasks.

        * **Weight**: Requires approximately 14-15 GB of memory for weights, consuming up to 21 GB of operational VRAM (together with other models) during inference and KV Cache calculation. This space can be significantly reduced by applying 8-bit quantization in `src/utils/config.py` with `QUANTIZATION_SWITCH = True` (the results in this repository are with quantization enabled).