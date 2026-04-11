import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils import config

def setup():
    print("🚀 Starting local environment setup...")

    # Download and Save Embedding Model
    if not config.LOCAL_EMBEDDING_PATH.exists():
        print(f"📥 Downloading Embedding: {config.EMBEDDING_MODEL_ID}...")
        model = SentenceTransformer(config.EMBEDDING_MODEL_ID, trust_remote_code=True)
        model.save(str(config.LOCAL_EMBEDDING_PATH))
        print(f"✅ Embedding saved in {config.LOCAL_EMBEDDING_PATH}")
    else:
        print("✔️ Embedding already present locally.")

    # Download and Save Reranker Model
    if not config.LOCAL_RERANKER_PATH.exists():
        print(f"📥 Downloading Reranker: {config.RERANKER_MODEL_ID}...")
        reranker = CrossEncoder(config.RERANKER_MODEL_ID)
        reranker.save(str(config.LOCAL_RERANKER_PATH))
        print(f"✅ Reranker saved in {config.LOCAL_RERANKER_PATH}")
    else:
        print("✔️ Reranker already present locally.")

    # Download and Save LLM (Tokenizer and Model)
    if not config.LOCAL_LLM_PATH.exists():
        print(f"📥 Downloading LLM: {config.LLM_MODEL_ID} (This might take a while...)")
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
        tokenizer.save_pretrained(str(config.LOCAL_LLM_PATH))
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cpu" 
        )
        model.save_pretrained(str(config.LOCAL_LLM_PATH))
        print(f"✅ LLM saved in {config.LOCAL_LLM_PATH}")
    else:
        print("✔️ LLM already present locally.")

if __name__ == "__main__":
    setup()