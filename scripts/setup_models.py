import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils import config

def setup():
    print("🚀 Starting local environment setup...")

    # Download and Save Embedding Model
    if not config.LOCAL_EMBEDDING_PATH.exists():
        print(f"📥 Downloading Embedding: {config.EMBEDDING_MODEL_ID}...")
        model = SentenceTransformer(config.EMBEDDING_MODEL_ID)
        model.save(str(config.LOCAL_EMBEDDING_PATH))
        print(f"✅ Embedding saved in {config.LOCAL_EMBEDDING_PATH}")
    else:
        print("✔️ Embedding already present locally.")

    # Download and Save LLM (Tokenizer and Model)
    if not config.LOCAL_LLM_PATH.exists():
        print(f"📥 Downloading LLM: {config.LLM_MODEL_ID} (This might take a while...)")
        
        # Download the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
        tokenizer.save_pretrained(str(config.LOCAL_LLM_PATH))
        
        # Download the model
        # NOTE: To save it already quantized, a specific procedure is needed,
        # here we save the base weights. Quantization will happen at loading.
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_ID,
            torch_dtype=torch.float16, # Standard compressed format
            device_map="cpu" # Download the model
        )
        model.save_pretrained(str(config.LOCAL_LLM_PATH))
        print(f"✅ LLM saved in {config.LOCAL_LLM_PATH}")
    else:
        print("✔️ LLM already present locally.")

if __name__ == "__main__":
    setup()