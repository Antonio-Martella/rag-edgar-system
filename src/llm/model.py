import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils import config

def get_quantization_config() -> BitsAndBytesConfig:
    """
    Returns the quantization configuration for loading the LLM model based on the QUANTIZATION_SWITCH in the config.
    If QUANTIZATION_SWITCH is True, it returns a BitsAndBytesConfig for 8-bit quantization. 
    If False, it returns None, which means the model will be loaded in its original precision.
    """
    if config.QUANTIZATION_SWITCH:
        return BitsAndBytesConfig(
            load_in_8bit=True,                      # <-- Enable 8-bit loading
            bnb_8bit_compute_dtype=torch.float16,   # <-- Use float16 for computations (can also be float32 if you have more VRAM)
            bnb_8bit_use_double_quant=True,         # <-- Use double quantization for better accuracy (optional, but recommended)
            bnb_8bit_quant_type="nf4"               # <-- Use NormalFloat4 quantization (can also be "fp4" or "int8" depending on the model and your needs)
            )
    else:
        return None 

def setup_llm() -> None:
    """
    Set up the LLM model for use.
    """
    # Check if the LLM model is already downloaded and saved locally, if not, download and save it.
    if not config.LOCAL_LLM_PATH.exists():
        print(f"📥 Downloading LLM: {config.LLM_MODEL_ID} (This might take a while...)")

        # Load the tokenizer and model with the specified quantization configuration
        tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL_ID,
            token=config.HF_TOKEN
        )
        tokenizer.save_pretrained(str(config.LOCAL_LLM_PATH))

        # Load the model with the quantization configuration (if any)
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_ID,
            quantization_config=get_quantization_config(),
            torch_dtype=torch.float16,
            token=config.HF_TOKEN,
            device_map="auto" 
        )
        
        model.save_pretrained(str(config.LOCAL_LLM_PATH))
        print(f"✅ LLM saved in {config.LOCAL_LLM_PATH}")
    else:
        print("✔️ LLM already present locally.")

