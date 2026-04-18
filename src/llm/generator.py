import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import config
from src.llm.model import get_quantization_config
from src.llm.prompt import build_rag_messages

class LLMGenerator:
    def __init__(self, model_path=config.LOCAL_LLM_PATH):
        """
        Initializes the LLMGenerator by loading the tokenizer and model from the specified local path.
        It checks if the model exists locally and raises an error if not, prompting the user to run the setup_models.py script to download it. 
        The model is loaded with the appropriate quantization configuration based on the QUANTIZATION_SWITCH in the config.
        """
        print(f"--- Loading Local LLM: {model_path} ---")

        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ Model not found in {model_path}. "
                "Please make sure you have successfully executed 'python3 scripts/run_setupmodels.py'."
            )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) 

        # Load the model with the specified quantization configuration
        self.model = AutoModelForCausalLM.from_pretrained( 
            model_path,
            quantization_config=get_quantization_config(), 
            device_map="auto"
        )

    def generate_answer(self, query, context_chunks, history=None, max_new_tokens=1024):
        """
        Constructs the prompt dynamically and generates the final answer.
        """
        # Build the structured messages (System, History, User) ready to be formatted by the tokenizer of any model.
        messages = build_rag_messages(query, context_chunks, history)

        # Format the messages into a single prompt string using the tokenizer's chat template, without tokenizing yet, and adding the generation prompt at the end.
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # <-- This will add the appropriate generation prompt (e.g., "Assistant:") at the end of the formatted messages, which is important for guiding the model to generate the answer in the correct format.
        )

        # Tokenize the prompt and move it to the GPU
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate the response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1, 
                top_p=0.9,
                repetition_penalty=1.1,
            )
        
        # Extract only the generated tokens by removing the input prompt length
        generated_tokens = output_ids[0][len(inputs["input_ids"][0]):]

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()