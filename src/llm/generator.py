import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils import config

class LLMGenerator:
    def __init__(self, model_path=config.LOCAL_LLM_PATH):
        """
        Initializes the LLM with a 4-bit configuration to save VRAM.
        """
        
        print(f"--- Loading Local LLM: {model_path} ---")

        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ Model not found in {model_path}. "
                "Please make sure you have successfully executed 'python3 scripts/setup_models.py'."
            )
        
        # Configuration for model compression to save VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Enables 4-bit quantization of the model to reduce VRAM usage
            bnb_4bit_use_double_quant=True,       # Enables double quantization to improve the precision of the quantized model
            bnb_4bit_quant_type="nf4",            # Use the NF4 quantization format, which is more efficient for language models
            bnb_4bit_compute_dtype=torch.bfloat16 # This is the format in which the computer does calculations. It's very fast on modern GPUs
        )

        # Load the tokenizer associated with the specified model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) 

        # Load the model with the specified quantization configuration, assigning it automatically to the available GPU
        self.model = AutoModelForCausalLM.from_pretrained( 
            model_path,
            quantization_config=bnb_config, # Apply the quantization configuration to the model
            device_map="auto"
        )

    def generate_answer(self, query, context_chunks, history=None, max_new_tokens=1024):
        """
        Constructs the prompt and generates the final answer.
        """

        # Combine context chunks into a single text to be inserted into the prompt
        context = "\n\n".join(context_chunks)

        # Prepare the conversation history in a readable format for the prompt. We will format it as a simple dialogue
        history_str = ""
        if history:
            for user_q, bot_a in history:
                history_str += f"User: {user_q}\nAssistant: {bot_a}\n"
        
        # Prompt Template: Let's train AI to be a serious analyst
        prompt = f"""<s>[INST] <<SYS>>
        You are a strict financial auditor. Your task is to extract EXACT data from the provided context.
        - ONLY use the provided CONTEXT to answer.
        - If the specific number for a specific year is not explicitly written in the CONTEXT, say "I cannot find this information in the documents."
        - NEVER perform mathematical operations (addition, subtraction, etc.) to guess or estimate missing values.
        - DO NOT invent logic to explain missing data.
        <</SYS>>

        CONVERSATION HISTORY:
        {history_str if history_str else "No previous history."}

        CONTEXT:
        {context}


        QUESTION:
        {query} [/INST]

        ANSWER:"""

        # Tokenize the prompt and move it to the same device as the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate the response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,                                  # The input text from which to generate the response
                max_new_tokens=max_new_tokens,             # Limits the number of tokens generated for the response
                pad_token_id=self.tokenizer.eos_token_id,  # Ensures that the model knows when to stop generating text
                do_sample=True,                            # Enables random sampling to make the responses more varied and natural
                temperature=0.1,                           # Controls the creativity of the response (lower values make responses more deterministic)
                top_p=0.9,                                 # Controls the diversity of the response by limiting the choice of tokens to the most probable ones (lower values make responses more conservative)
                num_return_sequences=1,                    # Specifies that we want to generate only one response for each input prompt
                repetition_penalty=1.1,                    # Penalizes the model for repeating the same tokens, which helps to reduce redundan
            )
        
        # Extract only the generated tokens (the response) by removing the input prompt tokens from the output
        generated_tokens = output_ids[0][len(inputs["input_ids"][0]):]

        # Decode the generated tokens back into text, removing any special tokens and extra whitespace
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()