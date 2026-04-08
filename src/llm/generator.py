import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils import config

class LLMGenerator:
    def __init__(self, model_id=config.LLM_MODEL):
        """
        Initializes the LLM with a 4-bit configuration to save VRAM.
        """

        print(f"--- Loading LLM: {model_id} ---")
        
        # Configuration for model compression to save VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Enables 4-bit quantization of the model to reduce VRAM usage
            bnb_4bit_use_double_quant=True,       # Enables double quantization to improve the precision of the quantized model
            bnb_4bit_quant_type="nf4",            # Use the NF4 quantization format, which is more efficient for language models
            bnb_4bit_compute_dtype=torch.bfloat16 # This is the format in which the computer does calculations. It's very fast on modern GPUs
        )

        # Load the tokenizer associated with the specified model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id) 

        # Load the model with the specified quantization configuration, assigning it automatically to the available GPU
        self.model = AutoModelForCausalLM.from_pretrained( 
            model_id,
            quantization_config=bnb_config, # Apply the quantization configuration to the model
            device_map="auto"
        )
        
        # Text generation pipeline
        self.pipe = pipeline(
            task="text-generation",  # Specifies that we want to use the template for text generation
            model=self.model,        # Use the loaded template with the quantization configuration
            tokenizer=self.tokenizer # Use the tokenizer associated with the model to convert input text to tokens and vice versa
        )

    def generate_answer(self, query, context_chunks, max_new_tokens=512):
        """
        Constructs the prompt and generates the final answer.
        """

        # Combine context chunks into a single text to be inserted into the prompt
        context = "\n\n".join(context_chunks)
        
        # Prompt Template: Let's train AI to be a serious analyst
        prompt = f"""<s>[INST] You are a professional financial analyst. 
        Use the following context from SEC 10-K filings to answer the question. 
        If the answer is not in the context, say you don't know. Do not hallucinate.

        CONTEXT:
        {context}


        QUESTION:
        {query} [/INST]

        ANSWER:"""

        # Generate the response
        outputs = self.pipe(
            text_inputs=prompt,            # The input text from which to generate the response
            max_new_tokens=max_new_tokens, # Limits the number of tokens generated for the response
            do_sample=True,                # Enables random sampling to make the responses more varied and natural
            temperature=0.1,               # Controls the creativity of the response (lower values make responses more deterministic)
            top_p=0.9                      # Controls the diversity of the response by limiting the choice of tokens to the most probable ones (lower values make responses more conservative)
        )
        
        # Clean the output to return only the answer
        generated_text = outputs[0]["generated_text"]

        # Extract the final answer from the generated text, removing any parts of the prompt and whitespace
        return generated_text.split("ANSWER:")[-1].strip()