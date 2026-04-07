import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMGenerator:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        """Inizializza l'LLM con configurazione 4-bit per risparmiare VRAM."""
        print(f"--- Caricamento LLM: {model_id} ---")
        
        # Configurazione per la compressione del modello (Quantizzazione)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto"
        )
        
        # Pipeline di generazione testo
        self.pipe = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

    def generate_answer(self, query, context_chunks, max_new_tokens=512):
        """Costruisce il prompt e genera la risposta finale."""
        context = "\n\n".join(context_chunks)
        
        # Prompt Template: Istruiamo l'IA a essere un analista serio
        prompt = f"""<s>[INST] You are a professional financial analyst. 
        Use the following context from SEC 10-K filings to answer the question. 
        If the answer is not in the context, say you don't know. Do not hallucinate.

        CONTEXT:
        {context}


        QUESTION:
        {query} [/INST]

        ANSWER:"""

        outputs = self.pipe(
            prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.1, 
            top_p=0.9
        )
        
        # Puliamo l'output per restituire solo la risposta
        generated_text = outputs[0]["generated_text"]
        return generated_text.split("ANSWER:")[-1].strip()