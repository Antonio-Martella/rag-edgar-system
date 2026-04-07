import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMGenerator:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        """Inizializza l'LLM con configurazione 4-bit per risparmiare VRAM."""
        print(f"--- Caricamento LLM: {model_id} ---")
        
        # Configurazione per la compressione del modello per risparmiare VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Abilita la quantizzazione a 4 bit del modello per ridurre l'uso di VRAM
            bnb_4bit_use_double_quant=True,       # Abilita la doppia quantizzazione per migliorare la precisione del modello quantizzato
            bnb_4bit_quant_type="nf4",            # Utilizza il formato di quantizzazione NF4, che è più efficiente per i modelli di linguaggio
            bnb_4bit_compute_dtype=torch.bfloat16 # È il formato con cui il computer fa i calcoli. È molto veloce sulle GPU moderne.
        )

        # Carica il tokenizer associato al modello specificato
        self.tokenizer = AutoTokenizer.from_pretrained(model_id) 

        # Carica il modello con la configurazione di quantizzazione specificata, assegnandolo automaticamente alla GPU disponibile
        self.model = AutoModelForCausalLM.from_pretrained( 
            model_id,
            quantization_config=bnb_config, # Applica la configurazione di quantizzazione al modello
            device_map="auto"
        )
        
        # Pipeline di generazione testo
        self.pipe = pipeline(
            task="text-generation",  # Specifica che vogliamo utilizzare il modello per la generazione di testo
            model=self.model,        # Utilizza il modello caricato con la configurazione di quantizzazione
            tokenizer=self.tokenizer # Utilizza il tokenizer associato al modello per convertire il testo in input in token e viceversa
        )

    def generate_answer(self, query, context_chunks, max_new_tokens=512):
        """Costruisce il prompt e genera la risposta finale."""
        # Combina i chunk di contesto in un unico testo da inserire nel prompt
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

        # Genera la risposta 
        outputs = self.pipe(
            prompt,                        # Il testo di input da cui generare la risposta
            max_new_tokens=max_new_tokens, # Limita il numero di token generati per la risposta
            do_sample=True,                # Abilita la generazione casuale per rendere le risposte più varie e naturali
            temperature=0.1,               # Controlla la creatività della risposta (valori più bassi rendono le risposte più deterministiche)
            top_p=0.9                      # 
        )
        
        # Puliamo l'output per restituire solo la risposta
        generated_text = outputs[0]["generated_text"]

        # Estrae la risposta finale dal testo generato, rimuovendo eventuali parti del prompt e spazi bianchi
        return generated_text.split("ANSWER:")[-1].strip()