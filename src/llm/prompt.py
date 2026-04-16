def build_rag_messages(query: str, context_chunks: list, history: list = None) -> list:
    """
    Costruisce la lista dei messaggi strutturata (System, History, User)
    pronta per essere formattata dal tokenizer di qualsiasi modello.
    """
    context_items = []
    
    # 1. Costruiamo il blocco di contesto
    for c in context_chunks:
        if isinstance(c, dict):
            year = c.get('metadata', {}).get('year', 'N/A')
            ticker = c.get('metadata', {}).get('ticker', 'N/A')
            content = c.get('content', '')
            context_items.append(f"[Source: {ticker} FY{year}]\n{content}")
        else:
            context_items.append(str(c))

    context_text = "\n\n---\n\n".join(context_items)

    # 2. Definiamo le istruzioni di sistema
    system_prompt = f"""You are a strict financial auditor. Your task is to extract EXACT data from the provided context.
    - ONLY use the provided CONTEXT to answer.
    - TABULAR DATA: Read line-by-line. Pay EXTREME attention to the exact metric name. Do not confuse broader categories (e.g., "Total Revenues") with specific segments.
    - If the specific number for a specific year is not explicitly written in the CONTEXT, say "I cannot find this information in the documents."
    - NEVER perform mathematical operations (addition, subtraction, etc.) to guess or estimate missing values.
    - DO NOT invent logic to explain missing data.

    CONTEXT:
    {context_text}"""

    # 3. Assembliamo la conversazione
    messages = [{"role": "system", "content": system_prompt}]
    
    # Se c'è uno storico, lo inseriamo come botta e risposta
    if history:
        for user_q, bot_a in history:
            messages.append({"role": "user", "content": user_q})
            messages.append({"role": "assistant", "content": bot_a})
            
    # Aggiungiamo la domanda finale dell'utente
    messages.append({"role": "user", "content": query})
    
    return messages