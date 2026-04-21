def build_rag_messages(query: str, context_chunks: list, history: list = None) -> list:
    """
    Builds a structured message list (System, History, User)
    ready to be formatted by any model's tokenizer.
    """
    context_items = []
    
    # Build the context section by iterating over the provided chunks
    for c in context_chunks:
        if isinstance(c, dict):
            year = c.get('metadata', {}).get('year', 'N/A')
            ticker = c.get('metadata', {}).get('ticker', 'N/A')
            content = c.get('content', '')
            context_items.append(f"[Source: {ticker} FY{year}]\n{content}")
        else:
            context_items.append(str(c))

    context_text = "\n\n---\n\n".join(context_items)

    # We define a strict system prompt to guide the model's behavior in extracting data from the context
    system_prompt = f"""You are a strict financial auditor. Your task is to extract EXACT data from the provided context.
    - ONLY use the provided CONTEXT to answer.
    - TABULAR DATA: Read line-by-line. Pay EXTREME attention to the exact metric name. Do not confuse broader categories (e.g., "Total Revenues") with specific segments.
    - If the specific number for a specific year is not explicitly written in the CONTEXT, say "I cannot find this information in the documents."
    - NEVER perform mathematical operations (addition, subtraction, etc.) to guess or estimate missing values.
    - DO NOT invent logic to explain missing data.

    CONTEXT:
    {context_text}"""

    # Assembly the messages list starting with the system prompt
    messages = [{"role": "system", "content": system_prompt}]
    
    # If there is a history, we insert it as a series of user-assistant exchanges
    if history:
        for user_q, bot_a in history:
            messages.append({"role": "user", "content": user_q})
            messages.append({"role": "assistant", "content": bot_a})
            
    # Add the final user query
    messages.append({"role": "user", "content": query})
    
    return messages