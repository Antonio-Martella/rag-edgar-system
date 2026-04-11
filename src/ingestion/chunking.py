from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_smart_chunks(text, ticker, report_type):
    """
    Prende il testo raffinato e lo divide in chunk con metadati.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=600,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    
    raw_chunks = text_splitter.split_text(text)
    
    final_chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        final_chunks.append({
            "content": chunk_text,
            "metadata": {
                "ticker": ticker,
                "report_type": report_type,
                "year": "2024", 
                "chunk_index": i
            }
        })
    return final_chunks