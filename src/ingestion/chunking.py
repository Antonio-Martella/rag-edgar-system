import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import config

def create_chunks(text: str, ticker: str, report_type: str, fiscal_year: int = None) -> list:
    """
    Creates a text chunk enriched with specific context (ticker, report type, fiscal year) and section.
    Uses a text splitter that attempts to keep report sections intact,
    identifying section delimiters as [SECTION: ...].
    Each chunk contains a prefix with the context and the current section, followed by the chunk text itself.
    Carriage returns are retained to preserve table structure,
    but multiple spaces are reduced to a single space.
    """

    # Define a text splitter that prioritizes splitting at section boundaries, then by paragraphs, and finally by sentences.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.LEN_CHUNKS,  # <-- Adjust the chunk size as needed, ensuring it is large enough to capture meaningful sections but not too large to exceed model limits. 
        chunk_overlap = config.OVERLAP_CHUNKS,  # <-- Adjust the overlap to ensure that important context is retained across chunks, especially around section boundaries.
        separators=["\n[SECTION:", "\n\n", "\n", ". ", " "],  # <-- Prioritize splitting at section headers, then by paragraphs, and finally by sentences.
        is_separator_regex=False, 
    )
    
    # First, split the text into raw chunks based on the defined separators.
    raw_chunks = text_splitter.split_text(text)
    
    # Now, we will enrich each chunk with the context and section information.
    final_chunks = []
    current_section = "General Overview" 
    section_pattern = re.compile(r'\[SECTION:\s*(.*?)\]')

    # Iterate through the raw chunks and enrich them with context and section information.
    for i, chunk_text in enumerate(raw_chunks):

        # Check if the chunk contains a section header and update the current section accordingly.
        section_match = section_pattern.search(chunk_text)

        # If a section header is found, update the current section and remove the header from the chunk text.
        if section_match:
            current_section = section_match.group(1).strip()
            chunk_text = section_pattern.sub('', chunk_text).strip()
        
        # Reduce multiple spaces to a single space, but keep carriage returns to preserve table structure.
        chunk_text = re.sub(r' {2,}', ' ', chunk_text)
        
        # Create a context prefix for the chunk that includes the ticker, report type, fiscal year, and current section.
        context_prefix = f"[COMPANY: {ticker} | FY: {fiscal_year} | FORM: {report_type} | SECTION: {current_section}]\n"
        enriched_content = context_prefix + chunk_text.strip()
        
        # Append the enriched chunk to the final list of chunks, including metadata for later reference.
        final_chunks.append({
            "content": enriched_content,
            "metadata": {
                "ticker": ticker,
                "report_type": report_type,
                "year": fiscal_year, 
                "section": current_section, 
                "chunk_index": i
            }
        })
        
    return final_chunks