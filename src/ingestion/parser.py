import re
from bs4 import BeautifulSoup

def clean_sec_text(raw_html):
    """
    Removes HTML tags, scripts, and CSS styles from a SEC document.
    """

    # We use lxml as parser because it is very fast with large files (6MB+)
    soup = BeautifulSoup(raw_html, "lxml")
    # We remove elements that do not contain text useful for reading
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    # We extract the text by separating the blocks with a space
    text = soup.get_text(separator=" ")
    # Initial cleanup: remove multiple spaces and whitespace at edges
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def refined_clean(text):
    """
    Removes SEC-specific noise (XBRL tags and technical IDs).
    """

    # Removes technical tags like us-gaap, srt, dei, etc.
    text = re.sub(r'\b(us-gaap|srt|dei|iso4217):\S+', '', text)
    # Removes long numeric sequences (often filing or CIK IDs)
    # Example: 0001318605
    text = re.sub(r'\b\d{10}\b', '', text) 
    # Final cleanup of resulting whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text