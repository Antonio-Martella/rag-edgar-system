import re
from bs4 import BeautifulSoup

def clean_sec_text(raw_html):
    """
    Rimuove i tag HTML, gli script e gli stili CSS da un documento SEC.
    """
    # Usiamo lxml come parser perché è molto veloce con file grandi (6MB+)
    soup = BeautifulSoup(raw_html, "lxml")
    
    # Rimuoviamo elementi che non contengono testo utile alla lettura
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Estraiamo il testo separando i blocchi con uno spazio
    text = soup.get_text(separator=" ")

    # Pulizia iniziale: rimuovi spazi multipli e whitespace ai bordi
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def refined_clean(text):
    """
    Rimuove il rumore specifico dei documenti SEC (tag XBRL e ID tecnici).
    """
    # Rimuove tag tecnici come us-gaap, srt, dei, ecc.
    text = re.sub(r'\b(us-gaap|srt|dei|iso4217):\S+', '', text)
    
    # Rimuove sequenze numeriche lunghe (spesso ID di filing o CIK isolati)
    # Esempio: 0001318605
    text = re.sub(r'\b\d{10}\b', '', text) 
    
    # Pulizia finale degli spazi bianchi risultanti
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text