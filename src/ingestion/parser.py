import re
from bs4 import BeautifulSoup

def clean_sec_text(raw_html):
    """
    Removes HTML tags, scripts, and CSS styles from a SEC document.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    
    # Rimuoviamo il superfluo
    for element in soup(["script", "style", "header", "footer"]):
        element.decompose()

    # Gestiamo le tabelle (spesso usate nei documenti SEC per dati finanziari)
    for table in soup.find_all("table"):
        table_data = []
        rows = table.find_all("tr")
        for row in rows:
            # Estraiamo le celle (td o th) e puliamo il testo
            cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
            # Filtriamo celle vuote che spesso sono usate per padding nei documenti SEC
            cols = [c for c in cols if c] 
            if cols:
                table_data.append(" | ".join(cols))
        
        # Sostituiamo la tabella HTML con una versione testuale strutturata
        structured_table = "\n" + "\n".join(table_data) + "\n"
        table.replace_with(structured_table)

    # Estrazione finale
    text = soup.get_text(separator="\n") # Usiamo \n per mantenere separazione tra paragrafi
    
    # Pulizia mirata (senza distruggere la struttura)
    text = re.sub(r'\b(us-gaap|srt|dei|iso4217):\S+', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
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