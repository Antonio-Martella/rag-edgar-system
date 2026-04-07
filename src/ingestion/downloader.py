import os
from sec_edgar_downloader import Downloader
from dotenv import load_dotenv
# Importiamo il percorso centralizzato
from src.utils.config import RAW_PATH 

load_dotenv()

class EdgarDownloader:
    # Usiamo RAW_PATH come default, così punta sempre alla cartella giusta
    def __init__(self, download_path=RAW_PATH):
        self.user_agent = os.getenv("SEC_USER_AGENT")
        if not self.user_agent:
            raise ValueError("❌ SEC_USER_AGENT non trovato!")
        
        self.dl = Downloader("MyCompany", self.user_agent, download_path)

    def fetch_10k(self, ticker, limit=1):
        print(f"📥 Scaricando in: {RAW_PATH}")
        self.dl.get("10-K", ticker, limit=limit, after="2020-01-01")

if __name__ == "__main__":
    # Test rapido: prova a scaricare l'ultimo di Microsoft
    downloader = EdgarDownloader()
    downloader.fetch_10k("MSFT", limit=1)