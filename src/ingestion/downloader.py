import os
from sec_edgar_downloader import Downloader
from dotenv import load_dotenv
from src.utils import config  

load_dotenv()

class EdgarDownloader:
    def __init__(self, download_path=config.RAW_DATA_DIR):
        self.user_agent = os.getenv("SEC_USER_AGENT")
        if not self.user_agent:
            raise ValueError("❌ SEC_USER_AGENT non trovato!")
        
        self.dl = Downloader("MyCompany", self.user_agent, download_path)

    def fetch_10k(self, ticker, limit=1, date_after="2023-01-01"):
        """
        Scarica i documenti 10-K per un dato ticker.
        """
        print(f"📥 Scaricando in: {config.RAW_DATA_DIR}")
        self.dl.get("10-K", ticker, limit=limit, after=date_after)
