import os
from sec_edgar_downloader import Downloader
from dotenv import load_dotenv
from src.utils import config  

# Percorso assoluto della cartella principale del progetto (rag-edgar-system)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Crea il percorso esatto verso il file .env
env_path = os.path.join(project_root, '.env')

# Carica esplicitamente QUEL file
load_dotenv(dotenv_path=env_path)
print(f"📁 Loading environment variables from: {env_path}")

#load_dotenv()

class EdgarDownloader:
    def __init__(self, download_path=config.RAW_DATA_DIR):
        self.user_agent = os.getenv("SEC_USER_AGENT")
        if not self.user_agent:
            print(f"📁 Loading environment variables from: {env_path}")
            raise ValueError("❌ SEC_USER_AGENT not found in environment variables. Please set it in your .env file.")
        
        self.dl = Downloader("MyCompany", self.user_agent, download_path)

    def fetch_10k(self, format="10-K", ticker=None, limit=1, date_after="2023-01-01"):
        """
        Downloads the 10-K documents for a given ticker.
        """
        print(f"📥 Downloading to: {config.RAW_DATA_DIR}")
        self.dl.get(form=format, ticker_or_cik=ticker, limit=limit, after=date_after)
