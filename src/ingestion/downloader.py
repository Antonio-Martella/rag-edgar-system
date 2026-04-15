import os
import re
from sec_edgar_downloader import Downloader
from dotenv import load_dotenv
from src.utils import config

# Define the project root and load environment variables from the .env file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
env_path = os.path.join(project_root, '.env')

# Load environment variables from the .env file
load_dotenv(dotenv_path=env_path)

class EdgarDownloader:
    """
    A class for managing document downloads from the SEC EDGAR platform.

    Implements advanced search logic that goes beyond simple downloads,
    verifying internal file metadata (CONFORMED PERIOD OF REPORT) to
    ensure that the downloaded fiscal year exactly matches the requested one,
    avoiding ambiguities in filing dates for different companies.
    """
    def __init__(self, download_path = config.RAW_DATA_DIR):
        """
        Initializes the SEC downloader with the destination path and user agent.

        Args:
            download_path (Path, optional): Local path where raw files will be saved.
                                            Default set in config.RAW_DATA_DIR.

        Raises:
            ValueError: If the environment variable 'SEC_USER_AGENT' is not defined in the .env file.
        """
        # Set the download path and load the user agent from the environment variables
        self.download_path = download_path
        self.user_agent = os.getenv("SEC_USER_AGENT")
        
        # Verify that the user agent has been loaded correctly
        if not self.user_agent:
            print(f"📁 Unable to load from: {env_path}")
            raise ValueError("❌ SEC_USER_AGENT not found in environment variables. Please set it in your .env file.")
        
        # Initialize the SEC downloader with the specified user agent and download path
        self.dl = Downloader("Personal", self.user_agent, self.download_path)


    def fetch_and_read(self, ticker: str, target_year: str, report_type: str = "10-K") -> str:
        """
        Downloads the requested document and extracts its text content filtered by fiscal year.

        The function downloads the latest available report before a calculated due date,
        then recursively scans local folders to find the 'full-submission.txt' file
        that contains the fiscal year (target_year) in its metadata.

        Args:
            ticker (str): The company's ticker symbol (e.g., 'TSLA', 'AAPL').
            target_year (str/int): The fiscal year of interest.
            report_type (str, optional): The type of SEC document. Default: "10-K".

        Returns:
            str: The full text content (HTML/Raw) of the report if found, otherwise: None.
        """
        # Calculate the due date for the report (using "before" to ensure we get the right report)
        filing_deadline = f"{int(target_year) + 1}-04-01"
        print(f"📥 Download {report_type} for fiscal year {target_year}...")
        
        # Download the latest available report before it expires
        self.dl.get(report_type, ticker, limit=1, before=filing_deadline)

        # After downloading, we look for the "full-submission.txt" file which contains the official SEC metadata, 
        # including the fiscal year.
        try:
            full_path = None
            # 2. Recursively browse the download folder to find the "full-submission.txt" file
            for temp_path in self.download_path.rglob("full-submission.txt"):
                # We check if the file belongs to the correct ticker and if it contains the target fiscal year in its metadata.
                if ticker.upper() in str(temp_path).upper() or temp_path.parent.parent.name.isdigit():
                    # If the file is relevant, we read its header to find the "CONFORMED PERIOD OF REPORT" field, which indicates the fiscal year of the report.
                    with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                        header = f.read(5000)
                        # We use a regular expression to extract the fiscal year from the "CONFORMED PERIOD OF REPORT" field in the header.
                        match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{4})", header)
                        # If the fiscal year matches the target year, we select this file as the correct one to read.
                        if match and match.group(1) == str(target_year):
                            full_path = temp_path
                            break
            
            # If we found the correct file, we read and return its content. Otherwise, we print an error message.
            if full_path and full_path.exists():
                print(f"✅ Selected the exact file: {full_path}")
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()
            # If no file was found that matches the fiscal year, we return None and print a message.
            else:
                print(f"❌ No report found for fiscal year {target_year} (looking for {full_path})")
                return None
        # For any error during the file search and reading process
        except Exception as e:
            print(f"❌ Error finding or reading the file: {e}")
            return None
