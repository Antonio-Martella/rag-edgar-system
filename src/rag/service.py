import os
from src.retrieval import Retriever
from src.llm import LLMGenerator
from src.utils import config

class RAGService:
    def __init__(self):
        print("🤖 Brain Initialization (RAG Service)...")
        self.generator = LLMGenerator()
        self.retriever = None
        self.current_ticker = None
        self.current_year = None

    def load_company_data(self, ticker: str, year: str, report_type: str) -> None:
        """
        Loads the FAISS index and chunks for a specific company and fiscal year.
        This method must be called before asking any questions about the company.
        """
        # Get the paths for the index and chunks based on the ticker, report type, and year
        paths = config.get_paths(ticker, report_type, year)

        # Check if the required files exist before trying to load them, and raise an error if they are missing.
        if not os.path.exists(paths["index"]) or not os.path.exists(paths["chunks"]):
            raise FileNotFoundError(f"❌ No data found for {ticker} ({year}). Please run indexing first (run_indexing.py).")
        
        # Load the retriever with the specified index and chunks, which will be used for all subsequent questions about this company until a new company is loaded.
        self.retriever = Retriever(paths["index"], paths["chunks"])
        self.current_ticker = ticker
        self.current_year = year
        print(f"✅ Data loaded for {ticker} ({year}). The analyst is ready.")

    def ask(self, query: str, history: list = None, initial_k: int = 20, final_k: int = 5) -> str:
        """
        This is the main method to ask a question about the currently loaded company.
        It performs the following steps:
        1. Checks if the retriever is loaded (i.e., if company data is loaded).
        2. Performs semantic search and reranking to get the top relevant chunks.
        3. Passes the query, retrieved chunks, and conversation history to the LLM generator to get the final answer.
        4. Returns the generated answer.
        """
        if not self.retriever:
            return "❌ Error: You must upload company data first."
            
        # Search for relevant context chunks using the retriever, which internally performs both the FAISS search and the reranking to return the top final_k chunks.
        top_chunks = self.retriever.search(query, initial_k=initial_k, final_k=final_k)
        
        if not top_chunks:
            return "I did not find any relevant information in the documents for this question."
            
        # Generation of the final answer using the LLM, passing the retrieved context and the conversation history
        answer = self.generator.generate_answer(query, top_chunks, history=history)
        
        return answer