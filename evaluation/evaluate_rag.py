import sys
import json
from pathlib import Path

# Aggiungiamo la root del progetto al path per importare src
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.retrieval.retriever import Retriever
from src.llm.generator import LLMGenerator
from src.utils import config

def run_evaluation():

    print("🧪 Starting RAG Evaluation Process...")
    
    # Caricamento Verità
    gt_path = root_path / "evaluation" / "test_queries.json"
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    # Inizializzazione RAG (usiamo Tesla come test)
    ticker = "tsla"
    paths = config.get_paths(ticker, "10-K")
    
    retriever = Retriever(paths["index"], paths["chunks"])
    generator = LLMGenerator()
    
    results = []

    # Ciclo di test
    for item in ground_truth:
        query = item["question"]
        print(f"\n❓ Testing: {query}")
        
        # Recupero e Generazione
        chunks = retriever.search(query, k=20)
        answer = generator.generate_answer(query, chunks)
        
        # Salvataggio risultato per analisi
        res = {
            "question": query,
            "expected": item["answer"],
            "generated": answer,
            "passed": item["answer"].lower() in answer.lower() # Semplice controllo testuale
        }
        results.append(res)
        
        print(f"✅ Expected: {res['expected']}")
        print(f"🤖 Generated: {res['generated']}")
        print(f"⚖️ Status: {'PASS' if res['passed'] else 'FAIL'}")

    # Salvataggio Report Finale
    report_path = root_path / "evaluation" / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n📊 Evaluation complete. Report saved in {report_path}")

if __name__ == "__main__":
    run_evaluation()