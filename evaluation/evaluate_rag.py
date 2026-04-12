import sys
import json
import re
from pathlib import Path

# Aggiungiamo la root del progetto al path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.retrieval.retriever import Retriever
from src.retrieval.reranker import RAGReranker
from src.llm.generator import LLMGenerator
from src.utils import config

def ask_judge(generator, question, expected, generated):
    """
    Usa l'LLM per confrontare la risposta generata con quella attesa.
    """
    judge_prompt = f"""<s>[INST] <<SYS>>
                        You are an impartial financial auditor. Compare the "Generated Answer" against the "Expected Answer" for the given question.
                        Determine if the Generated Answer is factually correct and contains the key numerical data from the Expected Answer.

                        Rules:
                        - Ignore formatting differences (e.g., "$7B" vs "7,000 million" is a PASS).
                        - If the Generated Answer contains the correct numbers but extra text, it's a PASS.
                        - If the numbers are different or the answer says "I don't know", it's a FAIL.
                        - Answer ONLY with the word "PASS" or "FAIL".
                        <</SYS>>

                        Question: {question}
                        Expected Answer: {expected}
                        Generated Answer: {generated}

                        Verdict (PASS/FAIL): [/INST]"""

    # Usiamo il metodo di generazione del nostro oggetto generator
    # Nota: usiamo pochi token perché ci serve solo una parola
    inputs = generator.tokenizer(judge_prompt, return_tensors="pt").to(generator.model.device)
    output = generator.model.generate(**inputs, max_new_tokens=5, temperature=0.01)
    verdict = generator.tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip().upper()
    
    # Pulizia minima del verdetto
    return "PASS" if "PASS" in verdict else "FAIL"

def run_evaluation():
    print("🧪 Starting RAG Evaluation with LLM-as-a-Judge...")
    
    gt_path = root_path / "evaluation" / "test_queries.json"
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    ticker = "tsla"
    paths = config.get_paths(ticker, "10-K")
    retriever = Retriever(paths["index"], paths["chunks"])
    reranker = RAGReranker() 
    generator = LLMGenerator()
    
    final_results = []
    passes = 0

    for item in ground_truth:
        query = item["question"]
        expected = item["answer"] 
        
        print(f"\n❓ Question: {query}")
        
        # 1. Retrieval & Generation 
        chunks = retriever.search(query, k=40) 
        refined_chunks = reranker.rerank(query, chunks, top_n=40)
        generated = generator.generate_answer(query, refined_chunks[:15])
        
        # 2. Judging
        verdict = ask_judge(generator, query, expected, generated)
        
        is_pass = (verdict == "PASS")
        if is_pass: passes += 1

        print(f"🤖 Bot Answer: {generated[:100]}...")
        print(f"⚖️ Judge Verdict: {verdict}")

        final_results.append({
            "question": query,
            "expected": expected,
            "generated": generated,
            "verdict": verdict
        })

    # 3. Report
    accuracy = (passes / len(ground_truth)) * 100
    report = {
        "overall_accuracy": f"{accuracy}%",
        "details": final_results
    }
    
    with open(root_path / "evaluation" / "eval_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    print(f"\n📊 Evaluation complete! Accuracy: {accuracy}%")

if __name__ == "__main__":
    run_evaluation()