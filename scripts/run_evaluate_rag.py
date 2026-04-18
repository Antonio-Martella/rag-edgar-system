import sys
import json
import re
from pathlib import Path

# Path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.rag import RAGService

def ask_judge(generator: RAGService, question: str, expected: str, generated: str) -> str:
    """
    Use the LLM (Mistral) to compare the generated response with the expected one.
    """
    system_prompt = """You are an impartial financial auditor. Compare the "Generated Answer" against the "Expected Answer" for the given question.
    Determine if the Generated Answer is factually correct and contains the key numerical data from the Expected Answer.

    Rules:
    - Ignore formatting differences (e.g., "$7B" vs "7,000 million" is a PASS).
    - If the Generated Answer contains the correct numbers but extra text, it's a PASS.
    - If the numbers are different or the answer says "I don't know", it's a FAIL.
    - Answer ONLY with the word "PASS" or "FAIL"."""

    user_prompt = f"Question: {question}\nExpected Answer: {expected}\nGenerated Answer: {generated}\n\nVerdict (PASS/FAIL):"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Tokenize the prompt without adding generation tokens, and generate a short verdict (max_new_tokens=5) with low temperature to ensure deterministic output. We check if "PASS" is in the verdict to return PASS or FAIL.
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = generator.tokenizer(prompt, return_tensors="pt").to(generator.model.device)
    
    # Generate the verdict with max_new_tokens=5 and low temperature for deterministic output
    output = generator.model.generate(**inputs, max_new_tokens=5, temperature=0.01)
    verdict = generator.tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip().upper()
    
    return "PASS" if "PASS" in verdict else "FAIL"

def run_evaluation_suite() -> None:
    """
    Run the automated evaluation suite for RAG on the test queries in the 'evaluation' folder.
    For each test set (e.g., eval_tsla_10k_2023), it loads the corresponding data into the RAG, runs the queries, and uses the ask_judge function 
    to evaluate the generated answers against the expected ones. Finally, it saves a report JSON with the overall accuracy and details for each question.
    """

    print("🧪 Starting AUTOMATED RAG Evaluation Suite...")
    
    eval_dir = root_path / "evaluation"
    if not eval_dir.exists():
        print("❌ Error: Folder 'evaluation' not found.")
        return

    # Initialize the RAG service once
    rag_app = RAGService()

    # We scan all subfolders inside 'evaluation'
    for test_folder in eval_dir.iterdir():
        if not test_folder.is_dir() or not test_folder.name.startswith("eval_"):
            continue

        # Extract Ticker and Year from the folder name (e.g. eval_tsla_10k_2023)
        match = re.search(r"eval_([a-zA-Z]+)_10[kK]_(\d{4})", test_folder.name)
        if not match:
            continue

        ticker = match.group(1).upper()
        year = match.group(2)
        
        test_file = test_folder / f"test_queries_{year}.json"
        
        if not test_file.exists():
            print(f"⚠️ Skipping {ticker} {year}: file {test_file.name} not found.")
            continue

        print(f"\n" + "="*50)
        print(f"🏢 EVALUATION IN PROGRESS: {ticker} - {year}")
        print("="*50)

        # Load company data into RAG (FAISS + Chunks)
        try:
            rag_app.load_company_data(ticker, year, report_type="10-K")
        except Exception as e:
            print(f"❌ Unable to load data for {ticker} {year}: {e}")
            continue

        # Load test queries
        with open(test_file, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        final_results = []
        passes = 0

        for item in ground_truth:
            query = item["question"]
            expected = item["answer"] # Based on your test_queries.json
            
            print(f"\n❓ Question: {query}")
            
            # Generation: Turn off history, expanded retrieval network (40), final k=10
            generated = rag_app.ask(query, history=None, initial_k=20, final_k=10)
            
            # Evaluation
            verdict = ask_judge(rag_app.generator, query, expected, generated)
            
            if verdict == "PASS": 
                passes += 1

            print(f"🤖 Answer: {generated[:100]}...")
            print(f"⚖️ Verdict: {verdict}")

            # Save the detail with the exact structure you requested
            final_results.append({
                "question": query,
                "expected": expected,
                "generated": generated,
                "verdict": verdict
            })

        # Final calculation and JSON report creation
        accuracy_val = round((passes / len(ground_truth)) * 100, 1)
        
        report = {
            "overall_accuracy": f"{accuracy_val}%",
            "details": final_results
        }
        
        report_path = test_folder / f"eval_report_{year}.json"
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Completed {ticker} {year} | Accuracy: {accuracy_val}%")
        print(f"💾 Report saved in: {report_path}")

    print("\n🎉 ALL EVALUATIONS COMPLETED!")

if __name__ == "__main__":
    run_evaluation_suite()