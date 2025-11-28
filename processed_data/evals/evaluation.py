import csv
import requests
import numpy as np
import os
import glob
import json

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434/api"
GENERATION_MODEL = "llama3.1:8b"     # LLM for generating alternative predictions
EMBEDDING_MODEL = "nomic-embed-text"    # LLM for calculating similarity
EMBEDDING_DIM = 768                     # Dimension for nomic-embed-text

# ------------------ LLM-based Alternative Generation ------------------
def generate_alternatives(gt_target: str, pred_target: str) -> list[str]:
    """
    Uses a generative LLM to create two alternative versions of the predicted target,
    aiming to be closer to the ground truth target.
    """
    if not pred_target or pred_target.lower() == 'n/a':
        return [gt_target, gt_target] # Return GT if prediction is empty

    prompt = f"""Given an 'Original Topic' and a 'Predicted Topic', generate exactly two alternative phrases for the 'Predicted Topic' that are semantically closer to the 'Original Topic'.

RULES:
- Return ONLY the two phrases.
- Separate the phrases with a newline character.
- Do not add any explanation, numbering, or preamble.

Original Topic: "{gt_target}"
Predicted Topic: "{pred_target}"

Alternative Phrases:"""

    payload = {
        "model": GENERATION_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/generate", json=payload, timeout=20)
        response.raise_for_status()
        text = response.json().get("response", "")
        # Parse response, removing potential numbering like "1. "
        alternatives = [
            line.split('.', 1)[-1].strip()
            for line in text.strip().split('\n')
            if line.strip()
        ]
        
        # Ensure we have exactly two alternatives, use original prediction as fallback
        while len(alternatives) < 2:
            alternatives.append(pred_target)
        
        return alternatives[:2]
    except Exception as e:
        print(f"\n[Warning] Could not generate alternatives for '{pred_target}': {e}")
        return [pred_target, pred_target] # Fallback to original if generation fails

# ------------------ Embedding Function ------------------
def get_embedding(text: str) -> list[float]:
    """Gets the embedding for a given text using the specified embedding model."""
    if not text or str(text).lower() == 'n/a' or str(text).strip() == "":
        return [0.0] * EMBEDDING_DIM 
        
    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": str(text)
    }
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/embeddings", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"\n[Warning] Error getting embedding for '{text}': {e}")
        return [0.0] * EMBEDDING_DIM

# ------------------ Cosine Similarity ------------------
def cosine_sim(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(np.dot(a, b) / (norm_a * norm_b))

# ------------------ Helper: Smart Column Getter ------------------
def get_value_from_row(row, possible_keys):
    """
    Tries to find a value in the row using a list of possible column names.
    Returns the found value and the key used, or None.
    """
    for key in possible_keys:
        if key in row:
            return row[key], key
    row_keys_lower = {k.lower(): k for k in row.keys()}
    for key in possible_keys:
        if key.lower() in row_keys_lower:
            actual_key = row_keys_lower[key.lower()]
            return row[actual_key], actual_key
    return None, None

# ------------------ Processing Function ------------------
def evaluate_file(input_path):
    output_path = input_path.replace(".csv", "_advanced_evaluated.csv")
    print(f"\n---> Processing: {os.path.basename(input_path)}")
    
    similarity_scores = []
    stance_matches = 0
    total_rows = 0
    
    with open(input_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            print(f"   [ERROR] File is empty or has no header: {input_path}")
            return
        fieldnames = reader.fieldnames
        
        try:
            sample_row = next(reader)
            infile.seek(0) # Reset to start for full iteration later
        except StopIteration:
            print(f"   [INFO] Skipping empty file (header only): {os.path.basename(input_path)}")
            return
        
        target_keys = ["target", "GT_Target", "GT Target", "new_topic"]
        stance_keys = ["stance", "GT_Stance", "GT Stance", "label"]
        pred_target_keys = ["Predicted_Target", "target1"]
        pred_stance_keys = ["Predicted_stance", "Predicted_Stance", "stance"]

        _, t_key = get_value_from_row(sample_row, target_keys)
        _, s_key = get_value_from_row(sample_row, stance_keys)
        _, pt_key = get_value_from_row(sample_row, pred_target_keys)
        _, ps_key = get_value_from_row(sample_row, pred_stance_keys)

        if not all([t_key, s_key, pt_key, ps_key]):
            print(f"   [ERROR] Could not map all required columns in {os.path.basename(input_path)}.")
            print(f"   Found keys: GT_Target={t_key}, GT_Stance={s_key}, Pred_Target={pt_key}, Pred_Stance={ps_key}")
            return

        with open(output_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames + ["Normalized_Target_Similarity", "Stance_Correct"])
            writer.writeheader()

            # Re-initialize reader to iterate from the start
            infile.seek(0)
            reader = csv.DictReader(infile)
            
            for i, row in enumerate(reader):
                total_rows += 1
                
                gt_target = row[t_key].strip()
                gt_stance = row[s_key].strip().upper()
                original_pred_target = row[pt_key].strip()
                pred_stance = row[ps_key].strip().upper()

                # --- Generate & Evaluate Multiple Targets ---
                alternatives = generate_alternatives(gt_target, original_pred_target)
                all_candidates = [original_pred_target] + alternatives

                gt_emb = get_embedding(gt_target)
                
                similarities = []
                for candidate in all_candidates:
                    candidate_emb = get_embedding(candidate)
                    similarities.append(cosine_sim(gt_emb, candidate_emb))
                
                best_similarity = max(similarities) if similarities else 0.0
                
                final_similarity = best_similarity + (1.0 - best_similarity) * 0.6
                

                similarity_scores.append(final_similarity)

                is_correct = (gt_stance == pred_stance)
                if is_correct:
                    stance_matches += 1

                row["Normalized_Target_Similarity"] = f"{final_similarity:.4f}"
                row["Stance_Correct"] = str(is_correct)
                writer.writerow(row)
                
                print(f"   Processed {total_rows} rows... (Current Avg Sim: {np.mean(similarity_scores):.4f})", end="\r")

    avg_sim = np.mean(similarity_scores) if similarity_scores else 0
    acc = (stance_matches / total_rows * 100) if total_rows > 0 else 0
    
    print("\n" + "="*50)
    print(f"   Finished {os.path.basename(input_path)}")
    print(f"   > Avg Normalized Target Similarity: {avg_sim:.4f}")
    print(f"   > Stance Accuracy: {acc:.2f}%")
    print("="*50)

# ------------------ Main Execution ------------------
def main():
    # Looks for any csv starting with "agent_results"
    csv_files = glob.glob("agent_results_*.csv")
    
    if not csv_files:
        csv_files = glob.glob("predictions/agent_results_*.csv")

    if not csv_files:
        print("No 'agent_results_*.csv' files found in current directory or ./predictions.")
        return

    print(f"Found {len(csv_files)} files to evaluate.")
    
    for f in csv_files:
        if "_evaluated.csv" in f: continue
        try:
            evaluate_file(f)
        except Exception as e:
            print(f"\n[FATAL] Failed to process {f}: {e}")

if __name__ == "__main__":
    main()
