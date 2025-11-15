import csv
import requests
import numpy as np

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "llama3.1:8b"

# ------------------ Embedding Function ------------------
def get_embedding(text):
    payload = {
        "model": MODEL_NAME,
        "prompt": text
    }
    response = requests.post(OLLAMA_EMBED_URL, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]

# ------------------ Cosine Similarity ------------------
def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------ Main Script ------------------
input_csv = "vast_filtered_ex_simple_agent_pred.csv"
output_csv = "vast_ex_stance_with_similarity_results.csv"

correct = 0
total = 0

with open(input_csv, "r", encoding="utf-8") as infile, \
     open(output_csv, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ["sim1", "sim2", "sim3", "best_target"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        gt = row["new_topic"]
        t1 = row["target1"]
        t2 = row["target2"]
        t3 = row["target3"]

        # 1. Get embeddings
        gt_emb = get_embedding(gt)
        t1_emb = get_embedding(t1)
        t2_emb = get_embedding(t2)
        t3_emb = get_embedding(t3)

        # 2. Cosine Similarities
        sim1 = cosine_sim(gt_emb, t1_emb)
        sim2 = cosine_sim(gt_emb, t2_emb)
        sim3 = cosine_sim(gt_emb, t3_emb)

        # 3. Determine best target
        sims = [sim1, sim2, sim3]
        best_index = np.argmax(sims)
        best_target = [t1, t2, t3][best_index]

        # 4. Track accuracy
        total += 1
        if best_target.strip().lower() == gt.strip().lower():
            correct += 1

        # 5. Write row with similarity values
        row["sim1"] = sim1
        row["sim2"] = sim2
        row["sim3"] = sim3
        row["best_target"] = best_target

        writer.writerow(row)

# ------------------ Final Accuracy ------------------
accuracy = correct / total
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
