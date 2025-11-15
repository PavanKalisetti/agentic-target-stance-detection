def calculate_average_best_similarity(csv_path):
    """
    Calculates the average best cosine similarity score from the CSV file.
    Assumes each row contains sim1, sim2, and sim3 columns.
    """

    best_sims = []
    import csv

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                sim1 = float(row["sim1"])
                sim2 = float(row["sim2"])
                sim3 = float(row["sim3"])

                best_sim = max(sim1, sim2, sim3)
                best_sims.append(best_sim)

            except ValueError:
                # Skip rows with invalid similarity values
                continue

    if len(best_sims) == 0:
        print("No similarity values found.")
        return 0.0

    average_best_sim = sum(best_sims) / len(best_sims)

    print(f"\nAverage Best Cosine Similarity: {average_best_sim:.4f}")
    return average_best_sim



calculate_average_best_similarity("/home/rgukt/Documents/major project/major-project/vast_ex_stance_with_similarity_results.csv")