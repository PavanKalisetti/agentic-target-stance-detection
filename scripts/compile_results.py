
import os
import json
import pandas as pd

# --- Configuration ---
# Directories are relative to the project root where this script is expected to be run from
AGENT_RUNS_DIR = "agent_runs"
OUTPUT_CSV_PATH = "agent_runs_summary.csv"

def compile_agent_runs():
    """
    Reads all JSON files from the agent_runs directory, extracts key information,
    and saves it into a single CSV file.
    """
    if not os.path.isdir(AGENT_RUNS_DIR):
        print(f"[Error] Directory not found: {AGENT_RUNS_DIR}")
        print("Please ensure you have run the agent at least once.")
        return

    all_runs_data = []
    print(f"Reading run files from: {AGENT_RUNS_DIR}")

    # Get all json files, sort them to have a consistent order
    run_files = sorted([f for f in os.listdir(AGENT_RUNS_DIR) if f.endswith(".json")])

    for filename in run_files:
        file_path = os.path.join(AGENT_RUNS_DIR, filename)
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract debate history length
            debate_history = data.get("result", {}).get("debate_history", [])
            debate_turns = len(debate_history) if debate_history is not None else 0

            # Extract the final internal target from the agent
            agent_final_target = data.get("result", {}).get("target", None)

            # Collect the desired data in a dictionary
            run_summary = {
                "run_id": data.get("run_id"),
                "timestamp": data.get("timestamp"),
                "status": data.get("status"),
                "input_text": data.get("input_text"),
                "original_topic": data.get("original_topic"),
                "original_label": data.get("original_label"),
                "predicted_target": data.get("predicted_target"),
                "predicted_stance": data.get("predicted_stance"),
                "debate_turns": debate_turns,
                "agent_final_target": agent_final_target, # The last target before the final formatted response
            }
            all_runs_data.append(run_summary)

        except json.JSONDecodeError:
            print(f"[Warning] Could not decode JSON from file: {filename}")
        except Exception as e:
            print(f"[Warning] An unexpected error occurred while processing {filename}: {e}")

    if not all_runs_data:
        print("No valid run files were found to process.")
        return

    # --- Create and save the CSV ---
    df = pd.DataFrame(all_runs_data)
    
    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSuccessfully compiled {len(df)} runs.")
        print(f"Summary saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"\n[Error] Failed to save output CSV file: {e}")

if __name__ == "__main__":
    compile_agent_runs()
