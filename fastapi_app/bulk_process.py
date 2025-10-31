
import pandas as pd
import os
import sys
import json
import uuid
from datetime import datetime
import xml.etree.ElementTree as ET
import asyncio

# Add project root to system path to allow imports from other directories
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from langgraph_stance_analyzer.main import app as langgraph_app

# --- Configuration ---
# Assuming the 'data' directory is at the project root
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "vast"))
INPUT_CSV_PATH = os.path.join(DATA_DIR, "vast_filtered_ex.csv")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "vast_filtered_ex_with_predictions.csv")
AGENT_RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agent_runs"))
NUM_ROWS_TO_PROCESS = 50

def parse_final_response(final_response_str: str) -> tuple[str | None, str | None]:
    """Parses the XML output from the final agent to extract target and stance."""
    try:
        # The final response might be wrapped in markdown, so we clean it
        if "```xml" in final_response_str:
            clean_xml = final_response_str.split("```xml\n")[1].split("```")[0].strip()
        else:
            clean_xml = final_response_str.strip()

        root = ET.fromstring(clean_xml)
        target = root.find("target").text
        stance = root.find("stance").text
        return target, stance
    except (ET.ParseError, AttributeError, IndexError) as e:
        print(f"  \n[Error] Could not parse XML response: {final_response_str}. Error: {e}")
        return "parsing_error", "parsing_error"

async def process_dataset():
    """
    Reads the input CSV, runs the stance analysis agent on each post,
    and saves the results to a new CSV.
    """
    print(f"Starting bulk processing for {INPUT_CSV_PATH}")
    os.makedirs(AGENT_RUNS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"\n[Error] Input file not found at: {INPUT_CSV_PATH}")
        print("Please ensure the 'data/vast/vast_filtered_ex.csv' file exists.")
        return

    # Limit the dataframe to the first N rows for processing
    df_to_process = df.head(NUM_ROWS_TO_PROCESS).copy()
    
    predicted_targets = []
    predicted_stances = []

    print(f"\nProcessing the first {NUM_ROWS_TO_PROCESS} rows...")

    for index, row in df_to_process.iterrows():
        run_id = str(uuid.uuid4())
        timestamp = datetime.now()
        input_text = row['post']
        
        print(f"\n[{index + 1}/{NUM_ROWS_TO_PROCESS}] Processing run_id: {run_id}")
        print(f"  Input text: \"{input_text[:80]}...\"")

        try:
            # --- Invoke the LangGraph agent ---
            initial_state = {"input": input_text, "target": "", "max_turns": 3}
            # NOTE: Using ainvoke for async compatibility if needed, but running synchronously here.
            result = langgraph_app.invoke(initial_state)
            status = "completed"
            
            # --- Parse the final result ---
            final_response = result.get("final_response", "")
            pred_target, pred_stance = parse_final_response(final_response)
            
        except Exception as e:
            print(f"  \n[Error] An exception occurred during agent invocation: {e}")
            result = {"error": str(e)}
            status = "failed"
            pred_target, pred_stance = "invocation_error", "invocation_error"

        predicted_targets.append(pred_target)
        predicted_stances.append(pred_stance)
        print(f"  -> Predicted Target: {pred_target}")
        print(f"  -> Predicted Stance: {pred_stance}")

        # --- Save the full agent run log ---
        run_data = {
            "run_id": run_id,
            "status": status,
            "input_text": input_text,
            "original_label": row.get('label'),
            "original_topic": row.get('new_topic'),
            "predicted_target": pred_target,
            "predicted_stance": pred_stance,
            "result": result,
            "timestamp": timestamp.isoformat()
        }
        
        file_path = os.path.join(AGENT_RUNS_DIR, f"{run_id}.json")
        with open(file_path, "w") as f:
            # Use a custom encoder to handle non-serializable types if they exist
            json.dump(run_data, f, indent=4, default=str)

    # Add the predictions as new columns to the processed dataframe
    df_to_process['predicted_target'] = predicted_targets
    df_to_process['predicted_stance'] = predicted_stances

    # --- Save the updated dataframe to a new CSV file ---
    try:
        df_to_process.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSuccessfully processed {len(df_to_process)} rows.")
        print(f"Output with predictions saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"\n[Error] Failed to save output CSV file: {e}")

if __name__ == "__main__":
    # Using asyncio.run() to execute the async function
    asyncio.run(process_dataset())
