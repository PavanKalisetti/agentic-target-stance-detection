
import requests
import json
import csv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# --- Ollama LLM Communication ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = 'llama3.1:8b'

def stream_ollama(prompt):
    """
    Streams the chat response from the Ollama API to the terminal 
    and returns the full, concatenated response string.
    """
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,  # Enable streaming
    }
    
    full_response = []
    try:
        with requests.post(OLLAMA_API_URL, json=data, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        json_data = json.loads(line)
                        if "content" in json_data["message"]:
                            chunk = json_data["message"]["content"]
                            print(chunk, end="", flush=True)  # Print chunk to terminal
                            full_response.append(chunk)
                    except json.JSONDecodeError:
                        pass  # Ignore non-json lines
        print() # Add a newline after the stream ends
        return "".join(full_response)
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama: {e}")
        return None

def parse_json_from_response(response_str: str):
    """Extracts a JSON object from a string that may contain other text."""
    if not response_str:
        return None
    try:
        # First, try to load it directly after stripping whitespace
        return json.loads(response_str.strip())
    except json.JSONDecodeError:
        # If that fails, fall back to finding the JSON block
        try:
            json_start = response_str.find('{')
            json_end = response_str.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return None
            json_str = response_str[json_start:json_end]
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            return None

# --- Graph State ---
class AgentState(TypedDict):
    tweet: str
    gt_target: str # Ground Truth Target from tse_implicit.csv
    gt_stance: str # Ground Truth Stance from tse_implicit.csv
    linguistic_analysis: str
    target1: str
    target2: str
    target3: str
    stance: str

# --- Graph Nodes ---
def linguistic_analyzer_node(state):
    """
    Performs linguistic analysis on the tweet.
    """
    print("\n--- Running Linguistic Analyzer ---")
    prompt = f"""Analyze the following tweet for its linguistic features, including sentiment, tone, try to keep the analysis 2-3 lines.
    Tweet: "{state['tweet']}"
    Provide a brief analysis:"""
    analysis = stream_ollama(prompt)
    if not analysis:
        analysis = "Error in analysis."
    print(f"--- LINGUISTIC ANALYSIS ---\n{analysis}\n")
    return {"linguistic_analysis": analysis}

def target_detection_node(state):
    """
    Detects potential targets in the tweet.
    """
    print("\n--- Running Target Detector ---")
    prompt = f"""You are a JSON-only API. Your sole purpose is to identify concise, 2-3 word targets from a tweet.

**Instructions:**
- Analyze the user's tweet.
- Identify the top 3 targets.
- The targets must be concise (2-3 words).
- Your response MUST be a single JSON object.
- DO NOT provide any text, explanation, or code before or after the JSON object.

**Example:**
[INPUT]
Tweet: "I can't believe they are still pushing that awful new update. It's slow and buggy."
[OUTPUT]
{{
  "target1": "new update",
  "target2": "software quality",
  "target3": "company performance"
}}

**Task:**
[INPUT]
Tweet: "{state['tweet']}"
[OUTPUT]
"""
    full_response = stream_ollama(prompt)
    targets = parse_json_from_response(full_response)
    if not targets:
        targets = {"target1": "ERROR", "target2": "ERROR", "target3": "ERROR"}
    print(f"--- PARSED TARGETS ---\n{targets}\n")
    return {
        "target1": targets.get("target1", "N/A"),
        "target2": targets.get("target2", "N/A"),
        "target3": targets.get("target3", "N/A"),
    }

def stance_detection_node(state):
    """
    Determines the stance towards the primary target.
    """
    print("\n--- Running Stance Detector ---")
    primary_target = state["target1"]
    if primary_target in ["N/A", "ERROR"]:
        print("--- SKIPPING STANCE DETECTION (No valid primary target) ---")
        return {"stance": "UNABLE_TO_DETERMINE"}

    prompt = f"""You are a JSON-only API. Your sole purpose is to determine the stance towards a target from a tweet.

**Instructions:**
- Analyze the user's tweet and the given target.
- The stance MUST be one of 'FAVOR', 'AGAINST', or 'NEUTRAL'.
- Your response MUST be a single JSON object.
- DO NOT provide any text, explanation, or code before or after the JSON object.

**Example:**
[INPUT]
Tweet: "I can't believe they are still pushing that awful new update. It's slow and buggy."
Target: "new update"
[OUTPUT]
{{
  "stance": "AGAINST"
}}

**Task:**
[INPUT]
Tweet: "{state['tweet']}"
Target: "{primary_target}"
[OUTPUT]
"""
    full_response = stream_ollama(prompt)
    parsed_json = parse_json_from_response(full_response)
    stance = parsed_json.get("stance", "ERROR") if parsed_json else "ERROR"
    print(f"--- PARSED STANCE ---\n{stance}\n")
    return {"stance": stance}

# --- Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("linguiprimary_tastic_analyzer", linguistic_analyzer_node)
workflow.add_node("target_detector", target_detection_node)
workflow.add_node("stance_detector", stance_detection_node)

workflow.set_entry_point("linguistic_analyzer")
workflow.add_edge("linguistic_analyzer", "target_detector")
workflow.add_edge("target_detector", "stance_detector")
workflow.add_edge("stance_detector", END)

app = workflow.compile()

# --- Main Execution ---
def main():
    """
    Main function to run the stance detection agent and save results to a CSV file.
    """
    input_csv_path = "/home/rgukt/Documents/major project/major-project/data/tse/tse_implicit.csv"
    output_csv_path = "/home/rgukt/Documents/major project/major-project/stance_analysis_results_tse_implicit.csv"
    
    print(f"Processing tweets from: {input_csv_path}")
    print(f"Saving results to: {output_csv_path}")

    with open(input_csv_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        
        # Write header to the output file
        writer.writerow(['tweet', 'GT Target', 'GT Stance', 'target1', 'target2', 'target3', 'stance'])
        
        for i, row in enumerate(reader):
            if i >= 100:
                break
            
            tweet_text = row['tweet']
            gt_target = row['GT Target']
            gt_stance = row['GT Stance']
            
            print(f"\n--- PROCESSING TWEET {i+1}/{100} ---")
            
            initial_state = {
                "tweet": tweet_text,
                "gt_target": gt_target,
                "gt_stance": gt_stance
            }
            
            final_state = app.invoke(initial_state)
            
            # Write the results to the output CSV
            writer.writerow([
                final_state['tweet'],
                final_state['gt_target'],
                final_state['gt_stance'],
                final_state.get('target1', 'N/A'),
                final_state.get('target2', 'N/A'),
                final_state.get('target3', 'N/A'),
                final_state.get('stance', 'N/A')
            ])
            outfile.flush() # Force write to disk after each row
            print("--- Result saved to CSV ---")

if __name__ == "__main__":
    main()
