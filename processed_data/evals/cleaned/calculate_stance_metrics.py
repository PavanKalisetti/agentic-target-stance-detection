import pandas as pd
import glob
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_metrics():
    # 1. Find the files (Targeting the 'clean' ones, or falling back to evaluated)
    # Adjust this pattern if your files are named differently
    files = glob.glob("*_clean_results.csv")
    if not files:
        files = glob.glob("*_evaluated.csv")
    if not files:
        files = glob.glob("agent_results_*.csv")
        
    if not files:
        print("No result files found to evaluate. Please run this in the folder with your CSVs.")
        return

    print(f"Found {len(files)} files. Calculating Stance Metrics...\n")
    
    all_metrics = []

    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            
            # --- Robust Column Mapping ---
            col_map = {k.lower(): k for k in df.columns}
            
            # Identify Ground Truth Stance Column
            gt_col = col_map.get('gt_stance') or col_map.get('stance') or col_map.get('label')
            
            # Identify Predicted Stance Column
            pred_col = col_map.get('predicted_stance') or col_map.get('pred_stance')
            
            if not gt_col or not pred_col:
                print(f"[SKIP] {os.path.basename(filepath)}: Could not find Stance columns (GT or Pred).")
                continue
                
            # --- Data Cleaning ---
            # 1. Convert to string
            # 2. Uppercase (to match FAVOR vs Favor)
            # 3. Strip whitespace
            y_true = df[gt_col].astype(str).str.upper().str.strip()
            y_pred = df[pred_col].astype(str).str.upper().str.strip()
            
            # --- Metric Calculation ---
            # Accuracy
            acc = accuracy_score(y_true, y_pred)
            
            # Macro Average (Calculates metrics for each label, and finds their unweighted mean)
            # Good for understanding performance on minority classes.
            prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            
            # Micro Average (Calculate metrics globally by counting total true positives, etc.)
            # Good for overall performance if classes are imbalanced.
            prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='micro', zero_division=0
            )
            
            # Weighted Average (Calculate metrics for each label, and find their average weighted by support)
            prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )

            print(f"--> Processed: {os.path.basename(filepath)}")

            all_metrics.append({
                "File": os.path.basename(filepath),
                "Accuracy": round(acc, 4),
                "F1_Macro": round(f1_macro, 4),
                "F1_Micro": round(f1_micro, 4),
                "F1_Weighted": round(f1_weighted, 4),
                "Precision_Macro": round(prec_macro, 4),
                "Recall_Macro": round(rec_macro, 4),
                "Precision_Micro": round(prec_micro, 4),
                "Recall_Micro": round(rec_micro, 4)
            })
            
        except Exception as e:
            print(f"[ERROR] Processing {filepath}: {e}")

    # --- Save Report ---
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        output_filename = "stance_metrics_report.csv"
        metrics_df.to_csv(output_filename, index=False)
        
        print("\n" + "="*60)
        print(f"REPORT SAVED: {output_filename}")
        print("="*60)
        print(metrics_df.to_string(index=False))
    else:
        print("\nNo metrics could be calculated.")

if __name__ == "__main__":
    calculate_metrics()