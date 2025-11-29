import pandas as pd
import glob
import os
import numpy as np

def inverse_formula(boosted_score):
    """
    Reverses the formula: y = x + (1-x)*0.6
    Derived inverse: x = (y - 0.6) / 0.4
    """
    try:
        y = float(boosted_score)
        
        # If it's exactly 1.0 or higher, the raw score was likely 1.0
        if y >= 1.0:
            return 1.0
            
        # Apply inverse formula
        raw_score = (y - 0.6) / 0.4
        
        # Clamp between 0 and 1 (to handle any floating point rounding errors)
        return max(0.0, min(1.0, raw_score))
    except:
        return 0.0

def process_files():
    # Find the specific files
    files = glob.glob("*_advanced_evaluated.csv")
    
    if not files:
        print("No '_advanced_evaluated.csv' files found.")
        return

    print(f"Found {len(files)} files to restore.\n")

    for filepath in files:
        print(f"---> Processing: {filepath}")
        
        try:
            # Load CSV
            df = pd.read_csv(filepath)
            
            # Check if the column exists
            if "Normalized_Target_Similarity" not in df.columns:
                print(f"   [SKIP] Column 'Normalized_Target_Similarity' not found.")
                continue

            # Apply the inverse formula
            # We rename the column to 'Raw_Cosine_Similarity' to be accurate
            df['Raw_Cosine_Similarity'] = df['Normalized_Target_Similarity'].apply(inverse_formula)
            
            # Drop the artificially boosted column
            df = df.drop(columns=['Normalized_Target_Similarity'])
            
            # Reorder columns to put Similarity near Stance_Correct if possible
            cols = [c for c in df.columns if c not in ['Raw_Cosine_Similarity', 'Stance_Correct']]
            # Add metrics to the end
            cols = cols + ['Raw_Cosine_Similarity', 'Stance_Correct']
            # Only keep columns that actually exist in the dataframe
            final_cols = [c for c in cols if c in df.columns]
            df = df[final_cols]

            # Calculate Stats
            avg_sim = df['Raw_Cosine_Similarity'].mean()
            
            # Save to new file
            output_name = filepath.replace("_advanced_evaluated.csv", "_clean_results.csv")
            df.to_csv(output_name, index=False)
            
            print(f"   Saved to: {output_name}")
            print(f"   Real Average Similarity: {avg_sim:.4f}")
            
        except Exception as e:
            print(f"   [ERROR] {e}")

if __name__ == "__main__":
    process_files()