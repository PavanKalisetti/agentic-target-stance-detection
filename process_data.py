import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. Configuration for the 4 Datasets
# We map the specific column names from your raw files to a standard format:
# "tweet", "target", "stance"
datasets_config = [
    {
        "name": "tse_explicit",
        "path": "/Users/pavan/Documents/college/major-project/agentic-target-stance-detection/data/tse/tse_explicit.csv",
        "mapping": {"tweet": "tweet", "GT Target": "target", "GT Stance": "stance"}
    },
    {
        "name": "tse_implicit",
        "path": "/Users/pavan/Documents/college/major-project/agentic-target-stance-detection/data/tse/tse_implicit.csv",
        "mapping": {"tweet": "tweet", "GT Target": "target", "GT Stance": "stance"}
    },
    {
        "name": "vast_explicit",
        "path": "/Users/pavan/Documents/college/major-project/agentic-target-stance-detection/data/vast/vast_filtered_ex.csv",
        "mapping": {"post": "tweet", "new_topic": "target", "label": "stance"}
    },
    {
        "name": "vast_implicit",
        "path": "/Users/pavan/Documents/college/major-project/agentic-target-stance-detection/data/vast/vast_filtered_im.csv",
        "mapping": {"post": "tweet", "new_topic": "target", "label": "stance"}
    }
]

# Output directory (Files will be saved where you run the script, or change this path)
output_dir = "./processed_data"
os.makedirs(output_dir, exist_ok=True)

# List to hold all training dataframes before merging
all_train_data = []

print(f"Processing datasets and saving to: {output_dir}\n")

for ds in datasets_config:
    try:
        print(f"--> Processing {ds['name']}...")
        
        # 1. Load Data
        df = pd.read_csv(ds["path"])
        
        # 2. Rename Columns to Standard Format
        df = df.rename(columns=ds["mapping"])
        
        # 3. Keep only necessary columns
        df = df[["tweet", "target", "stance"]]
        
        # 4. Drop rows with missing values (just in case)
        df = df.dropna()
        
        # 5. Split Data (80% Train, 20% Test)
        # random_state ensures the split is the same every time you run it
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # 6. Save the Test Set immediately (keeping them separate as requested)
        test_filename = os.path.join(output_dir, f"test_{ds['name']}.csv")
        test_df.to_csv(test_filename, index=False)
        print(f"    Saved Test Set: {test_filename} (Rows: {len(test_df)})")
        
        # 7. Add the Train Set to our list
        all_train_data.append(train_df)
        print(f"    Queued Train Set (Rows: {len(train_df)})")

    except FileNotFoundError:
        print(f"ERROR: Could not find file at {ds['path']}. Please check the path.")
    except KeyError as e:
        print(f"ERROR: Column not found in {ds['name']}. Missing: {e}")

# 8. Merge all training data
if all_train_data:
    print("\n--> Merging Training Data...")
    final_train_df = pd.concat(all_train_data, ignore_index=True)
    
    # 9. Shuffle the combined dataset
    # This is crucial for fine-tuning so the model doesn't learn "Vast" patterns 
    # then "TSE" patterns sequentially.
    final_train_df = final_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 10. Save the Final Merged Train Set
    train_filename = os.path.join(output_dir, "merged_train_dataset.csv")
    final_train_df.to_csv(train_filename, index=False)
    
    print(f"\nSUCCESS!")
    print(f"Merged Training File Saved: {train_filename}")
    print(f"Total Training Samples: {len(final_train_df)}")
    print(f"Columns: {list(final_train_df.columns)}")
else:
    print("\nNo data was processed.")