"""
Script to merge multiple stance detection training datasets into one unified training set,
and process test/val datasets with standardized columns, preserving folder structure.
"""

import pandas as pd
import os
from pathlib import Path

# Base directory for datasets
BASE_DIR = Path("/Users/pavan/Documents/college/major-project/agentic-target-stance-detection/data_new/Zero_Stance-Chat_GPT")

# Output directories
OUTPUT_DIR = Path("./processed_data/unified_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output directory for processed test/val (mirrors original structure)
PROCESSED_DATA_DIR = Path("./data_processed/Zero_Stance-Chat_GPT")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Standard column names
STANDARD_COLS = {
    'tweet': ['Tweet', 'tweet', 'post', 'text'],
    'target': ['Target 1', 'target', 'GT Target', 'new_topic', 'Target ori'],
    'stance': ['Stance 1', 'stance', 'label', 'GT Stance']
}

# Stance normalization
STANCE_MAP = {
    'FAVOR': 'FAVOR',
    'AGAINST': 'AGAINST',
    'NONE': 'NONE',
    'NEUTRAL': 'NONE',
    'support': 'FAVOR',
    'refute': 'AGAINST',
    'unrelated': 'NONE'
}

# Training datasets to merge (original, clean datasets only)
# Skip augmented versions (_asda, _bt, _eda, _aux, etc.)
TRAIN_DATASETS = [
    # Original datasets - use main files, not augmented versions
    'Originial datasets/covid19/raw_train_all_onecol.csv',
    'Originial datasets/ibm30k/raw_train_all_onecol.csv',
    'Originial datasets/pstance/raw_train_all_onecol.csv',
    'Originial datasets/semeval2016/raw_train_all_onecol.csv',
    'Originial datasets/vast/raw_train_all_onecol.csv',
    'Originial datasets/wtwt/raw_train_all_onecol.csv',
    
    # Open Stance datasets
    'Open Stance datsets/openstance_semeval2016/raw_train_all_onecol.csv',
    'Open Stance datsets/openstance_vast/raw_train_all_onecol.csv',
    
    # Chat GPT datasets
    'Chat GPT datasets/chatgpt_covid19/raw_train_all_onecol.csv',
    'Chat GPT datasets/chatgpt_ibm30k/raw_train_all_onecol.csv',
    'Chat GPT datasets/chatgpt_pstance/raw_train_all_onecol.csv',
    'Chat GPT datasets/chatgpt_semeval2016/raw_train_all_onecol.csv',
    'Chat GPT datasets/chatgpt_vast/raw_train_all_onecol.csv',
    'Chat GPT datasets/chatgpt_wtwt/raw_train_all_onecol.csv',
]

# Special training datasets with subdirectories
SPECIAL_TRAIN_PATHS = [
    'Originial datasets/ibm30k/pos_targets/raw_train_all_onecol.csv',
    'Originial datasets/wtwt/keep_unrelated_remove_comment/raw_train_all_onecol.csv',
]

# Test and Val datasets to process (same structure as train)
TEST_VAL_DATASETS = [
    # Original datasets
    'Originial datasets/covid19/raw_test_all_onecol.csv',
    'Originial datasets/covid19/raw_val_all_onecol.csv',
    'Originial datasets/ibm30k/raw_test_all_onecol.csv',
    'Originial datasets/ibm30k/raw_val_all_onecol.csv',
    'Originial datasets/pstance/raw_test_all_onecol.csv',
    'Originial datasets/pstance/raw_val_all_onecol.csv',
    'Originial datasets/semeval2016/raw_test_all_onecol.csv',
    'Originial datasets/semeval2016/raw_val_all_onecol.csv',
    'Originial datasets/vast/raw_test_all_onecol.csv',
    'Originial datasets/vast/raw_val_all_onecol.csv',
    'Originial datasets/wtwt/raw_test_all_onecol.csv',
    'Originial datasets/wtwt/raw_val_all_onecol.csv',
    
    # Open Stance datasets
    'Open Stance datsets/openstance_semeval2016/raw_test_all_onecol.csv',
    'Open Stance datsets/openstance_semeval2016/raw_val_all_onecol.csv',
    'Open Stance datsets/openstance_vast/raw_test_all_onecol.csv',
    'Open Stance datsets/openstance_vast/raw_val_all_onecol.csv',
    
    # Chat GPT datasets
    'Chat GPT datasets/chatgpt_covid19/raw_test_all_onecol.csv',
    'Chat GPT datasets/chatgpt_covid19/raw_val_all_onecol.csv',
    'Chat GPT datasets/chatgpt_ibm30k/raw_test_all_onecol.csv',
    'Chat GPT datasets/chatgpt_ibm30k/raw_val_all_onecol.csv',
    'Chat GPT datasets/chatgpt_pstance/raw_test_all_onecol.csv',
    'Chat GPT datasets/chatgpt_pstance/raw_val_all_onecol.csv',
    'Chat GPT datasets/chatgpt_semeval2016/raw_test_all_onecol.csv',
    'Chat GPT datasets/chatgpt_semeval2016/raw_val_all_onecol.csv',
    'Chat GPT datasets/chatgpt_vast/raw_test_all_onecol.csv',
    'Chat GPT datasets/chatgpt_vast/raw_val_all_onecol.csv',
    'Chat GPT datasets/chatgpt_wtwt/raw_test_all_onecol.csv',
    'Chat GPT datasets/chatgpt_wtwt/raw_val_all_onecol.csv',
]

# Special test/val datasets with subdirectories
SPECIAL_TEST_VAL_PATHS = [
    'Originial datasets/ibm30k/pos_targets/raw_test_all_onecol.csv',
    'Originial datasets/ibm30k/pos_targets/raw_val_all_onecol.csv',
    'Originial datasets/wtwt/keep_unrelated_remove_comment/raw_test_all_onecol.csv',
    'Originial datasets/wtwt/keep_unrelated_remove_comment/raw_val_all_onecol.csv',
]


def normalize_stance(stance):
    """Normalize stance labels to standard format."""
    if pd.isna(stance):
        return None
    stance_str = str(stance).strip().upper()
    return STANCE_MAP.get(stance_str, stance_str if stance_str in ['FAVOR', 'AGAINST', 'NONE'] else None)


def find_column(df, possible_names):
    """Find a column in dataframe by trying multiple possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def standardize_columns(df):
    """Standardize dataframe columns to tweet, target, stance."""
    # Find the actual column names
    tweet_col = find_column(df, STANDARD_COLS['tweet'])
    target_col = find_column(df, STANDARD_COLS['target'])
    stance_col = find_column(df, STANDARD_COLS['stance'])
    
    if not tweet_col or not target_col or not stance_col:
        return None
    
    # Create standardized dataframe
    standardized = pd.DataFrame({
        'tweet': df[tweet_col].astype(str),
        'target': df[target_col].astype(str),
        'stance': df[stance_col]
    })
    
    # Normalize stance
    standardized['stance'] = standardized['stance'].apply(normalize_stance)
    
    # Drop rows with missing or invalid data
    standardized = standardized.dropna(subset=['tweet', 'target', 'stance'])
    standardized = standardized[standardized['tweet'].str.strip() != '']
    standardized = standardized[standardized['target'].str.strip() != '']
    standardized = standardized[standardized['stance'].isin(['FAVOR', 'AGAINST', 'NONE'])]
    
    # Remove duplicates
    standardized = standardized.drop_duplicates(subset=['tweet', 'target', 'stance'])
    
    return standardized


def load_dataset(file_path, split_type="train"):
    """Load and standardize a dataset file."""
    try:
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_csv(file_path, low_memory=False)
        standardized = standardize_columns(df)
        
        if standardized is None or len(standardized) == 0:
            return None
        
        print(f"  ✓ Loaded {split_type}: {len(standardized)} rows from {os.path.basename(file_path)}")
        return standardized
    
    except Exception as e:
        print(f"  ✗ Error loading {file_path}: {e}")
        return None


def collect_train_datasets():
    """Collect all training datasets from priority list."""
    train_datasets = []
    
    print("Collecting training datasets...\n")
    
    # Load main training datasets
    for rel_path in TRAIN_DATASETS:
        full_path = BASE_DIR / rel_path
        df = load_dataset(full_path, "train")
        if df is not None:
            train_datasets.append(df)
    
    # Load special training datasets
    for rel_path in SPECIAL_TRAIN_PATHS:
        full_path = BASE_DIR / rel_path
        df = load_dataset(full_path, "train")
        if df is not None:
            train_datasets.append(df)
    
    return train_datasets


def process_test_val_datasets():
    """Process test and val datasets, standardize columns, and save with same folder structure."""
    print(f"\n{'=' * 80}")
    print("Processing test and validation datasets...")
    print(f"{'=' * 80}\n")
    
    all_test_val = TEST_VAL_DATASETS + SPECIAL_TEST_VAL_PATHS
    processed_count = 0
    
    for rel_path in all_test_val:
        full_path = BASE_DIR / rel_path
        
        # Determine split type from filename
        if 'raw_test' in rel_path:
            split_type = 'test'
        elif 'raw_val' in rel_path:
            split_type = 'val'
        else:
            continue
        
        # Load and standardize
        df = load_dataset(full_path, split_type)
        if df is None:
            continue
        
        # Create output path preserving folder structure
        output_path = PROCESSED_DATA_DIR / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save standardized dataset
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved {split_type}: {output_path}")
        processed_count += 1
    
    print(f"\n✓ Processed {processed_count} test/val datasets")
    return processed_count


def merge_and_deduplicate(df_list):
    """Merge multiple dataframes and remove duplicates."""
    if not df_list:
        return pd.DataFrame(columns=['tweet', 'target', 'stance'])
    
    merged = pd.concat(df_list, ignore_index=True)
    merged = merged.drop_duplicates(subset=['tweet', 'target', 'stance'])
    return merged.reset_index(drop=True)




def main():
    print("=" * 80)
    print("Dataset Processing Script")
    print("=" * 80)
    print()
    
    # Step 1: Collect all training datasets
    train_datasets = collect_train_datasets()
    
    print(f"\n{'=' * 80}")
    print("Dataset Summary:")
    print(f"{'=' * 80}")
    print(f"Training datasets found: {len(train_datasets)}")
    print()
    
    # Step 2: Merge all training datasets
    print("Merging training datasets...")
    train_merged = merge_and_deduplicate(train_datasets)
    
    print(f"\nAfter merging and deduplication:")
    print(f"  Train: {len(train_merged)} rows")
    
    # Step 3: Final statistics for training
    print(f"\n{'=' * 80}")
    print("Final Training Dataset Statistics:")
    print(f"{'=' * 80}")
    print(f"Total rows: {len(train_merged)}")
    print(f"  - FAVOR: {len(train_merged[train_merged['stance'] == 'FAVOR'])}")
    print(f"  - AGAINST: {len(train_merged[train_merged['stance'] == 'AGAINST'])}")
    print(f"  - NONE: {len(train_merged[train_merged['stance'] == 'NONE'])}")
    print(f"  - Unique targets: {train_merged['target'].nunique()}")
    print(f"  - Unique tweets: {train_merged['tweet'].nunique()}")
    print()
    
    # Step 4: Shuffle and save unified training dataset
    print(f"{'=' * 80}")
    print("Saving unified training dataset...")
    print(f"{'=' * 80}")
    
    # Shuffle dataset
    train_final = train_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    train_path = OUTPUT_DIR / "unified_train.csv"
    train_final.to_csv(train_path, index=False)
    
    print(f"\n✓ Saved unified training dataset: {train_path}")
    
    # Step 5: Process test and val datasets
    processed_count = process_test_val_datasets()
    
    print(f"\n{'=' * 80}")
    print("SUCCESS! All datasets processed and saved.")
    print(f"{'=' * 80}")
    print(f"  - Unified training dataset: {train_path}")
    print(f"  - Processed test/val datasets: {processed_count} files")
    print(f"  - Output location: {PROCESSED_DATA_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

