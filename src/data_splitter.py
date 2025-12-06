import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
# NOTE: The CSV and the images should be in the same root folder for this script to work.
DATA_ROOT = './data'
CSV_PATH = os.path.join(DATA_ROOT, 'train.csv')
TARGET_DIR = './data/processed'
SEED = 42
random.seed(SEED)
os.makedirs(TARGET_DIR, exist_ok=True)

def split_and_convert_data(data_root, csv_path, target_root, test_size=0.15, val_size=0.15):
    """
    Reads train.csv, performs stratified split, and copies images into
    numeric (0-4) Training, Validation, and Testing folders.
    """

    # 1. Load Metadata and Prepare Data Index
    if not os.path.exists(csv_path):
        print(f"Error: train.csv not found at {csv_path}. Cannot perform stratified split.")
        return

    df = pd.read_csv(csv_path)

    # 2. Convert string folder names (if they exist) to their numeric labels
    # NOTE: Since the data structure is assumed to be (ID, Diagnosis), we can use the diagnosis column directly.
    # The actual image names may need a '.png' suffix added for matching.
    df['file_path'] = df['id_code'].apply(lambda x: os.path.join(data_root, f"{x}.png"))

    # --- Clean-up old target directory structure (important for re-running) ---
    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    os.makedirs(target_root, exist_ok=True)

    # 3. Stratified Split (Training, Temp/Validation + Testing)
    # Split 1: Separate 70% for Training, 30% for Temp (Validation + Testing)
    train_df, temp_df = train_test_split(
        df,
        test_size=(test_size + val_size),
        stratify=df['diagnosis'],
        random_state=SEED
    )

    # Split 2: Divide Temp into Validation (15% / 30% of original = 50% of temp) and Testing (15%)
    # Ratio must be relative to the temp_df size: 0.15 / (0.15 + 0.15) = 0.5
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_size / (test_size + val_size)),
        stratify=temp_df['diagnosis'],
        random_state=SEED
    )

    splits = {
        'Training': train_df,
        'Validation': val_df,
        'Testing': test_df
    }

    print(f"Data Distribution:")
    print(f"Training: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"Testing: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")


    # 4. Copy files to the final numeric destination folders
    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name}...")

        # Create destination directories for this split (e.g., ./data/processed/Training/0, /1, /2...)
        for label in split_df['diagnosis'].unique():
            os.makedirs(os.path.join(target_root, split_name, str(label)), exist_ok=True)

        for index, row in split_df.iterrows():
            src_file_name = f"{row['id_code']}.png" # Assuming files are .png
            src_path = os.path.join(data_root, src_file_name)

            # The destination folder is the numeric diagnosis label
            dst_folder = str(row['diagnosis'])
            dst_path = os.path.join(target_root, split_name, dst_folder, src_file_name)

            try:
                # Assuming the image files are directly in the colored_images folder
                shutil.copy(src_path, dst_path)
            except FileNotFoundError:
                # Handle images not found (e.g., if files are not .png or if the file is missing)
                print(f"Warning: File not found for ID: {row['id_code']}")

    print("\nData splitting and numeric conversion complete!")
    print(f"Data is ready in the {TARGET_DIR} folder.")


if __name__ == "__main__":
    split_and_convert_data(DATA_ROOT, CSV_PATH, TARGET_DIR)
