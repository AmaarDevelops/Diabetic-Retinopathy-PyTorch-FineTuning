import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Assuming the structure is: ./data/images/colored_images/[CLASS_FOLDER]
DATA_ROOT = './data'
TARGET_DIR = './data/processed'

# 1. Path to the folder containing the class folders (Mild, Severe, etc.)
# This ensures the path correctly includes the 'images' subfolder.
IMAGE_SOURCE_DIR = os.path.join(DATA_ROOT, 'images', 'colored_images')

# 2. Path to the CSV file (assumed to be directly in the data root)
CSV_PATH = os.path.join(DATA_ROOT, 'train.csv')

SEED = 42
random.seed(SEED)
os.makedirs(TARGET_DIR, exist_ok=True)

# Define the mapping from numeric label (used in CSV and destination folders)
# to the string folder name (used in the source data structure).
LABEL_TO_FOLDER = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR'
    # NOTE: Double-check the spelling of 'Proliferate_DR' against your source data if errors persist.
}

# ----------------------------------------------------------------------

def split_and_convert_data(image_source_dir, csv_path, target_root, test_size=0.15, val_size=0.15):
    """
    Reads train.csv, performs stratified split, and copies images from the
    string-named source folders into numeric (0-4) Training, Validation, and Testing folders.
    """

    # 1. Load Metadata and Clean-up
    if not os.path.exists(csv_path):
        print(f"Error: train.csv not found at {csv_path}. Cannot perform stratified split.")
        return

    df = pd.read_csv(csv_path)

    # Clean-up old target directory structure
    if os.path.exists(target_root):
        print(f"Cleaning existing directory: {target_root}")
        shutil.rmtree(target_root)
    os.makedirs(target_root, exist_ok=True)

    # 2. Stratified Split (70/15/15)
    print("Performing stratified 70/15/15 split...")

    # Split 1: Training vs. Temp (Validation + Testing)
    train_df, temp_df = train_test_split(
        df,
        test_size=(test_size + val_size),
        stratify=df['diagnosis'],
        random_state=SEED
    )

    # Split 2: Divide Temp into Validation and Testing
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

    print(f"\n--- Data Distribution ---")
    print(f"Total Images: {len(df)}")
    print(f"Training: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"Testing: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")
    print("-------------------------")


    # 3. Copy files to the final numeric destination folders
    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} split...")
        files_copied = 0

        # Create numeric destination folders (e.g., ./data/processed/Training/0, /1, /2...)
        for label in split_df['diagnosis'].unique():
            os.makedirs(os.path.join(target_root, split_name, str(label)), exist_ok=True)

        for index, row in split_df.iterrows():
            src_file_name = f"{row['id_code']}.png"

            # --- CRITICAL FIX: Constructing the source path ---
            # 1. Get the source class folder name (e.g., 'Mild')
            source_class_folder = LABEL_TO_FOLDER[row['diagnosis']]

            # 2. Build the source path: ./data/images/colored_images/Mild/image.png
            src_path = os.path.join(image_source_dir, source_class_folder, src_file_name)

            # Destination remains numeric: ./data/processed/Training/1/image.png
            dst_folder = str(row['diagnosis'])
            dst_path = os.path.join(target_root, split_name, dst_folder, src_file_name)

            try:
                # Copy the file from the source class folder to the numeric destination folder
                shutil.copy(src_path, dst_path)
                files_copied += 1
            except FileNotFoundError:
                print(f"Warning: File not found at source: {src_path}")

        print(f"Finished {split_name}. Copied {files_copied} files.")


    print("\nData splitting and numeric conversion complete! 🎉")
    print(f"Data is ready in the {TARGET_DIR} folder.")


if __name__ == "__main__":
    # The function call passes the correct image source directory
    split_and_convert_data(IMAGE_SOURCE_DIR, CSV_PATH, TARGET_DIR)
