import os
import shutil
import random
import pandas as pd # Assuming metadata is in a pandas-readable format like CSV
from sklearn.model_selection import train_test_split


# --- Configuration ---
# Set the base paths for your downloaded datasets
VINDR_MAMMO_ORIGINAL_PATH = './data/mammo' # Adjust this path if needed
# Path to the VinDr-Mammo metadata file (e.g., CSV with image paths and labels)
VINDR_MAMMO_METADATA_PATH = './data/mammo/finding_annotations.csv' # <<< ADJUST THIS PATH

BUSI_ORIGINAL_PATH = './data/ultrasound' # Adjust this path if needed
# KAU_BCMD_ORIGINAL_PATH = './data/kau_bcmd' # Path for KAU-BCMD (handle separately for multimodal pairing)

# Set the base path for the new organized dataset structure
ORGANIZED_DATASET_BASE_PATH = './organized_multimodal_dataset'

# Define split ratios (train, validation, test) - should sum to 1.0
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Hugging Face configuration
HF_REPO_NAME = 'your-username/multimodal-breast-cancer-imaging' # Replace with your Hugging Face username and desired repo name
HF_TOKEN = 'YOUR_HF_WRITE_TOKEN' # Replace with your actual Hugging Face write token

# --- Data Splitting Function ---
def split_dataset(original_path, metadata_path=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits image files into train, val, and test sets based on provided metadata or
    assuming benign/malignant subfolders.
    Returns dictionaries mapping split names to lists of (file_path, label) tuples.
    """
    print(f"Splitting dataset from: {original_path}")

    all_files = []
    all_labels = []
    label_map = {'benign': 0, 'malignant': 1, 'normal': 0} # Assuming 'normal' can be grouped with 'benign' or handled separately

    if metadata_path and os.path.exists(metadata_path):
        print(f"Using metadata file: {metadata_path}")
        # Assuming metadata is a CSV with columns like 'image_id', 'finding_category', 'file_path'
        # You might need to adjust column names based on your actual metadata file
        try:
            metadata_df = pd.read_csv(metadata_path)

            # Filter for relevant findings (e.g., 'Benign', 'Malignant') - ADJUST AS NEEDED
            # This part is highly dependent on the structure and content of your metadata CSV
            relevant_findings_df = metadata_df[metadata_df['finding_category'].isin(['Benign', 'Malignant'])]

            for index, row in relevant_findings_df.iterrows():
                # Construct the full path to the image file
                # This assumes the 'file_path' column in metadata is relative to original_path
                # You might need to adjust how the file path is constructed based on your file structure
                image_file_name = row['image_id'] + '.dicom' # Example: adjust extension if needed
                image_path = os.path.join(original_path, row['image_id'].split('_')[0], image_file_name) # Example path construction

                if os.path.exists(image_path):
                    label = label_map.get(row['finding_category'].lower(), -1) # Get label, default to -1 if unknown
                    if label != -1:
                        all_files.append(image_path)
                        all_labels.append(label)
                    else:
                        print(f"Warning: Unknown finding category '{row['finding_category']}' for image {image_path}. Skipping.")
                else:
                    print(f"Warning: Image file not found at {image_path}. Skipping.")

        except Exception as e:
            print(f"Error reading or processing metadata file {metadata_path}: {e}")
            return None

    else:
        # Fallback to assuming benign/malignant subfolders if no metadata or metadata not found
        print("No metadata path provided or metadata file not found. Assuming 'benign' and 'malignant' subfolders.")
        benign_dir_original = os.path.join(original_path, 'benign')
        malignant_dir_original = os.path.join(original_path, 'malignant')

        if not os.path.exists(benign_dir_original) or not os.path.exists(malignant_dir_original):
            print(f"Error: Expected 'benign' and 'malignant' subfolders in {original_path} or a valid metadata file. Cannot split.")
            return None

        benign_files = [os.path.join(benign_dir_original, f) for f in os.listdir(benign_dir_original) if not f.startswith('.')]
        malignant_files = [os.path.join(malignant_dir_original, f) for f in os.listdir(malignant_dir_original) if not f.startswith('.')]

        all_files = benign_files + malignant_files
        all_labels = [0] * len(benign_files) + [1] * len(malignant_files)


    if not all_files:
        print(f"No valid image files found in {original_path} or via metadata. Cannot split.")
        return None

    # Perform the split
    # Ensure test_size is calculated correctly to sum val_ratio + test_ratio
    test_size_temp = val_ratio + test_ratio
    if test_size_temp >= 1.0:
         print(f"Error: train_ratio ({train_ratio}) is too low or val_ratio ({val_ratio}) + test_ratio ({test_ratio}) is >= 1.0. Cannot split.")
         return None


    files_train, files_temp, labels_train, labels_temp = train_test_split(
        all_files, all_labels, test_size=test_size_temp, stratify=all_labels, random_state=42
    )

    # Calculate test_size for the second split relative to the temporary set
    # Avoid division by zero if test_size_temp is 0
    test_size_final = test_ratio / test_size_temp if test_size_temp > 0 else 0

    files_val, files_test, labels_val, labels_test = train_test_split(
        files_temp, labels_temp, test_size=test_size_final, stratify=labels_temp, random_state=42
    )

    splits = {
        'train': list(zip(files_train, labels_train)),
        'validation': list(zip(files_val, labels_val)),
        'test': list(zip(files_test, labels_test))
    }

    print(f"Splitting complete. Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
    return splits

# --- Data Organization Function ---
def organize_split_data(split_data, organized_base_path, dataset_name):
    """
    Organizes the split data (list of (file_path, label) tuples)
    into the target directory structure.
    """
    print(f"Organizing split data for: {dataset_name}")

    for split_name, data_list in split_data.items():
        split_dir = os.path.join(organized_base_path, dataset_name, split_name)
        benign_split_dir = os.path.join(split_dir, 'benign')
        malignant_split_dir = os.path.join(split_dir, 'malignant')

        os.makedirs(benign_split_dir, exist_ok=True)
        os.makedirs(malignant_split_dir, exist_ok=True)

        print(f"Copying files for {dataset_name}/{split_name}...")
        for file_path, label in data_list:
            file_name = os.path.basename(file_path)
            if label == 0: # Benign (or Normal)
                shutil.copy(file_path, os.path.join(benign_split_dir, file_name))
            else: # Malignant
                shutil.copy(file_path, os.path.join(malignant_split_dir, file_name))
        print(f"Finished copying for {dataset_name}/{split_name}. Total files: {len(data_list)}")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Process VinDr-Mammo
    # Pass the metadata path for VinDr-Mammo
    vindr_mammo_splits = split_dataset(
        VINDR_MAMMO_ORIGINAL_PATH,
        metadata_path=VINDR_MAMMO_METADATA_PATH,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    if vindr_mammo_splits:
        organize_split_data(
            vindr_mammo_splits,
            ORGANIZED_DATASET_BASE_PATH,
            'vindr_mammo'
        )

    # 2. Process BUSI
    # BUSI dataset likely has benign/malignant subfolders directly, so no metadata_path needed
    busi_splits = split_dataset(
        BUSI_ORIGINAL_PATH,
        metadata_path=None, # No metadata path needed for BUSI if it's in subfolders
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    if busi_splits:
        organize_split_data(
            busi_splits,
            ORGANIZED_DATASET_BASE_PATH,
            'busi'
        )

    # TODO: Implement specific logic for KAU-BCMD to handle paired images for multimodal evaluation.
    # This might involve reading a manifest or using file naming conventions to group images.
    # For now, KAU-BCMD is not automatically split and organized by this script.
    print("\nNote: KAU-BCMD dataset splitting and organization for multimodal pairing needs to be handled separately.")


