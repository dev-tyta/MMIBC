import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Set the base path for your downloaded VinDr-Mammo dataset
VINDR_MAMMO_ORIGINAL_PATH = './data/mammo' # Adjust this path if needed

# Paths to the VinDr-Mammo metadata files
BREAST_LEVEL_METADATA_PATH = os.path.join(VINDR_MAMMO_ORIGINAL_PATH, 'breast-level_annotations.csv')
FINDING_METADATA_PATH = os.path.join(VINDR_MAMMO_ORIGINAL_PATH, 'finding_annotations.csv')
# METADATA_CSV_PATH = os.path.join(VINDR_MAMMO_ORIGINAL_PATH, 'metadata.csv') # Optional: for image details

# Set the base path for the new organized dataset structure
ORGANIZED_DATASET_BASE_PATH = './mmibc/vindr_mammo' # Specific path for VinDr-Mammo
ORGANIZED_METADATA_FILE = os.path.join(ORGANIZED_DATASET_BASE_PATH, 'vindr_mammo_metadata.csv') # Output metadata file

# Define split ratios for splitting the *original training data* into our train/validation sets
# The original 'test' split will be used as our test set.
TRAIN_RATIO_ORIGINAL_TRAINING = 0.8 # Ratio of original training data for our training set
VAL_RATIO_ORIGINAL_TRAINING = 0.2  # Ratio of original training data for our validation set
# Note: Original test data will be used entirely for our test set.

# --- Function to Process Metadata and Get Image Labels ---
def get_image_labels_and_paths(original_data_path, breast_metadata_path, finding_metadata_path):
    """
    Reads VinDr-Mammo metadata to determine binary labels (benign/malignant)
    and maps image IDs to file paths. Includes original split and study_id.
    Returns a list of dictionaries: [{'image_id': '...', 'label': 0/1, 'original_split': '...', 'file_path': '...', 'study_id': '...'}]
    """
    print("Processing VinDr-Mammo metadata...")

    image_info_list = [] # List to store info for each image

    # Load metadata files
    finding_df = None
    if os.path.exists(finding_metadata_path):
        finding_df = pd.read_csv(finding_metadata_path)
        # Map finding BI-RADS to binary malignancy (4, 5, 6 are malignant)
        finding_df['is_malignant_finding'] = finding_df['finding_birads'].isin(['BI-RADS 4', 'BI-RADS 5', 'BI-RADS 6']).astype(int)
    else:
        print(f"Warning: Finding metadata file not found at {finding_metadata_path}. Malignant cases might be missed.")

    breast_df = None
    if os.path.exists(breast_metadata_path):
        breast_df = pd.read_csv(breast_metadata_path)
    else:
        print(f"Warning: Breast-level metadata file not found at {breast_metadata_path}. Benign/normal cases might be missed.")

    # Combine relevant info from both dataframes
    if finding_df is not None and breast_df is not None:
        # Use breast_df as the base, as it lists all images (including normals)
        # Merge finding info onto breast info
        merged_df = pd.merge(breast_df, finding_df[['image_id', 'is_malignant_finding']], on='image_id', how='left')
        # Fill NaN malignant findings with 0 (meaning no malignant finding in this image)
        merged_df['is_malignant_finding'] = merged_df['is_malignant_finding'].fillna(0)

        # Determine final binary label (1 if any malignant finding, 0 otherwise)
        merged_df['final_label'] = merged_df['is_malignant_finding'].clip(upper=1) # Ensure label is 0 or 1

        # Now, iterate through the merged dataframe to collect image info
        for index, row in merged_df.iterrows():
            image_id = row['image_id']
            study_id = row['study_id']
            original_split = row['split']
            label = int(row['final_label']) # Ensure label is integer 0 or 1

            # Construct the full path to the image file
            # Assuming .png extension based on SHA256SUMS, adjust if necessary (e.g., .dicom)
            image_file_path = os.path.join(original_data_path, 'images', study_id, f'{image_id}.png')


            if os.path.exists(image_file_path):
                image_info_list.append({
                    'image_id': image_id,
                    'study_id': study_id,
                    'label': label,
                    'original_split': original_split,
                    'file_path': image_file_path,
                    # Add other relevant metadata if needed, e.g., 'breast_birads', 'finding_categories'
                    'breast_birads': row.get('breast_birads'), # Use .get to avoid error if column is missing
                    'breast_density': row.get('breast_density'),
                    # Note: finding_categories is a list in CSV, might need careful handling
                })
            else:
                 print(f"Warning: Image file not found for image {image_id} at {image_file_path}. Skipping.")

    elif breast_df is not None:
         print("Proceeding with breast-level metadata only. Malignant cases from findings might be missed.")
         for index, row in breast_df.iterrows():
            image_id = row['image_id']
            study_id = row['study_id']
            original_split = row['split']
            # Assume benign/normal if only breast-level metadata is available and BI-RADS is 1, 2, or 3
            label = 0 if row['breast_birads'] in ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3'] else -1 # Use -1 for uncertain

            if label != -1:
                 image_file_path = os.path.join(original_data_path, 'images', study_id, f'{image_id}.png')
                 if os.path.exists(image_file_path):
                     image_info_list.append({
                         'image_id': image_id,
                         'study_id': study_id,
                         'label': label,
                         'original_split': original_split,
                         'file_path': image_file_path,
                         'breast_birads': row.get('breast_birads'),
                         'breast_density': row.get('breast_density'),
                     })
                 else:
                     print(f"Warning: Image file not found for image {image_id} at {image_file_path}. Skipping.")
            else:
                 print(f"Warning: Uncertain label for image {image_id} based on breast-level BI-RADS {row['breast_birads']}. Skipping.")

    elif finding_df is not None:
         print("Proceeding with finding-level metadata only. Benign/normal cases might be missed.")
         # In this case, we only have info about images with findings
         for index, row in finding_df.iterrows():
             image_id = row['image_id']
             study_id = row['study_id']
             original_split = row['split']
             label = int(row['is_malignant_finding']) # Label is 1 if malignant finding, 0 otherwise

             image_file_path = os.path.join(original_data_path, 'images', study_id, f'{image_id}.png')
             if os.path.exists(image_file_path):
                 image_info_list.append({
                     'image_id': image_id,
                     'study_id': study_id,
                     'label': label,
                     'original_split': original_split,
                     'file_path': image_file_path,
                     # Breast-level info is not available here
                 })
             else:
                 print(f"Warning: Image file not found for image {image_id} at {image_file_path}. Skipping.")


    else:
        print("Error: Neither breast-level nor finding-level metadata files were found.")


    print(f"Finished metadata processing. Found {len(image_info_list)} relevant images with labels.")
    return image_info_list

# --- Function to Split Data Based on Original Split and Ratios ---
def split_vindr_mammo_data(image_info_list, train_ratio_orig_train, val_ratio_orig_train):
    """
    Splits the image info list into train, validation, and test sets,
    respecting the original 'split' column and applying ratios to the original training data.
    Returns a dictionary mapping split names to lists of dictionaries.
    """
    print("Splitting VinDr-Mammo data...")

    # Separate based on original split
    original_training_data = [item for item in image_info_list if item['original_split'] == 'training']
    original_test_data = [item for item in image_info_list if item['original_split'] == 'test']

    print(f"Original training data count: {len(original_training_data)}")
    print(f"Original test data count: {len(original_test_data)}")

    train_split_info = []
    val_split_info = []

    # Split original training data into our train and validation sets
    if original_training_data:
        # Extract file paths and labels for splitting
        orig_train_files = [item['file_path'] for item in original_training_data]
        orig_train_labels = [item['label'] for item in original_training_data]

        # Perform the split
        # Ensure there are enough samples for stratification and splitting
        if len(set(orig_train_labels)) < 2 or len(orig_train_labels) < 2:
             print("Warning: Not enough samples or classes in original training data for stratification. Splitting without stratify.")
             files_train, files_val, labels_train, labels_val = train_test_split(
                orig_train_files, orig_train_labels,
                test_size=(val_ratio_orig_train / (train_ratio_orig_train + val_ratio_orig_train)),
                random_state=42
            )
        else:
            files_train, files_val, labels_train, labels_val = train_test_split(
                orig_train_files, orig_train_labels,
                test_size=(val_ratio_orig_train / (train_ratio_orig_train + val_ratio_orig_train)), # Calculate test_size relative to the subset
                stratify=orig_train_labels,
                random_state=42
            )

        # Reconstruct the list of dictionaries for the splits
        # Need to look up the original info based on file_path
        original_info_map = {item['file_path']: item for item in original_training_data}

        train_split_info = [original_info_map[f] for f in files_train]
        val_split_info = [original_info_map[f] for f in files_val]

    else:
        print("Warning: No original training data found. Train and validation sets will be empty.")


    # The original test data becomes our test set
    # We only need file_path and label for the organized structure
    test_split_info = [{'file_path': item['file_path'], 'label': item['label'], 'image_id': item['image_id'], 'study_id': item['study_id']} for item in original_test_data]


    splits = {
        'train': train_split_info,
        'validation': val_split_info,
        'test': test_split_info
    }

    print(f"Splitting complete. Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
    return splits

# --- Function to Organize Split Files and Generate Metadata ---
def organize_split_files_and_generate_metadata(split_data, organized_base_path, metadata_output_path):
    """
    Organizes the split data (list of dictionaries) into the target directory structure
    and generates a metadata CSV file for the organized data.
    """
    print(f"Organizing split files into: {organized_base_path}")

    all_organized_metadata = []

    for split_name, data_list in split_data.items():
        split_dir = os.path.join(organized_base_path, split_name)
        benign_split_dir = os.path.join(split_dir, 'benign')
        malignant_split_dir = os.path.join(split_dir, 'malignant')

        os.makedirs(benign_split_dir, exist_ok=True)
        os.makedirs(malignant_split_dir, exist_ok=True)

        print(f"Copying files for {split_name}...")
        for image_info in data_list:
            original_file_path = image_info['file_path']
            label = image_info['label']
            image_id = image_info['image_id']
            study_id = image_info['study_id'] # Keep study_id for metadata

            # Determine the target directory
            target_dir = benign_split_dir if label == 0 else malignant_split_dir
            # Use original image_id as the new file name
            new_file_name = f'{image_id}.png' # Assuming .png, adjust if needed
            organized_file_path = os.path.join(target_dir, new_file_name)

            try:
                shutil.copy(original_file_path, organized_file_path)

                # Add information to the metadata list
                metadata_entry = {
                    'image_id': image_id,
                    'study_id': study_id,
                    'label': label,
                    'split': split_name, # This is our new split (train, val, test)
                    'organized_file_path': organized_file_path,
                    'original_file_path': original_file_path,
                    # Include other relevant info from original metadata if available
                    'breast_birads': image_info.get('breast_birads'),
                    'breast_density': image_info.get('breast_density'),
                    # Add path to segmentation mask if available and organized separately
                    # 'segmentation_mask_path': 'path/to/mask.png' # TODO: Add logic for mask paths
                }
                all_organized_metadata.append(metadata_entry)

            except FileNotFoundError:
                 print(f"Error: Source file not found during copy: {original_file_path}")
            except Exception as e:
                 print(f"Error copying file {original_file_path} to {organized_file_path}: {e}")


        print(f"Finished copying for {split_name}. Total files: {len(data_list)}")

    # Generate the metadata CSV
    if all_organized_metadata:
        metadata_df = pd.DataFrame(all_organized_metadata)
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)
        metadata_df.to_csv(metadata_output_path, index=False)
        print(f"\nGenerated organized dataset metadata file: {metadata_output_path}")
    else:
        print("\nNo metadata entries to write. Metadata file not generated.")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Process Metadata and Get Image Info
    vindr_mammo_image_info_list = get_image_labels_and_paths(
        VINDR_MAMMO_ORIGINAL_PATH,
        BREAST_LEVEL_METADATA_PATH,
        FINDING_METADATA_PATH
    )

    if vindr_mammo_image_info_list:
        # 2. Split Data
        vindr_mammo_splits = split_vindr_mammo_data(
            vindr_mammo_image_info_list,
            TRAIN_RATIO_ORIGINAL_TRAINING,
            VAL_RATIO_ORIGINAL_TRAINING
        )

        # 3. Organize Split Files and Generate Metadata
        organize_split_files_and_generate_metadata(
            vindr_mammo_splits,
            ORGANIZED_DATASET_BASE_PATH,
            ORGANIZED_METADATA_FILE
        )
        print("\nVinDr-Mammo data preparation complete.")
    else:
        print("\nVinDr-Mammo data preparation failed due to issues processing metadata or finding images.")

    # Note: This script only handles VinDr-Mammo.
    # You would run similar scripts for BUSI and KAU-BCMD,
    # and then a separate script to combine/upload to Hugging Face.

