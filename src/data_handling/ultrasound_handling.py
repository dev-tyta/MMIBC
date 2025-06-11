import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Set the base path for your downloaded BUSI dataset
# Assuming the structure is ./data/ultrasound/images/... and ./data/ultrasound/masks/...
BUSI_ORIGINAL_PATH = os.path.expanduser('~/data/ultrasound') # Adjust this path if needed
BUSI_IMAGES_PATH = os.path.join(BUSI_ORIGINAL_PATH, 'images')
BUSI_MASKS_PATH = os.path.join(BUSI_ORIGINAL_PATH, 'masks')


# Set the base path for the new organized dataset structure
ORGANIZED_DATASET_BASE_PATH = os.path.expanduser('~/mmibc/ultrasound') # Specific path for BUSI
ORGANIZED_METADATA_FILE = os.path.join(ORGANIZED_DATASET_BASE_PATH, 'busi_metadata.csv') # Output metadata file

# Define split ratios (train, validation, test) - should sum to 1.0
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# --- Function to Get Image Labels and Paths for BUSI ---
def get_image_labels_and_paths_busi(images_base_path, masks_base_path):
    """
    Reads BUSI dataset structure (assuming images/masks subfolders with
    benign, malignant, normal subfolders inside) and assigns binary labels
    (0 for benign/normal, 1 for malignant).
    Correctly finds mask files based on '_mask' naming convention.
    Returns a list of dictionaries: [{'image_id': '...', 'label': 0/1,
                                       'image_file_path': '...', 'mask_file_path': '...',
                                       'original_class': '...'}]
    """
    print("Processing BUSI dataset structure...")

    image_info_list = [] # List to store info for each image

    # Define the original class directories and their target binary labels
    class_directories = {
        'benign': 0,    # Map benign to binary label 0
        'normal': 0,    # Map normal to binary label 0
        'malignant': 1  # Map malignant to binary label 1
    }

    for class_name, binary_label in class_directories.items():
        class_images_dir_path = os.path.join(images_base_path, class_name)
        class_masks_dir_path = os.path.join(masks_base_path, class_name)


        if not os.path.exists(class_images_dir_path):
            print(f"Warning: Image directory not found: {class_images_dir_path}. Skipping this class.")
            continue
        # Note: We don't skip the class if mask directory is missing, just note it.
        if not os.path.exists(class_masks_dir_path):
             print(f"Warning: Mask directory not found: {class_masks_dir_path}. Segmentation masks might be missing for this class.")


        print(f"Processing class: {class_name} (Binary Label: {binary_label})")

        # List files in the class image directory
        try:
            image_files = [f for f in os.listdir(class_images_dir_path) if not f.startswith('.') and os.path.isfile(os.path.join(class_images_dir_path, f))]
        except Exception as e:
            print(f"Error listing files in {class_images_dir_path}: {e}")
            continue

        for file_name in image_files:
            image_file_path = os.path.join(class_images_dir_path, file_name)

            # Construct the expected mask filename by appending '_mask' before the extension
            name, ext = os.path.splitext(file_name)
            mask_file_name = f"{name}_mask{ext}"
            mask_file_path = os.path.join(class_masks_dir_path, mask_file_name)

            # Check if the constructed mask file path actually exists
            if not os.path.exists(mask_file_path):
                 # Handle cases like 'benign (4)_mask_1.png' if they exist alongside 'benign (4)_mask.png'
                 # This might require more sophisticated logic if there are multiple masks per image.
                 # For now, we'll just check for the primary '_mask' version.
                 # If the primary mask isn't found, check for '_mask_1', '_mask_2', etc.
                 # This is a common pattern in BUSI for multiple masks per image.
                 # Let's refine this to find *all* masks associated with an image ID.

                 associated_masks = []
                 if os.path.exists(class_masks_dir_path):
                     mask_files_in_dir = os.listdir(class_masks_dir_path)
                     # Find all files in the mask directory that start with the image name
                     # This is a more robust way to find associated masks
                     base_name = os.path.splitext(file_name)[0]
                     associated_mask_files = [
                         f for f in mask_files_in_dir
                         if f.startswith(base_name) and '_mask' in f and os.path.isfile(os.path.join(class_masks_dir_path, f))
                     ]
                     associated_masks = [os.path.join(class_masks_dir_path, f) for f in associated_mask_files]


                 if not associated_masks:
                     print(f"Warning: No mask file found for image {file_name} in {class_masks_dir_path}. Mask path will be None.")
                     mask_file_path_to_record = None # No mask found
                 elif len(associated_masks) > 1:
                      print(f"Warning: Found multiple masks for image {file_name}: {associated_masks}. Recording the first one found.")
                      mask_file_path_to_record = associated_masks[0] # Or decide how to handle multiple masks
                 else:
                      mask_file_path_to_record = associated_masks[0] # Only one mask found

            else:
                 # The primary '_mask' file was found
                 mask_file_path_to_record = mask_file_path


            # Use file_name as a simple image_id for BUSI
            image_id = file_name

            image_info_list.append({
                'image_id': image_id,
                'label': binary_label,
                'image_file_path': image_file_path,
                'mask_file_path': mask_file_path_to_record, # Include the found mask path (or None)
                'original_class': class_name # Keep original class name for metadata
            })

    print(f"Finished processing BUSI structure. Found {len(image_info_list)} relevant images with binary labels.")
    return image_info_list

# --- Function to Split Data ---
def split_busi_data(image_info_list, train_ratio, val_ratio, test_ratio):
    """
    Splits the image info list into train, validation, and test sets,
    stratifying by the binary label.
    Returns a dictionary mapping split names to lists of dictionaries.
    """
    print("Splitting BUSI data...")

    if not image_info_list:
        print("No image info found to split.")
        return {'train': [], 'validation': [], 'test': []}

    # Extract binary labels for splitting
    all_labels = [item['label'] for item in image_info_list]

    # Ensure there are enough samples and classes for stratification
    if len(set(all_labels)) < 2 or len(all_labels) < 2:
         print("Warning: Not enough samples or classes for stratification. Splitting without stratify.")
         train_info, temp_info = train_test_split(
            image_info_list, test_size=(val_ratio + test_ratio), random_state=42
        )
    else:
        # Perform the split using the list of dictionaries directly
        train_info, temp_info = train_test_split(
            image_info_list, test_size=(val_ratio + test_ratio), stratify=all_labels, random_state=42
        )

    # Calculate test_size for the second split relative to the temporary set
    test_size_temp = val_ratio + test_ratio
    test_size_final = test_ratio / test_size_temp if test_size_temp > 0 else 0

    # Extract labels from the temporary split for stratification
    temp_labels = [item['label'] for item in temp_info]

    if test_size_final > 0 and (len(set(temp_labels)) < 2 or len(temp_labels) < 2):
         print("Warning: Not enough samples or classes in temporary split for second stratification. Splitting without stratify.")
         val_info, test_info = train_test_split(
            temp_info, test_size=test_size_final, random_state=42
        )
    else:
        val_info, test_info = train_test_split(
            temp_info, test_size=test_size_final, stratify=temp_labels, random_state=42
        )

    splits = {
        'train': train_info,
        'validation': val_info,
        'test': test_info
    }

    print(f"Splitting complete. Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
    return splits

# --- Function to Organize Split Files and Generate Metadata ---
def organize_split_files_and_generate_metadata_busi(split_data, organized_base_path, metadata_output_path):
    """
    Organizes the split data (list of dictionaries) into the target directory structure
    and generates a metadata CSV file for the organized data for BUSI.
    Copies both images and masks.
    """
    print(f"Organizing split files into: {organized_base_path}")

    all_organized_metadata = []

    # Create base organized directories
    organized_images_base = os.path.join(organized_base_path, 'images')
    organized_masks_base = os.path.join(organized_base_path, 'masks')
    os.makedirs(organized_images_base, exist_ok=True)
    os.makedirs(organized_masks_base, exist_ok=True)


    for split_name, data_list in split_data.items():
        # Create split directories within images and masks
        split_images_dir = os.path.join(organized_images_base, split_name)
        split_masks_dir = os.path.join(organized_masks_base, split_name)

        # Create class directories within each split directory
        benign_images_dir = os.path.join(split_images_dir, 'benign')
        malignant_images_dir = os.path.join(split_images_dir, 'malignant')
        benign_masks_dir = os.path.join(split_masks_dir, 'benign')
        malignant_masks_dir = os.path.join(split_masks_dir, 'malignant')


        os.makedirs(benign_images_dir, exist_ok=True)
        os.makedirs(malignant_images_dir, exist_ok=True)
        os.makedirs(benign_masks_dir, exist_ok=True)
        os.makedirs(malignant_masks_dir, exist_ok=True)


        print(f"Copying files for {split_name}...")
        for image_info in data_list:
            original_image_file_path = image_info['image_file_path']
            original_mask_file_path = image_info['mask_file_path'] # This is the path found in get_image_labels_and_paths_busi
            label = image_info['label']
            image_id = image_info['image_id'] # Use original file name as image_id
            original_class = image_info['original_class'] # Keep original class name

            # Determine the target directory for image and mask
            target_images_dir = benign_images_dir if label == 0 else malignant_images_dir
            target_masks_dir = benign_masks_dir if label == 0 else malignant_masks_dir

            # Use original image_id as the new file name for the image
            new_image_file_name = image_id
            organized_image_file_path = os.path.join(target_images_dir, new_image_file_name)

            # For the mask, use the original mask file name if available, otherwise use the image_id + _mask
            if original_mask_file_path:
                 new_mask_file_name = os.path.basename(original_mask_file_path)
            else:
                 # If no mask was found, we won't copy one, but we could define a potential name
                 # for consistency in metadata, though it won't exist on disk.
                 # Let's just record None for the organized path if original was None.
                 new_mask_file_name = None

            organized_mask_file_path = os.path.join(target_masks_dir, new_mask_file_name) if new_mask_file_name else None


            try:
                # Copy the image
                shutil.copy(original_image_file_path, organized_image_file_path)

                # Copy the mask if it exists
                if original_mask_file_path and os.path.exists(original_mask_file_path):
                     if organized_mask_file_path is not None:
                         shutil.copy(original_mask_file_path, organized_mask_file_path)
                else:
                     organized_mask_file_path = None # Ensure organized path is None if mask wasn't copied


                # Add information to the metadata list
                metadata_entry = {
                    'image_id': image_id,
                    'label': label, # Binary label (0 or 1)
                    'original_class': original_class, # Original class (benign, malignant, normal)
                    'split': split_name, # This is our new split (train, val, test)
                    'organized_image_file_path': organized_image_file_path,
                    'original_image_file_path': original_image_file_path,
                    'organized_mask_file_path': organized_mask_file_path, # Path to the copied mask (or None)
                    'original_mask_file_path': original_mask_file_path # Path to the original mask (or None)
                }
                all_organized_metadata.append(metadata_entry)

            except FileNotFoundError:
                 print(f"Error: Source file not found during copy: {original_image_file_path} or {original_mask_file_path}")
            except Exception as e:
                 print(f"Error copying file {original_image_file_path} or its mask: {e}")


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
    # 1. Process BUSI dataset structure and Get Image Info
    busi_image_info_list = get_image_labels_and_paths_busi(
        BUSI_IMAGES_PATH, # Pass the images subfolder path
        BUSI_MASKS_PATH   # Pass the masks subfolder path
    )

    if busi_image_info_list:
        # 2. Split Data
        busi_splits = split_busi_data(
            busi_image_info_list,
            TRAIN_RATIO,
            VAL_RATIO,
            TEST_RATIO
        )

        # 3. Organize Split Files and Generate Metadata
        organize_split_files_and_generate_metadata_busi(
            busi_splits,
            ORGANIZED_DATASET_BASE_PATH,
            ORGANIZED_METADATA_FILE
        )
        print("\nBUSI data preparation complete.")
    else:
        print("\nBUSI data preparation failed due to issues processing dataset structure or finding images.")

    # Note: This script only handles BUSI.
    # You would run the VinDr-Mammo script separately,
    # handle KAU-BCMD, and then use a separate script to combine/upload to Hugging Face.