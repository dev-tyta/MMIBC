# training/utils/prepare_kaubcmd.py

import os
import shutil
import glob
import logging
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm

def setup_logging():
    """Sets up basic logging for the script."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def copy_files(files, dest_dir):
    """Copies a list of files to a destination directory with a progress bar."""
    os.makedirs(dest_dir, exist_ok=True)
    for f in tqdm(files, desc=f"Copying to {os.path.basename(dest_dir)}"):
        shutil.copy(f, dest_dir)

def main():
    """
    Main function to process the KAU-BCMD dataset. It maps BIRADS folders
    to benign/malignant classes and splits them into train/validation/test sets.
    """
    parser = argparse.ArgumentParser(description="Organize the KAU-BCMD dataset by splitting BIRADS folders into a standard train/val/test structure.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the root of the raw KAU-BCMD dataset directory (the one containing birads1, birads3, etc.).')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to where the new, split dataset will be created.')
    parser.add_argument('--test_size', type=float, default=0.15, help='Proportion of the dataset for the test split.')
    parser.add_argument('--val_size', type=float, default=0.15, help='Proportion of the dataset for the validation split.')
    
    args = parser.parse_args()
    setup_logging()
    
    if os.path.exists(args.output_dir):
        logging.warning(f"Output directory '{args.output_dir}' already exists. Files may be overwritten.")
    else:
        os.makedirs(args.output_dir)

    logging.info("Starting KAU-BCMD dataset processing...")

    # --- This mapping correctly targets the KAU-BCMD folder structure ---
    birads_map = {
        'benign': ['birads1', 'birads3'],
        'malignant': ['birads4', 'birads5']
    }
    logging.info(f"Using BIRADS mapping based on your directory: {birads_map}")

    # --- Collect all image paths for each super-class ---
    all_files = {'benign': [], 'malignant': []}
    for super_class, birads_folders in birads_map.items():
        for birads_folder in birads_folders:
            folder_path = os.path.join(args.input_dir, birads_folder)
            if os.path.isdir(folder_path):
                logging.info(f"Scanning folder: {folder_path}")
                # Use glob to find all image files, assuming common extensions
                files_in_folder = glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]")) + \
                                  glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                                  glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]"))
                all_files[super_class].extend(files_in_folder)
            else:
                logging.warning(f"BIRADS folder not found, skipping: {folder_path}")

    logging.info(f"Total files found: Benign - {len(all_files['benign'])}, Malignant - {len(all_files['malignant'])}")

    if not all_files['benign'] and not all_files['malignant']:
        logging.error("No image files were found. Please check the input directory and that it contains the birads subfolders.")
        return

    # --- Split and copy files for each class ---
    for class_name, file_list in all_files.items():
        if not file_list:
            logging.warning(f"No files to process for class '{class_name}'. Skipping.")
            continue

        logging.info(f"Splitting {len(file_list)} files for class '{class_name}'...")
        train_files, temp_files = train_test_split(
            file_list, 
            test_size=(args.test_size + args.val_size),
            random_state=42,
            shuffle=True
        )
        
        relative_val_size = args.val_size / (args.test_size + args.val_size)
        
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(1 - relative_val_size),
            random_state=42,
            shuffle=True
        )
        
        logging.info(f"Split results for '{class_name}': {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test.")

        # Copy files to their new locations
        copy_files(train_files, os.path.join(args.output_dir, 'train', class_name))
        copy_files(val_files, os.path.join(args.output_dir, 'validation', class_name))
        copy_files(test_files, os.path.join(args.output_dir, 'test', class_name))

    logging.info("="*30)
    logging.info("KAU-BCMD dataset processing complete!")
    logging.info(f"The organized dataset is ready at: {args.output_dir}")
    logging.info("You can now use this path for the `--mammo_dir` argument in the pairing_data.py script.")
    logging.info("="*30)

if __name__ == '__main__':
    main()
