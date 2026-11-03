# training/utils/prepare_bus_uc.py

import os
import shutil
import glob
import logging
from sklearn.model_selection import train_test_split
import argparse

def setup_logging():
    """Sets up basic logging for the script."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_and_copy_files(file_list, class_name, output_dir):
    """
    Splits a list of files into train, validation, and test sets and copies them
    to the appropriate destination folders.
    """
    # Create a reproducible 70/15/15 split
    train_files, temp_files = train_test_split(file_list, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42) # 0.5 * 0.3 = 0.15

    logging.info(f"Splitting class '{class_name}': {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test.")

    # Function to copy files to their new home
    def copy_files(files, split_name):
        dest_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            shutil.copy(f, dest_dir)
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'validation')
    copy_files(test_files, 'test')

def main():
    """Main function to organize the BUS_UC dataset."""
    parser = argparse.ArgumentParser(description="Split the BUS_UC dataset into train, validation, and test sets.")
    parser.add_argument('--input_dir', type=str, default="/teamspace/studios/this_studio/data/BUS_UC/BUS_UC/BUS_UC", required=True, help='Path to the root of the raw BUS_UC dataset directory.')
    parser.add_argument('--output_dir', type=str, default="mmibc/paired/ultrasound", required=True, help='Path to where the new, split dataset will be created.')
    args = parser.parse_args()
    
    setup_logging()
    
    if os.path.exists(args.output_dir):
        logging.warning(f"Output directory '{args.output_dir}' already exists. Files may be overwritten.")
    else:
        os.makedirs(args.output_dir)

    logging.info("Starting dataset splitting process...")

    # Define paths based on the BUS_UC structure
    benign_path = os.path.join(args.input_dir, "Benign", "images")
    malignant_path = os.path.join(args.input_dir, "Malignant", "images")
    
    # Get all image files
    benign_files = glob.glob(os.path.join(benign_path, "*.png"))
    malignant_files = glob.glob(os.path.join(malignant_path, "*.png"))

    if not benign_files or not malignant_files:
        logging.error("Could not find image files. Please check the input directory structure.")
        logging.error(f"Searched in '{benign_path}' and '{malignant_path}'.")
        return
        
    # Process each class
    split_and_copy_files(benign_files, 'benign', args.output_dir)
    split_and_copy_files(malignant_files, 'malignant', args.output_dir)
    
    logging.info("="*30)
    logging.info("Dataset splitting and copying complete!")
    logging.info(f"The organized dataset is ready at: {args.output_dir}")
    logging.info("You can now use this path for the `--us_dir` argument in the pairing_data.py script.")
    logging.info("="*30)

if __name__ == '__main__':
    main()
