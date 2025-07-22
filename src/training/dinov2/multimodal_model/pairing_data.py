# training/multimodal/pairing_data.py

import os
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.model_selection import train_test_split

def setup_logging():
    """Sets up basic logging for the script."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_image_paths_from_folders(root_dir):
    """
    Scans a directory structured like ImageFolder and returns lists of
    paths for each class.
    
    Args:
        root_dir (str): The root directory of the dataset (e.g., ../../data/mammo).

    Returns:
        dict: A dictionary where keys are class names (e.g., 'benign') and
              values are lists of full image paths.
    """
    paths_dict = {}
    # Iterate over train, validation, and test splits
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            logging.warning(f"Directory not found: {split_path}. Skipping.")
            continue
            
        # Iterate over class folders (benign, malignant)
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                if class_name not in paths_dict:
                    paths_dict[class_name] = []
                
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                        paths_dict[class_name].append(os.path.join(class_path, fname))
                        
    logging.info(f"Found {len(paths_dict.get('benign', []))} benign and {len(paths_dict.get('malignant', []))} malignant images in {root_dir}")
    return paths_dict

def create_paired_dataframe(mammo_paths, us_paths):
    """
    Creates synthetically paired data from two unimodal datasets,
    balancing them by the minority class count.

    Args:
        mammo_paths (dict): Dictionary of mammography image paths by class.
        us_paths (dict): Dictionary of ultrasound image paths by class.

    Returns:
        pd.DataFrame: A dataframe containing the paired paths and their labels.
    """
    all_pairs = []
    
    for label in ['benign', 'malignant']:
        logging.info(f"Pairing images for class: '{label}'")
        
        mammo_list = mammo_paths.get(label, [])
        us_list = us_paths.get(label, [])
        
        if not mammo_list or not us_list:
            logging.warning(f"Skipping class '{label}' due to missing images in one or both modalities.")
            continue

        # Determine the number of pairs based on the smaller dataset for this class
        num_pairs = min(len(mammo_list), len(us_list))
        logging.info(f"Can create a maximum of {num_pairs} pairs for class '{label}'.")
        
        # Randomly shuffle and truncate the lists to the same size
        np.random.shuffle(mammo_list)
        np.random.shuffle(us_list)
        
        mammo_list = mammo_list[:num_pairs]
        us_list = us_list[:num_pairs]
        
        # Create pairs
        for i in range(num_pairs):
            all_pairs.append({
                'mammo_path': mammo_list[i],
                'ultrasound_path': us_list[i],
                'label': label
            })
            
    return pd.DataFrame(all_pairs)

def main():
    "Main function to generate and save the paired dataset."
    parser = argparse.ArgumentParser(description="Create a synthetically paired dataset for multimodal training.")
    parser.add_argument('--mammo_dir', type=str, default= "/teamspace/studios/this_studio/mmibc/mammo", help='Root directory of the mammography dataset.')
    parser.add_argument('--us_dir', type=str, default="/teamspace/studios/this_studio/mmibc/ultrasound/images", help='Root directory of the ultrasound dataset.')
    parser.add_argument('--output_csv', type=str, default='/teamspace/studios/this_studio/mmibc/multimodal_pairs.csv', help='Path to save the output CSV file.')
    parser.add_argument('--test_size', type=float, default=0.15, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--val_size', type=float, default=0.15, help='Proportion of the dataset to include in the validation split.')
    
    args = parser.parse_args()
    setup_logging()

    # Step 1: Index all image paths
    logging.info("Indexing mammography images...")
    mammo_image_paths = get_image_paths_from_folders(args.mammo_dir)
    
    logging.info("Indexing ultrasound images...")
    us_image_paths = get_image_paths_from_folders(args.us_dir)
    
    # Step 2: Create the initial paired dataframe
    paired_df = create_paired_dataframe(mammo_image_paths, us_image_paths)
    
    if paired_df.empty:
        logging.error("No pairs were created. Please check your data directories and class folder names.")
        return
        
    logging.info(f"Successfully created a total of {len(paired_df)} balanced pairs.")

    # Step 3: Split the data into train, validation, and test sets
    labels = paired_df['label']
    
    # First, split into train and a temporary set (val + test)
    # stratify ensures the class distribution is the same across splits
    train_df, temp_df = train_test_split(
        paired_df, 
        test_size=(args.test_size + args.val_size),
        random_state=42,
        stratify=labels
    )
    
    # Calculate the proportion of the validation set relative to the temporary set
    relative_val_size = args.val_size / (args.test_size + args.val_size)
    
    # Split the temporary set into validation and test sets
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val_size),
        random_state=42,
        stratify=temp_df['label']
    )
    
    # Assign the split names
    train_df['split'] = 'train'
    val_df['split'] = 'validation'
    test_df['split'] = 'test'
    
    # Step 4: Combine and save the final dataframe
    final_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    
    logging.info("Data splitting complete:")
    logging.info(f"\n{final_df['split'].value_counts()}")
    
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    final_df.to_csv(args.output_csv, index=False)
    logging.info(f"Paired dataset metadata saved successfully to {args.output_csv}")

if __name__ == '__main__':
    main()
