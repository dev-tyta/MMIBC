# training/multimodal/multimodal_dataset.py

import os
import pandas as pd
import logging
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import cv2
import numpy as np

# This is the same advanced augmentation pipeline from our unimodal training.
# We will apply it to both modalities for consistency.
def get_advanced_medical_transforms(image_size=224):
    """
    Defines separate training and validation transforms for medical images,
    including advanced augmentations like CLAHE and Elastic Deformations.
    """
    class ApplyCLAHE:
        def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        def __call__(self, img):
            img_np = np.array(img.convert('L'))
            cl_img = self.clahe.apply(img_np)
            return Image.fromarray(cl_img).convert('RGB')

    train_transforms = transforms.Compose([
        ApplyCLAHE(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        ApplyCLAHE(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, eval_transforms

class MultimodalDataset(Dataset):
    """
    A custom PyTorch Dataset for loading paired mammography and ultrasound images
    from a metadata CSV file.
    """
    def __init__(self, csv_file, split='train', image_size=224):
        """
        Args:
            csv_file (str): Path to the multimodal_pairs.csv file.
            split (str): The dataset split to load ('train', 'validation', or 'test').
            image_size (int): The size to which images will be resized.
        """
        logging.info(f"Loading multimodal metadata from: {csv_file} for split: '{split}'")
        self.metadata = pd.read_csv(csv_file)
        self.split_df = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        
        train_transforms, eval_transforms = get_advanced_medical_transforms(image_size)
        self.transform = train_transforms if split == 'train' else eval_transforms
        
        self.label_map = {'benign': 0, 'malignant': 1}
        logging.info(f"Loaded {len(self.split_df)} pairs for the '{split}' split.")

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        
        mammo_path = row['mammo_path']
        us_path = row['ultrasound_path']
        label_str = row['label']
        label = self.label_map[label_str]
        
        try:
            mammo_img = Image.open(mammo_path).convert("RGB")
            us_img = Image.open(us_path).convert("RGB")
        except FileNotFoundError as e:
            logging.error(f"Image not found at index {idx}: {e}")
            return None, None, None

        # Apply transformations
        mammo_tensor = self.transform(mammo_img)
        us_tensor = self.transform(us_img)
            
        return mammo_tensor, us_tensor, torch.tensor(label, dtype=torch.long)
