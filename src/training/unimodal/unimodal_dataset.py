# training/unimodal/unimodal_dataset.py

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import logging

class UltrasoundDataset:
    """
    A wrapper class for the Ultrasound dataset that uses torchvision's ImageFolder.
    
    This class simplifies the process of creating data loaders for the training script.
    It assumes the data is structured as:
    - data_path/
        - train/
            - benign/
            - malignant/
        - validation/
            - benign/
            - malignant/
    """
    def __init__(self, data_path, image_size=224):
        """
        Args:
            data_path (str): The root path to the ultrasound images directory.
            image_size (int): The size to which images will be resized.
        """
        self.data_path = data_path
        self.image_size = image_size
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        """Defines the image transformations required for the DINOv2 model."""
        # DINOv2 requires specific normalization constants
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_train_dataset(self):
        """Returns the training dataset object."""
        train_path = os.path.join(self.data_path, 'train')
        if not os.path.isdir(train_path):
            raise FileNotFoundError(f"Training directory not found at: {train_path}")
        return datasets.ImageFolder(root=train_path, transform=self.transform)
        
    def get_validation_dataset(self):
        """Returns the validation dataset object."""
        val_path = os.path.join(self.data_path, 'validation')
        if not os.path.isdir(val_path):
            raise FileNotFoundError(f"Validation directory not found at: {val_path}")
        return datasets.ImageFolder(root=val_path, transform=self.transform)

class MammographyDataset(Dataset):
    """
    A custom PyTorch Dataset for the Mammography dataset that reads from a CSV.
    This version includes the '.targets' attribute needed for weighted sampling.
    """
    def __init__(self, csv_file, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        
        logging.info(f"Loading mammography metadata from: {csv_file} for split: '{split}'")
        metadata = pd.read_csv(csv_file)
        
        self.image_metadata = metadata[metadata['split'] == split].reset_index(drop=True)

        # --- FIX: Create the 'targets' attribute ---
        # The WeightedRandomSampler requires this attribute.
        # We create it from the 'label' column in our metadata.
        self.targets = self.image_metadata['label'].tolist()
        
        self.classes = sorted(self.image_metadata['label'].unique())

        logging.info(f"Loaded {len(self.image_metadata)} samples for the '{split}' mammography split.")

    def __len__(self):
        return len(self.image_metadata)

    def __getitem__(self, idx):
        # Assumes the CSV has a column with the direct path to the image
        img_path = self.image_metadata.loc[idx, 'organized_file_path']
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logging.error(f"Image not found at {img_path}. Index: {idx}.")
            return None, None 

        label = self.image_metadata.loc[idx, 'label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_mammo_transforms(image_size=224):
    """
    Defines separate training and validation transforms for the Mammography dataset.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def get_medical_transforms(image_size=224):
    """
    Defines separate training and validation transforms with augmentations
    suitable for medical imaging.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation and Test transforms are the same (no augmentation)
    eval_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, eval_transforms

# training/unimodal/unimodal_dataset.py

import os
import logging
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import cv2
import numpy as np
import torch

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NEW: Custom Transform for CLAHE ---
class ApplyCLAHE:
    """
    A custom transform to apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to a PIL image. This is a powerful technique for enhancing local contrast in
    medical images like mammograms.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def __call__(self, img):
        # Convert PIL image to numpy array (grayscale for CLAHE)
        img_np = np.array(img.convert('L'))
        # Apply CLAHE
        cl_img = self.clahe.apply(img_np)
        # Convert back to a 3-channel PIL image to be compatible with pre-trained models
        return Image.fromarray(cl_img).convert('RGB')

def get_advanced_mammo_transforms(image_size=224):
    """
    Defines separate training and validation transforms for Mammography,
    including advanced, domain-specific augmentations.
    """
    logging.info("Setting up ADVANCED transformations with CLAHE and Elastic Deformations for Mammography.")
    
    # --- Training transforms with advanced techniques ---
    train_transforms = transforms.Compose([
        ApplyCLAHE(), # Apply CLAHE first to enhance contrast before other transforms
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        # --- NEW: Elastic Transform for realistic tissue deformation ---
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        # --- NEW: Add Gaussian Noise for robustness ---
        transforms.ToTensor(), # Convert to tensor before adding noise
        transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)), # Add Gaussian noise
        transforms.ToPILImage(), # Convert back to PIL for normalization transform
        transforms.ToTensor(), # Convert back to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Validation transforms also get CLAHE for consistency ---
    val_transforms = transforms.Compose([
        ApplyCLAHE(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

class ImageFolderDataset:
    """
    A generic wrapper class for datasets structured like ImageFolder.
    It now uses a 'transform_func' argument to allow for different pipelines.
    """
    def __init__(self, data_path, image_size=224, transform_func=get_advanced_mammo_transforms):
        self.data_path = data_path
        self.image_size = image_size
        logging.info(f"Initializing ImageFolderDataset with data path: {data_path}")
        
        self.train_transform, self.eval_transform = transform_func(image_size)
        
    def get_train_dataset(self):
        """Returns the training dataset object with training transforms."""
        train_path = os.path.join(self.data_path, 'train')
        return datasets.ImageFolder(root=train_path, transform=self.train_transform)
        
    def get_validation_dataset(self):
        """Returns the validation dataset object with evaluation transforms."""
        val_path = os.path.join(self.data_path, 'validation')
        return datasets.ImageFolder(root=val_path, transform=self.eval_transform)
        
    def get_test_dataset(self):
        """Returns the test dataset object with evaluation transforms."""
        test_path = os.path.join(self.data_path, 'test')
        return datasets.ImageFolder(root=test_path, transform=self.eval_transform)
