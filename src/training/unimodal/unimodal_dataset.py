# training/unimodal/unimodal_dataset.py

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

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
    A custom PyTorch Dataset for the Mammography dataset.
    
    This class is designed to work with a metadata CSV file that links
    image filenames to their labels, as is common with medical datasets.
    
    Args:
        csv_file (str): Path to the metadata CSV file.
        root_dir (str): Path to the directory containing the image folders (e.g., 'train', 'test').
        split (str): The dataset split to load ('train', 'validation', or 'test').
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, csv_file, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split) # e.g., ../../data/mammo/train
        self.transform = transform
        
        # Load the metadata
        metadata = pd.read_csv(csv_file)
        
        # Filter the dataframe for the correct split
        # This assumes your CSV has a 'split' column. If not, you'll need to adapt.
        self.image_metadata = metadata[metadata['split'] == split].reset_index()

        print(f"Loaded {len(self.image_metadata)} samples for the '{split}' mammography split.")

    def __len__(self):
        return len(self.image_metadata)

    def __getitem__(self, idx):
        # Construct the full image path. Assumes CSV has an 'image_id' column.
        # You might need to adjust this to match your CSV's column names.
        img_name = os.path.join(self.root_dir, f"{self.image_metadata.loc[idx, 'image_id']}.png") # Assumes .png format
        
        try:
            image = Image.open(img_name).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image not found at {img_name}")
            return None, None # Handle missing images gracefully

        # Get the label. Assumes CSV has a 'label' column (e.g., 'benign'/'malignant').
        label_str = self.image_metadata.loc[idx, 'label']
        # Convert string label to integer (e.g., benign=0, malignant=1)
        label = 0 if label_str == 'benign' else 1
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Example of how you might get transforms for the Mammography dataset
def get_mammo_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
