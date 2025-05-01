import os
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from skimage.restoration import denoise_nl_means
import logging
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_image(
    image_path: str, 
    output_dir: str, 
    target_size: int = 518, 
    denoise: bool = True,
    denoise_params: Dict = {
        'patch_size': 5,
        'patch_distance': 3,
        'h': 0.8,
        'fast_mode': True
    },
    display_random_images: bool = False,
    display_probability: float = 0.1
) -> Optional[str]:
    """
    Preprocess an image: denoise (optional), resize preserving aspect ratio, and pad to square.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save processed image
        target_size: Target size for the output square image
        denoise: Whether to apply non-local means denoising
        denoise_params: Parameters for non-local means denoising
        display_random_images: Whether to display random images during processing
        display_probability: Probability of displaying an image
        
    Returns:
        Path to the saved image or None if processing failed
    """
    try:
        # Load image (BGR format by default)
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image {image_path}")
            return None
            
        # Processing in RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply denoising if requested
        if denoise:
            logger.debug(f"Applying denoising to {image_path}")
            img_denoised = denoise_nl_means(img, **denoise_params)
            img_denoised = (img_denoised * 255).astype(np.uint8)
            img_denoised = np.clip(img_denoised, 0, 255)
        else:
            img_denoised = img
        
        # Resize with aspect ratio preservation
        h, w = img_denoised.shape[:2]
        ratio = min(target_size/w, target_size/h)
        new_size = (int(w*ratio), int(h*ratio))
        resized = cv2.resize(img_denoised, new_size, interpolation=cv2.INTER_AREA)
        
        # Pad to square
        padded = _pad_to_square(resized, target_size)
        
        # Randomly display the image
        if display_random_images and random.random() < display_probability:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.subplot(1, 2, 2)
            plt.imshow(padded)
            plt.title("Processed Image")
            plt.show()
        
        # Save the processed image
        output_path = _get_output_path(image_path, output_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Processed {image_path} -> {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None
    
    
def _pad_to_square(image: np.ndarray, target_size: int) -> np.ndarray:
    """Pad image to a square with target_size dimensions."""
    h, w = image.shape[:2]
    delta_w = target_size - w
    delta_h = target_size - h
    
    return cv2.copyMakeBorder(
        image, 
        delta_h//2, delta_h - delta_h//2,
        delta_w//2, delta_w - delta_w//2,
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )

def _get_output_path(image_path: str, output_dir: str) -> str:
    """Create the output path keeping the class directory structure."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path {image_path} does not exist")
    
    class_name = os.path.basename(os.path.dirname(image_path))
    return os.path.join(output_dir, "images", class_name, os.path.basename(image_path))


def process_directory(
    input_dir: str, 
    output_dir: str,
    target_size: int = 518,
    denoise: bool = True,
    denoise_params: Dict = {
        'patch_size': 5,
        'patch_distance': 3,
        'h': 0.8,
        'fast_mode': True
    },
    display_random_images: bool = False,
    display_probability: float = 0.1,
    n_jobs: int = 4
) -> Dict:
    """
    Process all images in a directory containing label subdirectories
    
    Args:
        input_dir: Directory containing label subdirectories with images
        output_dir: Directory to save processed images
        target_size: Target size for the output square image
        denoise: Whether to apply non-local means denoising
        denoise_params: Parameters for non-local means denoising
        
    Returns:
        Dictionary with counts of processed and failed images by label
    """
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        raise ValueError(f"Invalid input directory: {input_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stats = {"processed": 0, "failed": 0, "by_label": {}}
    
    # Get all label directories
    label_dirs = [d for d in os.listdir(input_dir) 
                 if os.path.isdir(os.path.join(input_dir, d))]
    
    logger.info(f"Found {len(label_dirs)} label directories: {label_dirs}")
    
    for label in label_dirs:
        label_path = os.path.join(input_dir, label)
        stats["by_label"][label] = {"processed": 0, "failed": 0}
        
        # Process all images in the label directory
        image_files = [f for f in os.listdir(label_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        logger.info(f"Processing {len(image_files)} images in {label} category")
        
        # Process images in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(preprocess_image)(
                os.path.join(label_path, img_file), output_dir, target_size, denoise, denoise_params,
                display_random_images, display_probability
            ) for img_file in image_files
        )
        
        # Update stats based on the results
        for result in results:
            if result:
                stats["processed"] += 1
                stats["by_label"][label]["processed"] += 1
            else:
                stats["failed"] += 1
                stats["by_label"][label]["failed"] += 1
    
    return stats

def main():
    # Updated example usage for directory processing
    input_dir = "data/ultrasound/images"
    output_dir = "data/ultrasound/prepped"
    
    logger.info(f"Starting preprocessing of images from {input_dir}")
    stats = process_directory(
        input_dir, output_dir, display_random_images=False, n_jobs=8
    )
    
    logger.info(f"Processing complete. Total: {stats['processed']} processed, {stats['failed']} failed")
    for label, counts in stats["by_label"].items():
        logger.info(f"  {label}: {counts['processed']} processed, {counts['failed']} failed")

if __name__ == "__main__":
    main()