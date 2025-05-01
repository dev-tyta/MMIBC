import os
from PIL import Image
import numpy as np
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create a file handler
handler = logging.FileHandler('ultrasound_handling.log')
handler.setLevel(logging.INFO)


def convert_tmp_to_img(input_dir, output_dir):
    """
    Convert .tmp files to .png images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"Converting .bmp files in {input_dir} to .png format in {output_dir}")
    for class_name in ['benign', 'malignant']:
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Directory not found: {class_dir}")
            continue
            
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        logger.info(f"Processing class: {class_name}")
        for tmp_file in os.listdir(class_dir):
            logger.info(f"Processing file: {tmp_file}")
            if tmp_file.endswith('.bmp'):
                try:
                    logger.info(f"Converting {tmp_file} to PNG format")
                    # Load .tmp file (adjust based on actual format)
                    img_path = os.path.join(class_dir, tmp_file)
                    img = Image.open(img_path)  # Validate .tmp format
                    
                    # Save as PNG
                    output_path = os.path.join(output_class_dir, tmp_file.replace('.bmp', '.png'))
                    img.save(output_path)
                    print(f"Converted: {tmp_file} → {output_path}")
                except Exception as e:
                    logger.error(f"Error processing {tmp_file}: {str(e)}")

def check_img_shape(input_dir):
    """
    Check the shape of images in the directory.
    """
    for class_name in ['benign', 'malignant']:
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Directory not found: {class_dir}")
            continue
            
        logger.info(f"Checking images in class: {class_name}")
        for img_file in os.listdir(class_dir):
            if img_file.endswith('.png'):
                img_path = os.path.join(class_dir, img_file)
                try:
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    logger.info(f"Image {img_path} Image shape: {img_array.shape}")
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {str(e)}")


def organize_images_and_masks(source_dir, images_dir, masks_dir):
    """
    Move images and masks to separate directories while preserving class labels.
    
    Args:
        source_dir: Directory containing the original dataset with class folders
        images_dir: Target directory for all images, organized by class
        masks_dir: Target directory for all masks, organized by class
    """
    # Create main directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    logger.info(f"Organizing files from {source_dir} to {images_dir} and {masks_dir}")
    
    # Process each class directory
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
            
        # Create class directories in target locations
        class_images_dir = os.path.join(images_dir, class_name)
        class_masks_dir = os.path.join(masks_dir, class_name)
        os.makedirs(class_images_dir, exist_ok=True)
        os.makedirs(class_masks_dir, exist_ok=True)
        
        logger.info(f"Processing class: {class_name}")
        
        # Process files in this class directory
        for filename in os.listdir(class_path):
            src_path = os.path.join(class_path, filename)
            
            # Skip directories
            if os.path.isdir(src_path):
                continue
                
            # Check if it's a mask file
            if '_mask' in filename and (filename.endswith('.png') or filename.endswith('.jpg')):
                dst_path = os.path.join(class_masks_dir, filename)
                logger.info(f"Copying mask: {filename} to {class_name} masks")
                shutil.copy(src_path, dst_path)
            
            # Check if it's a regular image (but not a mask)
            elif (filename.endswith('.png') or filename.endswith('.jpg')) and '_mask' not in filename:
                dst_path = os.path.join(class_images_dir, filename)
                logger.info(f"Copying image: {filename} to {class_name} images")
                shutil.copy(src_path, dst_path)


if __name__ == "__main__":
    # convert_tmp_to_img("data/originals", "data/ultrasound-2")
    check_img_shape("data/ultrasound-2")
    check_img_shape("data/ultrasound/images")
    # organize_images_and_masks(
    #     "data/Dataset_BUSI_with_GT",
    #     "data/ultrasound/images",
    #     "data/ultrasound/masks"
    # )
